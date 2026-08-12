import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import TensorDataset, Subset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from typing import List, Tuple, Union, Sequence, Optional, Dict



def _safe_ratio(numerator: float, denominator: float) -> float:
    """Return numerator / denominator guarding against division by zero."""
    if denominator is None or denominator == 0:
        return np.nan
    try:
        if np.isnan(denominator):
            return np.nan
    except TypeError:
        pass
    return numerator / denominator
    

def _distribution_metrics(row: pd.Series) -> Dict[str, float]:
    """Compute Bowley skewness, Moors kurtosis, and tail asymmetry from quantiles."""
    # ┏━━━━━━━━━━ Quantiles required ━━━━━━━━━━┓
    q10 = row.get("q10")
    q50 = row.get("q50")
    q90 = row.get("q90")

    # ┏━━━━━━━━━━ Octiles required ━━━━━━━━━━┓
    oct2 = row.get("oct2")  # 0.25
    oct4 = row.get("oct4")  # 0.50
    oct6 = row.get("oct6")  # 0.75
    oct1 = row.get("oct1")  # 0.125
    oct3 = row.get("oct3")  # 0.375
    oct5 = row.get("oct5")  # 0.625
    oct7 = row.get("oct7")  # 0.875
    
    # ┏━━━━━━━━━━ Bowley skewness: (Q3 + Q1 - 2*Q2)/(Q3 - Q1) ━━━━━━━━━━┓
    bowley_num = (oct6 if oct6 is not None else np.nan) + (oct2 if oct2 is not None else np.nan) - 2 * (oct4 if oct4 is not None else np.nan)
    bowley_den = (oct6 if oct6 is not None else np.nan) - (oct2 if oct2 is not None else np.nan)
    bowley_skewness = _safe_ratio(bowley_num, bowley_den)   
    
    # ┏━━━━━━━━━━ Moors kurtosis using octiles: ((O7 - O5) + (O3 - O1)) / (O6 - O2) ━━━━━━━━━━┓
    moors_num = (oct7 if oct7 is not None else np.nan) - (oct5 if oct5 is not None else np.nan)
    moors_num += (oct3 if oct3 is not None else np.nan) - (oct1 if oct1 is not None else np.nan)
    moors_den = (oct6 if oct6 is not None else np.nan) - (oct2 if oct2 is not None else np.nan)
    moors_kurtosis = _safe_ratio(moors_num, moors_den)
   
    # ┏━━━━━━━━━━ Tail asymmetry using 10th/90th percentiles (or octiles if missing) ━━━━━━━━━━┓
    med = q50 if q50 is not None else oct4
    upper = q90 if q90 is not None else oct7
    lower = q10 if q10 is not None else oct1
    tail_num = (upper if upper is not None else np.nan) + (lower if lower is not None else np.nan) - 2 * (med if med is not None else np.nan)
    tail_den = (upper if upper is not None else np.nan) - (lower if lower is not None else np.nan)
    tail_asymmetry = _safe_ratio(tail_num, tail_den)
    
    return {"bowley_skewness": bowley_skewness,
            "moors_kurtosis": moors_kurtosis,
            "tail_asymmetry": tail_asymmetry}


def merge_meta_targets(asset_type: str,
                       asset: str,
                       data_dir: str,
                       output_dir: str = None,
                       set_index: bool = True,
                       column_features: Optional[Sequence[str]] = None,
                       context_features: Optional[Sequence[str]] = None,
                       meta_label_mode: str = "original") -> pd.DataFrame:

    """Merge DOWN/UP meta-target files keeping only requested features."""
    
    # ┏━━━━━━━━━━ Requested Column and Context Features ━━━━━━━━━━┓
    column_features = list(column_features or ['close'])
    context_features = list(context_features or [])
    requested = list(dict.fromkeys(column_features + context_features))

    # ┏━━━━━━━━━━ Asset & Paths ━━━━━━━━━━┓
    sym = asset.upper()
    down_path = Path(data_dir) / f"{sym}_down.csv"
    up_path = Path(data_dir) / f"{sym}_up.csv"

    # ┏━━━━━━━━━━ Reading M1 Predictions ━━━━━━━━━━┓
    df_down = pd.read_csv(down_path, parse_dates=['date']).set_index('date')
    df_up = pd.read_csv(up_path, parse_dates=['date']).set_index('date')

    # ┏━━━━━━━━━━ Suffix (TP or FP) from Meta-Labels ━━━━━━━━━━┓
    mode = (meta_label_mode or "original").lower()
    suffix = "FP" if mode == "fp" else "TP"
    down_target = f"is{suffix}_DN"
    up_target = f"is{suffix}_UP"

    rename_down = {'meta_label':    f'is{suffix}_DN',
                   'meta_target':   f'is{suffix}_DN',
                   'pred':          'm1_dn',
                   'prediction':    'm1_prediction',
                   'pred_proba':    'm1_pred_proba_dn',
                   'lab':           'lab_dn'}

    rename_up = {'meta_label':      f'is{suffix}_UP',
                 'meta_target':     f'is{suffix}_UP',
                 'pred':            'm1_up',
                 'pred_proba':      'm1_pred_proba_up',
                 'lab':             'lab_up'}

    # ┏━━━━━━━━━━ Renaming Columns ━━━━━━━━━━┓
    df_down = df_down.rename(columns = rename_down)
    df_up = df_up.rename(columns = rename_up)

    # ┏━━━━━━━━━━ Normalizer Function (False -> Nan) ━━━━━━━━━━┓
    def _normalise_meta(series: pd.Series) -> pd.Series:
        replacements = {False: np.nan}
        return series.replace(replacements)
    
    # ┏━━━━━━━━━━ Normalizing ━━━━━━━━━━┓
    if down_target in df_down.columns:
        df_down[down_target] = _normalise_meta(df_down[down_target])
    if up_target in df_up.columns:
        df_up[up_target] = _normalise_meta(df_up[up_target])
    
    # ┏━━━━━━━━━━ Additional Columns ━━━━━━━━━━┓
    #for extra_col in ("lab_dn", "lab_up", "m1_dn", "m1_up"):
    for extra_col in ("lab_dn", "lab_up"):
        if extra_col not in requested:
            requested.append(extra_col)

    # ┏━━━━━━━━━━ Statistical Features coming from Chronos & Possible Features to compute from them ━━━━━━━━━━┓
    quantile_cols = ["q10", "q50", "q90", "oct1", "oct2", "oct3", "oct4", "oct5", "oct6", "oct7"]
    metrics_targets = ["bowley_skewness", "moors_kurtosis", "tail_asymmetry"]
    metrics_requested = [metric for metric in metrics_targets if metric in requested]

    if metrics_requested:
        missing_quantiles = [q for q in quantile_cols if q not in df_down.columns]
        if missing_quantiles:
            print(f"[merge_meta_targets] WARNING: missing quantiles {missing_quantiles} for metrics computation")
        else:
            metrics_df = df_down[quantile_cols].apply(_distribution_metrics, axis=1).apply(pd.Series)
            df_down = pd.concat([df_down, metrics_df[metrics_requested]], axis=1)
    
    # ┏━━━━━━━━━━ Drop quantiles unless explicitly requested ━━━━━━━━━━┓
    keep_quantiles = set(quantile_cols) & set(requested)
    drop_quantiles = [q for q in quantile_cols if q in df_down.columns and q not in keep_quantiles]
    if drop_quantiles:
        df_down = df_down.drop(columns = drop_quantiles)

    # ┏━━━━━━━━━━ Creation of Empty DataFrame ━━━━━━━━━━┓
    idx = df_down.index.union(df_up.index).sort_values()
    merged = pd.DataFrame(index = idx)

    def pull(column: str) -> pd.Series:
        if column in df_down.columns:
            return df_down[column]
        if column in df_up.columns:
            return df_up[column]
        return pd.Series(pd.NA, index = idx)

    # ┏━━━━━━━━━━ Merging both CSVs ━━━━━━━━━━┓
    for column in requested:
        merged[column] = pull(column).reindex(idx)

    # ┏━━━━━━━━━━ Adding Meta-Labels in Merged file & Resetting index ━━━━━━━━━━┓
    merged[down_target] = pd.to_numeric(pull(down_target).reindex(idx), errors = 'coerce').astype('float')
    merged[up_target] = pd.to_numeric(pull(up_target).reindex(idx), errors = 'coerce').astype('float')
    
    # ┏━━━━━━━━━━ Drop quantiles unless explicitly requested ━━━━━━━━━━┓
    merged = merged.reset_index().rename(columns = {'index': 'date'})
    final_order = ['date'] + column_features + context_features + ['lab_dn', 'lab_up', down_target, up_target]
    #final_order = ['date'] + column_features + context_features + ['lab_dn', 'lab_up', 'm1_dn', 'm1_up', down_target, up_target]
    merged = merged.loc[:, [col for col in final_order if col in merged.columns]]
    
    if set_index:
        merged = merged.set_index('date')
    
    # ┏━━━━━━━━━━ Saving Merged CSV ━━━━━━━━━━┓
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        merged.to_csv(Path(output_dir) / f"{sym}_merge.csv")
    
    return merged


def count_meta_targets(df: pd.DataFrame,
                       columns: Sequence[str] = ("isTP_UP", "isTP_DN"),
                       task: Optional[str] = None) -> Dict[str, Dict[str, int]]:
    """
    Summarise class counts for the directional meta targets alongside lab labels.

    Parameters
    ----------
    df : pd.DataFrame
        Merged dataset produced by `merge_meta_targets`.
    columns : Sequence[str]
        Which meta-target columns to tally.
    task : Optional[str]
        Active task name ('UP' or 'DN'); when provided, the corresponding lab columns
        are prioritised (but both `lab_up` and `lab_dn` are tallied when available).

    Returns
    -------
    Dict[str, Dict[str, int]]
        Nested dictionary mapping column → {class_value: count}.
    """
    # ┏━━━━━━━━━━ Function to count ━━━━━━━━━━┓
    def _serialise_counts(series: pd.Series) -> Dict[str, int]:
        vc = series.value_counts(dropna = False)
        col_counts: Dict[str, int] = {}
        for value, freq in vc.items():
            if pd.isna(value):
                key = "nan"
            elif isinstance(value, (int, np.integer)):
                key = str(int(value))
            elif isinstance(value, float) and value.is_integer():
                key = str(int(value))
            else:
                key = str(value)
            col_counts[key] = int(freq)
        return col_counts

    # ┏━━━━━━━━━━ Meta-label counts ━━━━━━━━━━┓
    counts: Dict[str, Dict[str, int]] = {}
    for col in columns:
        if col not in df.columns:
            counts[col] = {}
            continue
        counts[col] = _serialise_counts(df[col])

    # ┏━━━━━━━━━━ Lab counts ━━━━━━━━━━┓
    task_upper = (task or "").upper()
    if task_upper == "DN":
        label_candidates = ("lab_dn", "lab_up")
    elif task_upper == "UP":
        label_candidates = ("lab_up", "lab_dn")
    else:
        label_candidates = ("lab_up", "lab_dn")

    for label_col in label_candidates:
        if label_col not in df.columns:
            continue
        counts[label_col] = _serialise_counts(df[label_col])

    return counts


def prepare_dataset(df: pd.DataFrame,
                    seq_len: int = 90,
                    column_features: Sequence[str] = ("close",),
                    context_features: Sequence[str] = (),
                    meta_label_mode: str = "original",
                    task: Optional[str] = None) -> TensorDataset:
    """
    Transform the merged dataframe into a TensorDataset of sliding windows plus targets.

    Parameters
    ----------
    df : pd.DataFrame
        Merged dataset produced by `merge_meta_targets`.
    seq_len : int
        Number of timesteps per window (before appending context features).
    column_features : Sequence[str]
        Feature columns used to build the time series windows.
    context_features : Sequence[str]
        Contextual features appended at the end of each window (broadcast per channel).

    Returns
    -------
    TensorDataset
        Dataset of shape ((N, C, seq_len + len(context_features)), (N,), (N,))
        containing inputs, UP targets, and DN targets respectively.
    """
    df = df.copy()
    n = len(df)
    N = n - seq_len + 1
    C = len(column_features)
    L = seq_len + len(context_features)
    suffix = "FP" if (meta_label_mode or "original").lower() == "fp" else "TP"
    up_col = f"is{suffix}_UP"
    dn_col = f"is{suffix}_DN"

    # ┏━━━━━━━━━━ 0) Sanity checks ━━━━━━━━━━┓
    for col in column_features:
        if col not in df.columns:
            raise KeyError(f"Feature '{col}' not found in DataFrame columns.")
    if n < seq_len:
        raise ValueError(f"Not enough rows ({n}) for seq_len={seq_len}")
    if up_col not in df.columns or dn_col not in df.columns:
        raise KeyError(f"Expected columns '{up_col}' and '{dn_col}' in dataframe.")
    
    # ┏━━━━━━━━━━ 1) Targets (preserve NaNs) ━━━━━━━━━━┓
    y_up = df[up_col].to_numpy(dtype=float)
    y_dn = df[dn_col].to_numpy(dtype=float)

    # ┏━━━━━━━━━━ 2) Raw values matrix for features (no global scaling!) ━━━━━━━━━━┓
    feats = df[list(column_features)].to_numpy(dtype=np.float32)  # shape: (n_rows, C)
    if context_features:    
        context_vals = df[list(context_features)].to_numpy(dtype=np.float32)
    else:
        context_vals = np.zeros((n, 0), dtype=np.float32)
    
    # ┏━━━━━━━━━━ 3) Allocate outputs ━━━━━━━━━━┓
    X_list: List[np.ndarray] = []
    Y_up_list: List[int] = []
    Y_dn_list: List[int] = []
    task_upper = task.upper()
    
    # ┏━━━━━━━━━━ 4) Build sliding windows with MinMax scaling per window and feature ━━━━━━━━━━┓
    # For each window [i:i+seq_len), scale each feature using min/max computed only on that window.
    for i in range(N):
        start, end     = i, i + seq_len          # 0,90 -> 1,91 -> ...
        window = feats[start:end, :]             # (seq_len, C)

        # ┏━━━━━━━━━━ Per-feature min/max within the window ━━━━━━━━━━┓
        w_min = window.min(axis=0)               # (C,)
        w_max = window.max(axis=0)               # (C,)
        diff = (w_max - w_min)
        
        # ┏━━━━━━━━━━ Avoid div-by-zero: if constant feature inside the window → all zeros after scaling ━━━━━━━━━━┓
        # (You can also set to 0.5, but zeros are fine and stable for CNN/RevIN.)
        diff[diff == 0.0] = 1.0

        # ┏━━━━━━━━━━ MinMax Scaling ━━━━━━━━━━┓
        w_scaled = (window - w_min) / diff       # (seq_len, C)
        w_scaled = w_scaled.T                    # (C, seq_len)

        # ┏━━━━━━━━━━ Get context features at window end (1D vector) ━━━━━━━━━━┓
        context_vector = context_vals[end - 1, :]
        context_expanded = np.tile(context_vector, (C, 1)) if context_features else np.zeros((C, 0), dtype=np.float32)

        # ┏━━━━━━━━━━ Concatenate along time axis (dim=1) ━━━━━━━━━━┓
        full_input = np.concatenate([w_scaled, context_expanded], axis=1) if context_features else w_scaled

        label_up = y_up[end - 1]
        label_dn = y_dn[end - 1]
        
        # ┏━━━━━━━━━━ Appending context and raw labels (NaNs preserved) ━━━━━━━━━━┓
        X_list.append(full_input)
        Y_up_list.append(label_up)
        Y_dn_list.append(label_dn)

    # ┏━━━━━━━━━━ CReating Dataset Tensors ━━━━━━━━━━┓
    X = np.stack(X_list).astype(np.float32)
    Y_up = np.asarray(Y_up_list, dtype=np.float32)
    Y_dn = np.asarray(Y_dn_list, dtype=np.float32)
    dataset = TensorDataset(torch.from_numpy(X),
                            torch.from_numpy(Y_up),
                            torch.from_numpy(Y_dn))

    # ┏━━━━━━━━━━ Adding Timestamp for each Label  ━━━━━━━━━━┓
    dataset.window_dates = list(df.index[seq_len - 1:])

    return dataset
                         

class FocalLoss(nn.Module):
    """Numerically stable focal loss supporting binary and multi-class logits."""
    def __init__(self, gamma: float = 2.5, alpha: float = 0.25, reduction: str = "mean"):
        """
        gamma: focusing parameter
        alpha: class-balance weight for the positive class
        reduction: 'none' | 'mean' | 'sum'
        """
        super().__init__()
        self.gamma    = gamma
        self.alpha    = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        logits: shape (N, C) for multi-class or (N,) for binary (pre-activation)
        targets: shape (N,) with {0,1} for binary or class indices for multi-class
        """
        if logits.dim() > 1:
            # ┏━━━━━━━━━━ Multi-class ━━━━━━━━━━┓
            probs   = F.softmax(logits, dim=1)
            p_t     = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            # ┏━━━━━━━━━━ Per-sample alpha_t ━━━━━━━━━━┓
            alpha_t = torch.where(targets == 1,
                                  self.alpha,
                                  1 - self.alpha).to(logits.device)
            # ┏━━━━━━━━━━ Per-sample cross-entropy (no reduction) ━━━━━━━━━━┓
            ce      = F.nll_loss(torch.log(probs),
                                 targets,
                                 reduction="none")
        else:
            # ┏━━━━━━━━━━ Binary ━━━━━━━━━━┓
            probs   = torch.sigmoid(logits)
            p_t     = probs * targets + (1 - probs) * (1 - targets)
            alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
            # ┏━━━━━━━━━━ Per-sample BCE (no reduction) ━━━━━━━━━━┓
            ce      = F.binary_cross_entropy_with_logits(logits,
                                                         targets.float(),
                                                         reduction="none")

        # ┏━━━━━━━━━━ Focal factor & Final per-sample loss ━━━━━━━━━━┓
        focal_factor = (1 - p_t).pow(self.gamma)
        loss = alpha_t * focal_factor * ce

        # ┏━━━━━━━━━━ Reduce  ━━━━━━━━━━┓
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss  # 'none'


def make_criteria(loss_type: str,
                  w_up: torch.Tensor,
                  w_dn: torch.Tensor,
                  device: torch.device,
                  focal_gamma: float = 2.5,
                  focal_alpha: float = 0.25) -> Tuple[nn.Module, nn.Module]:
    """
    Instantiate the pair of training criteria for UP and DN heads.

    Parameters
    ----------
    loss_type : str
        One of {'bce', 'cross_entropy', 'focal'} determining the criterion style.
    w_up, w_dn : torch.Tensor
        Class-imbalance weights computed from the training data.
    device : torch.device
        Target device for the instantiated loss modules.
    focal_gamma : float
        Focusing parameter when using the focal loss variant.
    focal_alpha : float
        Optional override for the positive-class weight in the focal loss.

    Returns
    -------
    Tuple[nn.Module, nn.Module]
        `(crit_up, crit_dn)` ready to train each directional head.
    """
    # ┏━━━━━━━━━━ BCE ━━━━━━━━━━┓
    if loss_type == 'bce':
        crit_up = nn.BCEWithLogitsLoss(pos_weight = w_up)
        crit_dn = nn.BCEWithLogitsLoss(pos_weight = w_dn)

    # ┏━━━━━━━━━━ Cross‐Entropy ━━━━━━━━━━┓
    elif loss_type == 'cross_entropy':
        crit_up = nn.CrossEntropyLoss(weight = w_up)
        crit_dn = nn.CrossEntropyLoss(weight = w_dn)

    # ┏━━━━━━━━━━ Focal Loss ━━━━━━━━━━┓
    elif loss_type == 'focal':
        # ┏━━━━━━━━━━ Select the positive‐class weight (index 1) if vector, else tensor itself ━━━━━━━━━━┓
        up_pos = w_up[1] if w_up.numel() > 1 else w_up
        dn_pos = w_dn[1] if w_dn.numel() > 1 else w_dn
        total = up_pos + dn_pos
        # ┏━━━━━━━━━━ Normalize into scalars in [0,1]  ━━━━━━━━━━┓
        alpha_up = (up_pos/total).item() if focal_alpha is None else focal_alpha
        alpha_dn = (dn_pos/total).item() if focal_alpha is None else focal_alpha
        
        # ┏━━━━━━━━━━ Criterion UP ━━━━━━━━━━┓
        crit_up = FocalLoss(
            gamma = focal_gamma,
            alpha = alpha_up,
            reduction = 'mean')

        # ┏━━━━━━━━━━ Criterion DN ━━━━━━━━━━┓
        crit_dn = FocalLoss(
            gamma = focal_gamma,
            alpha = alpha_dn,
            reduction = 'mean')
    else:
        raise ValueError(f"Unknown loss_type: {loss_type!r}")

    return crit_up.to(device), crit_dn.to(device)


def build_loaders(ds: TensorDataset,
                  cross_validation: bool,
                  target:           str,
                  props:            float,
                  train_frac:       float,
                  val_frac:         float,
                  test_frac:        float,
                  batch_size:       int,
                  loss_type:        str,
                  focal_gamma:      float,
                  focal_alpha:      float,
                  device:           torch.device) -> Tuple[List[Tuple[DataLoader, DataLoader, nn.Module]], DataLoader]:
    """
    Construct training/validation/test DataLoaders and the associated criteria.

    Parameters
    ----------
    ds : TensorDataset
        Dataset produced by `prepare_dataset`.
    cross_validation : bool
        Whether to iterate through rolling validation folds.
    target : str
        Active task ('UP' or 'DN') selecting the criterion to return.
    props : float
        Proportion controlling validation window position when using CV.
    train_frac, val_frac, test_frac : float
        Fractions of the dataset assigned to each split.
    batch_size : int
        Batch size for DataLoaders.
    loss_type : str
        Loss type forwarded to `make_criteria`.
    focal_gamma, focal_alpha : float
        Focal loss hyperparameters.
    device : torch.device
        Device where the criteria should be moved.

    Returns
    -------
    Tuple[List[Tuple[DataLoader, DataLoader, nn.Module]], DataLoader]
        Pair containing the folds (each with its own validation loader and criterion)
        and the held-out test loader.
    """

    N = len(ds)

    # ┏━━━━━━━━━━ Compute split values ━━━━━━━━━━┓
    n_train = int(train_frac * N)
    n_val   = int(val_frac   * N)
    n_test  = N - n_train - n_val

    # ┏━━━━━━━━━━ Compute split indices values ━━━━━━━━━━┓
    idx_train = list(range(0, n_train))
    idx_val   = list(range(n_train, n_train + n_val))
    idx_test  = list(range(n_train + n_val, N))

    # ┏━━━━━━━━━━ Extracting Target Tensor ━━━━━━━━━━┓
    target_upper = target.upper()
    target_tensor = ds.tensors[1] if target_upper == "UP" else ds.tensors[2]
    date_index = getattr(ds, "window_dates", None)

    # ┏━━━━━━━━━━ Helper Functions to remove NaN Values ━━━━━━━━━━┓
    # ┏━━━━━━━━━━ 1. Indices without NaN Values ━━━━━━━━━━┓
    def _drop_nan_from_indices(indices: List[int]) -> List[int]:
        if not indices:
            return indices
        idx_tensor = torch.tensor(indices, dtype=torch.long)
        rel = target_tensor[idx_tensor]
        if rel.dtype.is_floating_point:
            keep = ~torch.isnan(rel)
            return idx_tensor[keep].tolist()
        return indices

    # ┏━━━━━━━━━━ 2. Labels without NaN Values ━━━━━━━━━━┓
    def _clean_labels(labels: torch.Tensor) -> torch.Tensor:
        if labels.dtype.is_floating_point:
            mask = ~torch.isnan(labels)
            labels = labels[mask]
        return labels

    # ┏━━━━━━━━━━ 3. Printing Time Ranges for each Split ━━━━━━━━━━┓
    def _log_range(name: str, indices: List[int]) -> None:
        if not indices:
            print(f"[build_loaders] {name}: no samples")
            return
        if date_index and indices[-1] < len(date_index):
            first = date_index[indices[0]]
            last = date_index[indices[-1]]
            print(f"[build_loaders] {name}: {first} → {last} ({len(indices)} samples)")
        else:
            print(f"[build_loaders] {name}: idx {indices[0]} → {indices[-1]} ({len(indices)} samples)")

    # ┏━━━━━━━━━━ New Indices (w/o Nans) ━━━━━━━━━━┓
    idx_train_clean = _drop_nan_from_indices(idx_train)
    idx_val_clean = _drop_nan_from_indices(idx_val)
    idx_test_clean = _drop_nan_from_indices(idx_test)

    # ┏━━━━━━━━━━ Warning Check ━━━━━━━━━━┓
    if not idx_train_clean:
        raise ValueError(f"No training samples remain after filtering NaNs for target '{target_upper}'.")
    if not idx_val_clean:
        raise ValueError(f"No validation samples remain after filtering NaNs for target '{target_upper}'.")
    if not idx_test_clean:
        raise ValueError(f"No test samples remain after filtering NaNs for target '{target_upper}'.")

    val_loader_outer  = DataLoader(Subset(ds, idx_val_clean),
                                   batch_size = batch_size,
                                   shuffle    = False)

    test_loader       = DataLoader(Subset(ds, idx_test_clean),
                                   batch_size = batch_size,
                                   shuffle    = False)

    # ┏━━━━━━━━━━ Seed for all train‐set shuffles ━━━━━━━━━━┓
    _seed = 1493583942
    folds = []

    # expose surviving test indices for downstream exporters
    ds.test_indices = idx_test_clean
    
    if not cross_validation:
        # ┏━━━━━━━━━━ Single Fold (No CV) ━━━━━━━━━━┓
        # ┏━━━━━━━━━━ To print Time-Lines ━━━━━━━━━━┓
        #_log_range("Train split (post NaN filter)", idx_train_clean)
        #_log_range("Validation split (post NaN filter)", idx_val_clean)
        #_log_range("Test split", idx_test_clean)

        # ┏━━━━━━━━━━ Non-NaN Labels ━━━━━━━━━━┓
        y_up_tr = _clean_labels(ds.tensors[1][idx_train_clean])
        y_dn_tr = _clean_labels(ds.tensors[2][idx_train_clean])
        
        # ┏━━━━━━━━━━ Compute class-weights on TRAIN only ━━━━━━━━━━┓
        if loss_type in ('cross_entropy', 'focal'):
            # ┏━━━━━━━━━━ Compute class-weights on UP & DN ━━━━━━━━━━┓
            # Lets the training loop keep going even when one split 
            # temporarily lacks valid labels, instead of crashing on bincount
            cu = torch.bincount(y_up_tr.long(), minlength=2).float() if y_up_tr.numel() else torch.ones(2)
            cd = torch.bincount(y_dn_tr.long(), minlength=2).float() if y_dn_tr.numel() else torch.ones(2)
            tu, td = cu.sum(), cd.sum()
            w_up = tu / (cu + 1e-8)
            w_dn = td / (cd + 1e-8)
        
        elif loss_type == 'bce':
            # ┏━━━━━━━━━━ Compute class-weights on UP ━━━━━━━━━━┓
            if y_up_tr.numel():
                pu = y_up_tr.sum().float()
                nu = y_up_tr.numel() - pu
                w_up = nu / (pu + 1e-8)
            else:
                w_up = torch.tensor(1.0)
                if target_upper == "UP":
                    print("[build_loaders] No valid UP labels in train split; using neutral class weight 1.0.")
            
            # ┏━━━━━━━━━━ Compute class-weights on DN ━━━━━━━━━━┓
            if y_dn_tr.numel():
                pd = y_dn_tr.sum().float()
                nd = y_dn_tr.numel() - pd
                w_dn = nd / (pd + 1e-8)
            else:
                w_dn = torch.tensor(1.0)
                if target_upper == "DN":
                    print("[build_loaders] No valid DN labels in train split; using neutral class weight 1.0.")
        else:
            raise ValueError("loss_type must be 'bce', 'cross_entropy' or 'focal")

        w_up = w_up.to(device)
        w_dn = w_dn.to(device)

        # ┏━━━━━━━━━━ Build a standard shuffled DataLoader (New Indices) ━━━━━━━━━━┓
        train_loader = DataLoader(Subset(ds, idx_train_clean),
                                  batch_size = batch_size,
                                  shuffle    = True,
                                  generator  = torch.Generator().manual_seed(_seed))

        # ┏━━━━━━━━━━ Build both criteria, then pick the one for target ━━━━━━━━━━┓
        crit_up, crit_dn = make_criteria(loss_type, 
                                         w_up, 
                                         w_dn, 
                                         device, 
                                         focal_gamma, 
                                         focal_alpha)

        criterion = crit_up if target.upper() == 'UP' else crit_dn

        # ┏━━━━━━━━━━ Append datasets and criterion ━━━━━━━━━━┓
        folds.append((train_loader, val_loader_outer, criterion))

    else:
        # ┏━━━━━━━━━━ Multiple Folds (CV) ━━━━━━━━━━┓
        p_start = props 
        n_pool = n_train + n_val
        train_end = int(n_pool * p_start)
        val_window = n_val
        
        # ┏━━━━━━━━━━ To print Test Time-Line ━━━━━━━━━━┓
        #_log_range("Test split", idx_test_clean)

        fold_id = 1
        while train_end + val_window <= n_pool:
            # ┏━━━━━━━━━━ Original Indices (with Nans) ━━━━━━━━━━┓
            idx_train_fold = list(range(0, train_end))
            idx_val_fold   = list(range(train_end, train_end + val_window))

            # ┏━━━━━━━━━━ New Indices (w/o Nans) ━━━━━━━━━━┓
            idx_train_fold = _drop_nan_from_indices(idx_train_fold)
            idx_val_fold   = _drop_nan_from_indices(idx_val_fold)

            # ┏━━━━━━━━━━ Warning Check ━━━━━━━━━━┓
            if not idx_train_fold:
                raise ValueError(f"No training samples remain for fold (end={train_end}) after filtering NaNs.")
            if not idx_val_fold:
                raise ValueError(f"No validation samples remain for fold (end={train_end}) after filtering NaNs.")

            # ┏━━━━━━━━━━ Sanity Check ━━━━━━━━━━┓
            assert max(idx_train_fold) <= min(idx_val_fold), "Train/Val overlap!"
            assert max(idx_val_fold)   < n_pool, "Val fold exceeds pool boundary!"

            # ┏━━━━━━━━━━ To Print per-Fold Time-Line in Train & Val ━━━━━━━━━━┓
            #(f"CV Fold {fold_id} Train (post NaN filter)", idx_train_fold)
            #_log_range(f"CV Fold {fold_id} Val (post NaN filter)", idx_val_fold)

            # ┏━━━━━━━━━━ Extract targets on Train only ━━━━━━━━━━┓
            y_up_tr = _clean_labels(ds.tensors[1][idx_train_fold])
            y_dn_tr = _clean_labels(ds.tensors[2][idx_train_fold])

            # ┏━━━━━━━━━━ Compute class-weights on TRAIN Fold only ━━━━━━━━━━┓
            if loss_type in ('cross_entropy', 'focal'):
                # ┏━━━━━━━━━━ Compute class-weights on UP & DN ━━━━━━━━━━┓
                cu = torch.bincount(y_up_tr.long(), minlength=2).float() if y_up_tr.numel() else torch.ones(2)
                cd = torch.bincount(y_dn_tr.long(), minlength=2).float() if y_dn_tr.numel() else torch.ones(2)
                tu, td = cu.sum(), cd.sum()
                w_up = tu / (cu + 1e-8)
                w_dn = td / (cd + 1e-8)

            elif loss_type == 'bce':
                # ┏━━━━━━━━━━ Compute class-weights on UP ━━━━━━━━━━┓
                if y_up_tr.numel():
                    pu = y_up_tr.sum().float()
                    nu = y_up_tr.numel() - pu
                    w_up = nu / (pu + 1e-8)
                else:
                    w_up = torch.tensor(1.0)
                    if target_upper == "UP":
                        print("[build_loaders][CV] No valid UP labels for fold; using neutral class weight 1.0.")

                # ┏━━━━━━━━━━ Compute class-weights on DN ━━━━━━━━━━┓
                if y_dn_tr.numel():
                    pd = y_dn_tr.sum().float()
                    nd = y_dn_tr.numel() - pd
                    w_dn = nd / (pd + 1e-8)
                else:
                    w_dn = torch.tensor(1.0)
                    if target_upper == "DN":
                        print("[build_loaders][CV] No valid DN labels for fold; using neutral class weight 1.0.")
            else:
                raise ValueError("loss_type must be 'bce', 'cross_entropy' or 'focal")

            w_up = w_up.to(device)
            w_dn = w_dn.to(device)

            # ┏━━━━━━━━━━ Standard shuffled DataLoader for this fold ━━━━━━━━━━┓
            train_loader = DataLoader(Subset(ds, idx_train_fold),
                                      batch_size = batch_size,
                                      shuffle    = True,
                                      generator  = torch.Generator().manual_seed(_seed))
            
            val_loader   = DataLoader(Subset(ds, idx_val_fold),
                                      batch_size = batch_size,
                                      shuffle    = False)

            # ┏━━━━━━━━━━ Build both criteria, then pick the one for target ━━━━━━━━━━┓
            crit_up, crit_dn = make_criteria(loss_type, 
                                             w_up, 
                                             w_dn, 
                                             device, 
                                             focal_gamma, 
                                             focal_alpha)

            criterion = crit_up if target.upper() == 'UP' else crit_dn

            # ┏━━━━━━━━━━ Append datasets and criterion ━━━━━━━━━━┓
            folds.append((train_loader, val_loader, criterion))

            # ┏━━━━━━━━━━ Expand training window ━━━━━━━━━━┓
            train_end += val_window 
            fold_id += 1
        
        # ┏━━━━━━━━━━ Handle leftover samples in the pool ━━━━━━━━━━┓
        if train_end < n_pool:
            # ┏━━━━━━━━━━ Last fold should mirror original split sizes ━━━━━━━━━━┓
            idx_tr = _drop_nan_from_indices(list(range(0, n_train)))
            idx_va = _drop_nan_from_indices(list(range(n_train, n_train + n_val)))

            # ┏━━━━━━━━━━ Safety Check ━━━━━━━━━━┓
            if not idx_tr:
                raise ValueError("No training samples remain for the final fold after filtering NaNs.")
            if not idx_va:
                raise ValueError("No validation samples remain for the final fold after filtering NaNs.")

            # ┏━━━━━━━━━━ To print last fold Train & Val ━━━━━━━━━━┓
            #_log_range(f"CV Fold {fold_id} Train (post NaN filter)", idx_tr)
            #_log_range(f"CV Fold {fold_id} Val (post NaN filter)", idx_va)

            # ┏━━━━━━━━━━ Extract clean targets on last train set ━━━━━━━━━━┓
            y_up_tr = _clean_labels(ds.tensors[1][idx_tr])
            y_dn_tr = _clean_labels(ds.tensors[2][idx_tr])

            # ┏━━━━━━━━━━ Compute class-weights on TRAIN Fold only ━━━━━━━━━━┓
            if loss_type in ('cross_entropy', 'focal'):
                # ┏━━━━━━━━━━ Compute class-weights on UP & DN ━━━━━━━━━━┓
                cu = torch.bincount(y_up_tr.long(), minlength=2).float() if y_up_tr.numel() else torch.ones(2)
                cd = torch.bincount(y_dn_tr.long(), minlength=2).float() if y_dn_tr.numel() else torch.ones(2)
                tu, td = cu.sum(), cd.sum()
                w_up = tu / (cu + 1e-8)
                w_dn = td / (cd + 1e-8)

            elif loss_type == 'bce':
                # ┏━━━━━━━━━━ Compute class-weights on UP ━━━━━━━━━━┓
                if y_up_tr.numel():
                    pu = y_up_tr.sum().float()
                    nu = y_up_tr.numel() - pu
                    w_up = nu / (pu + 1e-8)
                else:
                    w_up = torch.tensor(1.0)
                    if target_upper == "UP":
                        print("[build_loaders][CV-final] No valid UP labels for final fold; using neutral class weight 1.0.")
                
                # ┏━━━━━━━━━━ Compute class-weights on DN ━━━━━━━━━━┓
                if y_dn_tr.numel():
                    pd = y_dn_tr.sum().float()
                    nd = y_dn_tr.numel() - pd
                    w_dn = nd / (pd + 1e-8)
                else:
                    w_dn = torch.tensor(1.0)
                    if target_upper == "DN":
                        print("[build_loaders][CV-final] No valid DN labels for final fold; using neutral class weight 1.0.")
            else:
                raise ValueError("loss_type must be 'bce', 'cross_entropy' or 'focal")

            w_up = w_up.to(device)
            w_dn = w_dn.to(device)

            # ┏━━━━━━━━━━ Standard shuffled DataLoader for last fold ━━━━━━━━━━┓
            train_loader = DataLoader(Subset(ds, idx_tr),
                                      batch_size = batch_size,
                                      shuffle    = True,
                                      generator  = torch.Generator().manual_seed(_seed))

            val_loader   = DataLoader(Subset(ds, idx_va),
                                      batch_size = batch_size,
                                      shuffle    = False)

            # ┏━━━━━━━━━━ Build both criteria, then pick the one for target ━━━━━━━━━━┓
            crit_up, crit_dn = make_criteria(loss_type, 
                                             w_up, 
                                             w_dn, 
                                             device, 
                                             focal_gamma, 
                                             focal_alpha)

            criterion = crit_up if target.upper() == 'UP' else crit_dn

            # ┏━━━━━━━━━━ Append datasets and criterion ━━━━━━━━━━┓
            folds.append((train_loader, val_loader, criterion))

    return folds, test_loader
