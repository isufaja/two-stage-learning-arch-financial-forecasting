import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from ..paths import dataset_path
from typing import Optional
from sklearn.metrics import (confusion_matrix,
                             ConfusionMatrixDisplay,
                             accuracy_score,
                             precision_score,
                             recall_score,
                             f1_score,
                             fbeta_score)

from .train_utils import compute_basic_metrics

def plot_cm_with_metrics(preds,
                         targets,
                         labels,
                         title,
                         out_dir,
                         best_threshold,
                         cmap="Oranges"):
    """
    Render a confusion matrix plus scalar metrics and dump it as PNG.
    """
    # ┏━━━━━━━━━━ Aggregate confusion matrix & metrics ━━━━━━━━━━┓
    cm    = confusion_matrix(targets, preds)
    acc   = accuracy_score(targets, preds)
    prec  = precision_score(targets, preds, zero_division = 0)
    rec   = recall_score(targets, preds)
    f1    = f1_score(targets, preds)
    fbeta = fbeta_score(targets, preds, beta = 0.9, zero_division = 0)

    # ┏━━━━━━━━━━ Plot setup ━━━━━━━━━━┓
    fig, ax = plt.subplots(figsize=(4, 4))
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot(cmap=cmap, ax=ax, colorbar=False)
    if best_threshold is not None:
        ax.set_title(title + f"@Tau = {best_threshold:.4f}")
    else:
        ax.set_title(title)

    # ┏━━━━━━━━━━ Text annotation ━━━━━━━━━━┓
    textstr = (f"Accuracy : {acc:.2f}\n"
               f"Precision: {prec:.2f}\n"
               f"Recall   : {rec:.2f}\n"
               f"F1 Score : {f1:.2f}\n"
               f"F-Beta-Score: {fbeta:.2f}")
    ax.text(1.05, 
            0.6, 
            textstr,
            transform         = ax.transAxes,
            fontsize          = 10,
            verticalalignment = 'top',
            bbox              = dict(boxstyle='round', facecolor='white', edgecolor='gray'))

    # ┏━━━━━━━━━━ Persist figure ━━━━━━━━━━┓
    fig.tight_layout()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    fname_core = title.replace(" — ", "_")
    if best_threshold is not None:
        fname_core += "@Tau"
    fname = fname_core + ".png"
    fig.savefig(out_path / fname, dpi=150)
    plt.close(fig)

    #print(f"Path: {out_path / fname}")


def export_predictions(df_asset,
                       dataset_tensor,
                       tpreds: np.ndarray,
                       tpreds_tau: np.ndarray,
                       cfg: dict,
                       checkpoint_dir: Path,
                       tprobs: np.ndarray = None,
                       meta_label_mode: Optional[str] = None) -> tuple[Path, dict]:
    """
    Export a CSV with test-timeline dates aligned to CTTS predictions.

    Parameters
    - df_asset: DataFrame returned by merge_meta_targets (indexed by 'date').
    - dataset_tensor: TensorDataset used to build loaders (len(...) equals number of windows).
    - tpreds: numpy array of test predictions at the default threshold (0.5).
    - tpreds_tau: numpy array of test predictions computed at the best threshold.
    - cfg: full training config dict (reads 'splits', 'sequence_length', 'dataset', 'training_mode').
    - checkpoint_dir: Path where to save the CSV.

    Returns
    -------
    Tuple[Path, dict]
        Path to the CSV and consensus metrics for M1+M2 (default & τ).
    """
    if tpreds_tau is None:
        raise ValueError("tpreds_tau must be provided to export best-threshold predictions.")

    # ┏━━━━━━━━━━ Recompute split sizes to recover test indices ━━━━━━━━━━┓
    idx_test_ds = getattr(dataset_tensor, "test_indices", None)
    if idx_test_ds is None:
        N_windows   = len(dataset_tensor)
        n_train_win = int(cfg["splits"]["train"] * N_windows)
        n_val_win   = int(cfg["splits"]["val"]   * N_windows)
        idx_test_ds = list(range(n_train_win + n_val_win, N_windows))
    else:
        idx_test_ds = list(idx_test_ds)

    # ┏━━━━━━━━━━ Map each dataset index k to its corresponding end-of-window date: df.index[k + seq_len - 1] ━━━━━━━━━━┓
    seq_len = cfg["sequence_length"]
    all_dates = df_asset.index
    mapped_dates = [all_dates[k + seq_len - 1] for k in idx_test_ds]

    # ┏━━━━━━━━━━ Sanity check ━━━━━━━━━━┓
    if len(mapped_dates) != len(tpreds):
        print(f"[WARN] Date/predictions length mismatch: dates={len(mapped_dates)} preds={len(tpreds)}")

    # ┏━━━━━━━━━━ Start with DataFrame of test dates ━━━━━━━━━━┓
    limit = min(len(mapped_dates), len(tpreds))
    base_df = pd.DataFrame({"date": mapped_dates[:limit]})

    # ┏━━━━━━━━━━ Load M1 CSVs (UP and DOWN) ━━━━━━━━━━┓
    provider = cfg["dataset"]["source"]
    market   = cfg["dataset"]["type"].capitalize()
    symbol   = cfg["dataset"]["symbol"]
    granularity = cfg["training_mode"]["granularity_usual"]
    training_mode = cfg["training_mode"]
    data_root = cfg.get("paths", {}).get("data_root")
    up_path = dataset_path(
        provider, market, symbol, "up", granularity=granularity,
        meta_label_mode=meta_label_mode, data_root=data_root,
    )
    dn_path = dataset_path(
        provider, market, symbol, "down", granularity=granularity,
        meta_label_mode=meta_label_mode, data_root=data_root,
    )

    up_df = pd.read_csv(up_path, parse_dates=["date"]) if up_path.exists() else pd.DataFrame(columns=["date"])
    dn_df = pd.read_csv(dn_path, parse_dates=["date"]) if dn_path.exists() else pd.DataFrame(columns=["date"])

    # ┏━━━━━━━━━━ Select and rename necessary columns ━━━━━━━━━━┓
    up_keep = [c for c in ["date", "prediction", "ground_truth", "pred", "pred_proba", "lab", "meta_target", "meta_label"] if c in up_df.columns]
    dn_keep = [c for c in ["date", "prediction", "ground_truth", "pred", "pred_proba", "lab", "meta_target", "meta_label"] if c in dn_df.columns]
    up_sel = up_df.loc[:, up_keep].rename(columns={"pred":        "m1_pred_up",
                                                   "pred_proba":  "m1_pred_proba_up",
                                                   "meta_target": "meta_label_up",
                                                   "meta_label":  "meta_label_up",
                                                   "lab":         "lab_up",
                                                   "ground_truth": "ground_truth_up",
                                                   "prediction":  "numerical_prediction_up"}) if len(up_keep) else pd.DataFrame(columns=["date"]) 

    dn_sel = dn_df.loc[:, dn_keep].rename(columns={"pred":        "m1_pred_down",
                                                   "pred_proba":  "m1_pred_proba_dn",
                                                   "meta_target": "meta_label_dn",
                                                   "meta_label":  "meta_label_dn",
                                                   "lab":         "lab_dn",
                                                   "ground_truth":"ground_truth_dn",
                                                   "prediction":  "numerical_prediction_dn",}) if len(dn_keep) else pd.DataFrame(columns=["date"]) 

    # ┏━━━━━━━━━━ Merge on test dates ━━━━━━━━━━┓
    merged = (base_df
              .merge(up_sel, on="date", how="left")
              .merge(dn_sel, on="date", how="left"))

    # ┏━━━━━━━━━━ Coalesce 'prediction' and 'ground_truth' ━━━━━━━━━━┓
    # ┏━━━━━━━━━━ Task-specific columns ━━━━━━━━━━┓
    task = training_mode["normal_task"].upper()
    task_lower = task.lower()

    pred_series_up = merged["numerical_prediction_up"]
    pred_series_dn = merged["numerical_prediction_dn"]
    gt_series_up   = merged["ground_truth_up"]
    gt_series_dn   = merged["ground_truth_dn"]

    task_pred_col = f"numerical_prediction_{task_lower}"
    task_gt_col   = f"ground_truth_{task_lower}"

    # ┏━━━━━━━━━━ Use the series that matches the active task, fall back to whichever exists  ━━━━━━━━━━┓
    if isinstance(merged[task_pred_col], pd.Series):
        merged["prediction"] = merged[task_pred_col]
    elif isinstance(pred_series_up, pd.Series) or isinstance(pred_series_dn, pd.Series):
        merged["prediction"] = (
            (pred_series_up if isinstance(pred_series_up, pd.Series) else pd.Series([pd.NA] * len(merged), index=merged.index))
            .combine_first(pred_series_dn if isinstance(pred_series_dn, pd.Series) else pd.Series([pd.NA] * len(merged), index=merged.index)))
    else:
        merged["prediction"] = pd.Series([pd.NA] * len(merged), index=merged.index)

    if isinstance(merged[task_gt_col], pd.Series):
        merged["ground_truth"] = merged[task_gt_col]
    elif isinstance(gt_series_up, pd.Series) or isinstance(gt_series_dn, pd.Series):
        merged["ground_truth"] = (
            (gt_series_up if isinstance(gt_series_up, pd.Series) else pd.Series([pd.NA] * len(merged), index=merged.index))
            .combine_first(gt_series_dn if isinstance(gt_series_dn, pd.Series) else pd.Series([pd.NA] * len(merged), index=merged.index)))
    else:
        merged["ground_truth"] = pd.Series([pd.NA] * len(merged), index=merged.index)

    if task == "UP":
        merged["meta_label"] = merged["meta_label_up"]
    else:
        merged["meta_label"] = merged["meta_label_dn"]

    # ┏━━━━━━━━━━ Include CTTS (M2) discrete predictions ━━━━━━━━━━┓
    pred_col = f"m2_pred_{task_lower}"
    pred_tau_col = f"{pred_col}_tau"
    merged[pred_col] = pd.NA
    merged[pred_tau_col] = pd.NA

    # ┏━━━━━━━━━━ Helper Function: Copy numpy window into merged DataFrame column ━━━━━━━━━━┓
    def _assign_numpy(values: np.ndarray, column: str, cast) -> None:
        arr = np.asarray(values)
        limit = min(len(arr), len(merged))
        if limit == 0:
            return
        merged.loc[merged.index[:limit], column] = cast(arr[:limit])

    _assign_numpy(tpreds, pred_col, cast = lambda x: x.astype(int))
    _assign_numpy(tpreds_tau, pred_tau_col, cast = lambda x: x.astype(int))
    
    # ┏━━━━━━━━━━ Optionally include M2 probability for the active task ━━━━━━━━━━┓
    if tprobs is not None:
        prob_col = f"m2_prob_{task_lower}"
        merged[prob_col] = pd.NA
        _assign_numpy(tprobs, prob_col, cast=lambda x: x)

    # ┏━━━━━━━━━━ Final column set (at minimum) ━━━━━━━━━━┓
    final_cols = ["date",
                  "prediction",
                  "m1_pred_up",
                  "m1_pred_down",
                  "m1_pred_proba_up",
                  "m1_pred_proba_dn",
                  "meta_label",
                  "ground_truth",
                  "lab_up",
                  "lab_dn",
                  pred_tau_col,
                  pred_col]

    # ┏━━━━━━━━━━ Include probability column for this task only if present ━━━━━━━━━━┓
    prob_col = f"m2_prob_{task_lower}"
    if prob_col in merged.columns:
        final_cols.append(prob_col)
    for col in final_cols:
        if col not in merged.columns:
            merged[col] = pd.NA

    out_df = merged[final_cols].copy()
    out_df["date"] = pd.to_datetime(out_df["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    # ┏━━━━━━━━━━ Consensus Metrics ━━━━━━━━━━┓
    training_mode = cfg["training_mode"]
    task          = training_mode["normal_task"].upper()
    task_lower    = task.lower()
    beta_metric   = cfg[f"train_{task_lower}"]["fbeta"]
    invert_consensus = meta_label_mode == "fp"
    
    # ┏━━━━━━━━━━ Column Names ━━━━━━━━━━┓
    m1_col = "m1_pred_down" if task == "DN" else "m1_pred_up"
    m2_col = f"m2_pred_{task_lower}"
    m2_tau_col = f"{m2_col}_tau"
    target_col = "lab_dn" if task == "DN" else "lab_up"

    # ┏━━━━━━━━━━ Empty Payload ━━━━━━━━━━┓
    metrics_payload = {"M1&M2_Accuracy":      0.0,
                       "M1&M2_Precision":     0.0,
                       "M1&M2_Recall":        0.0,
                       "M1&M2_F1":            0.0,
                       "M1&M2_FBeta":         0.0,
                       "M1&M2_Accuracy@Tau":  0.0,
                       "M1&M2_Precision@Tau": 0.0,
                       "M1&M2_Recall@Tau":    0.0,
                       "M1&M2_F1@Tau":        0.0,
                       "M1&M2_FBeta@Tau":     0.0}

    valid_mask = out_df[target_col].notna()
    if valid_mask.any():
        def _series_to_int_array(series: pd.Series) -> np.ndarray:
            """Fill gaps with zeros and ensure numeric dtype before casting to int."""
            return (
                pd.to_numeric(series, errors="coerce")
                .fillna(0)
                .astype(int)
                .to_numpy()
            )

        y_true = _series_to_int_array(out_df.loc[valid_mask, target_col])
        m1_preds = _series_to_int_array(out_df.loc[valid_mask, m1_col])
        m2_preds = _series_to_int_array(out_df.loc[valid_mask, m2_col])
        m2_preds_tau = _series_to_int_array(out_df.loc[valid_mask, m2_tau_col])
        
        # ┏━━━━━━━━━━ Helper Function: Consensus Logic for "original" & "tp" vs "fp" ━━━━━━━━━━┓
        def _consensus(m2_series: np.ndarray) -> np.ndarray:
            if invert_consensus:
                return ((m1_preds == 1) & (m2_series == 0)).astype(int)
            return ((m1_preds == 1) & (m2_series == 1)).astype(int)

        # ┏━━━━━━━━━━ Consensus Logic for "original" & "tp" vs "fp" depending on predictions ━━━━━━━━━━┓
        m1m2_default = _consensus(m2_preds)
        m1m2_tau = _consensus(m2_preds_tau)

        # ┏━━━━━━━━━━ Classification Metrics ━━━━━━━━━━┓
        metrics_default = compute_basic_metrics(y_true, m1m2_default, beta_metric)
        metrics_tau     = compute_basic_metrics(y_true, m1m2_tau, beta_metric)

        # ┏━━━━━━━━━━ Final Payload ━━━━━━━━━━┓
        metrics_payload = {"M1+M2_Accuracy":      metrics_default["accuracy"],
                           "M1+M2_Precision":     metrics_default["precision"],
                           "M1+M2_Recall":        metrics_default["recall"],
                           "M1+M2_F1":            metrics_default["f1"],
                           "M1+M2_FBeta":         metrics_default["fbeta"],
                           "M1+M2_Accuracy@Tau":  metrics_tau["accuracy"],
                           "M1+M2_Precision@Tau": metrics_tau["precision"],
                           "M1+M2_Recall@Tau":    metrics_tau["recall"],
                           "M1+M2_F1@Tau":        metrics_tau["f1"],
                           "M1+M2_FBeta@Tau":     metrics_tau["fbeta"]}

    # ┏━━━━━━━━━━ Write with requested naming ━━━━━━━━━━┓
    out_path = checkpoint_dir / f"{symbol}_{task}_predictions.csv"
    out_df.to_csv(out_path, index=False)

    return out_path, metrics_payload




def plot_meta_labeling_consensus(cfg: dict, checkpoint_dir: Path, best_threshold: float = None) -> Path:
    """
    Build consensus confusion matrices (default threshold and best threshold) using the
    exported predictions CSV. Consensus equals 1 only when the M1 prediction and CTTS
    prediction agree on 1 for the active task. The plots are stored as
    `M1+M2_<TASK>_Results.png` (default threshold) and `M1+M2@tau_<TASK>_Results.png`.

    Additionally, plot M1 vs lab confusion matrices for the UP and DOWN predictions
    (if the exported CSV provides those columns).
    """
    # ┏━━━━━━━━━━ Resolve training metadata needed for consensus ━━━━━━━━━━┓
    training_mode = cfg["training_mode"]
    task = training_mode["normal_task"].upper()
    meta_mode = training_mode["meta_label_usual"].lower()
    symbol = cfg["dataset"]["symbol"]
    csv_path = Path(checkpoint_dir) / f"{symbol}_{task}_predictions.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Predictions CSV not found at {csv_path}")

    # ┏━━━━━━━━━━ Load exported predictions ━━━━━━━━━━┓
    df = pd.read_csv(csv_path)
    task_lower = task.lower()
    m1_suffix = "down" if task == "DN" else "up"
    m1_col = f"m1_pred_{m1_suffix}"
    m2_col = f"m2_pred_{task_lower}"
    m2_tau_col = f"{m2_col}_tau"
    target_col = "lab_dn" if task == "DN" else "lab_up"

    # ┏━━━━━━━━━━ Ensure required columns exist ━━━━━━━━━━┓
    required_cols = {m1_col, m2_col, m2_tau_col, target_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns in predictions CSV: {sorted(missing)}")

    # ┏━━━━━━━━━━ Cast predictions/targets to clean integer arrays ━━━━━━━━━━┓
    m1_preds = df[m1_col].fillna(0).astype(int)
    m2_preds = df[m2_col].fillna(0).astype(int)
    m2_preds_tau = df[m2_tau_col].fillna(0).astype(int)
    targets  = df[target_col].fillna(0).astype(int)

    # ┏━━━━━━━━━━ Configure label names and FP-specific consensus rule ━━━━━━━━━━┓
    labels = (f"No_TP_{task}", f"TP_{task}")
    invert_consensus = meta_mode == "fp"

    def _plot_consensus(m2_series: pd.Series, suffix: str, tau_value: float = None, cmap: str = "Purples") -> Path:
        # ┏━━━━━━━━━━ Combine M1+M2 signals depending on meta_label mode ━━━━━━━━━━┓
        if invert_consensus:
            consensus_preds = ((m1_preds == 1) & (m2_series == 0)).astype(int)
        else:
            consensus_preds = ((m1_preds == 1) & (m2_series == 1)).astype(int)

        # ┏━━━━━━━━━━ Build confusion matrix and plot aesthetics ━━━━━━━━━━┓
        cm = confusion_matrix(targets, consensus_preds, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(4, 4))
        disp = ConfusionMatrixDisplay(cm, display_labels=labels)
        disp.plot(cmap=cmap, ax=ax, colorbar=False)

        title = f"M1+M2_{task} — Test"
        if tau_value is not None:
            title = f"{title}@Tau = {tau_value:.4f}"
        ax.set_title(title)
        ax.set_xlabel("Predicted Consensus")
        ax.set_ylabel("True Label")

        # ┏━━━━━━━━━━ Aggregate scalar metrics ━━━━━━━━━━┓
        acc  = accuracy_score(targets, consensus_preds)
        prec = precision_score(targets, consensus_preds, zero_division=0)
        rec  = recall_score(targets, consensus_preds, zero_division=0)
        f1   = f1_score(targets, consensus_preds, zero_division=0)

        textstr = (f"Accuracy : {acc:.2f}\n"
                   f"Precision: {prec:.2f}\n"
                   f"Recall   : {rec:.2f}\n"
                   f"F1 Score : {f1:.2f}")
        ax.text(1.05,
                0.6,
                textstr,
                transform         = ax.transAxes,
                fontsize          = 10,
                verticalalignment = 'top',
                bbox              = dict(boxstyle='round', facecolor='white', edgecolor='gray'))

        fig.tight_layout()
        fname = f"M1+M2{suffix}_{task}_Results.png"
        out_path = csv_path.parent / fname
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Meta-Labeling Results{'' if not suffix else ' (@Tau)'}: {out_path}")
        return out_path

    # ┏━━━━━━━━━━ Produce default and @tau consensus plots ━━━━━━━━━━┓
    default_path = _plot_consensus(m2_preds, suffix = "", tau_value = None)
    _plot_consensus(m2_preds_tau, suffix = "@tau", tau_value = best_threshold)

    cmap = "Blues"
    plot_specs = [("m1_pred_up", "lab_up", ("No_TP_UP", "TP_UP"), "M1_UP — Test"),
                  ("m1_pred_down", "lab_dn", ("No_TP_DN", "TP_DN"), "M1_DN — Test"),]

    for pred_col, lab_col, display_labels, title in plot_specs:
        preds = df[pred_col].fillna(0).astype(int)
        lab_targets = df[lab_col].fillna(0).astype(int)
        plot_cm_with_metrics(preds   = preds,
                             targets = lab_targets,
                             labels  = display_labels,
                             title   = title,
                             out_dir = checkpoint_dir,
                             best_threshold = None,
                             cmap    = cmap)

    return default_path
