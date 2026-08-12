import os
import torch
import random
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import Any, Dict, Optional, Literal, List
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score
from torch.amp import GradScaler, autocast
from .optim_utils import step_scheduler

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ THRESHOLD / GATING UTILITIES                                          ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
def build_m1_window(df_asset: pd.DataFrame, task: str, seq_len: int, total_samples: int) -> np.ndarray:
    """Align per-window M1 predictions with the sliding-window dataset."""
    task = task.upper()
    m1_col = "m1_up" if task == "UP" else "m1_dn"

    if m1_col not in df_asset.columns:
        raise KeyError(f"Column '{m1_col}' not found in merged dataset; required for risk/coverage analysis.")
    
    if total_samples <= 0:
        raise ValueError("Dataset contains no samples; cannot build M1 window alignment.")
    
    m1_series = df_asset[m1_col].fillna(0).to_numpy(dtype=float)
    end_indices = np.arange(seq_len - 1, seq_len - 1 + total_samples)
    
    if end_indices[-1] >= m1_series.shape[0]:
        raise IndexError("M1 series shorter than expected when aligning with dataset windows.")
    return np.nan_to_num(m1_series[end_indices], nan=0.0).astype(int)


def build_scores(probs: np.ndarray, gating: str) -> np.ndarray:
    """
    Adapt raw probabilities to the configured gating policy.

    Parameters
    ----------
    probs : np.ndarray
        Raw probability outputs from M2.
    gating : str
        Gating mode name (e.g., "prob_only", "positive_only_hard").
    """
    # ┏━━━━━━━━━━ No gating ━━━━━━━━━━┓
    if gating == "prob_only":
        return probs
    # ┏━━━━━━━━━━ Hard-positive gating ━━━━━━━━━━┓
    if gating == "positive_only_hard":
        base = probs >= 0.5
        return np.where(base, probs, float("-inf"))
    # ┏━━━━━━━━━━ Unknown policy ━━━━━━━━━━┓
    raise ValueError(f"Unknown gating mode: {gating!r}") 


def apply_threshold(scores: np.ndarray, threshold: float) -> np.ndarray:
    """
    Convert continuous scores into binary decisions via τ.
    """
    # ┏━━━━━━━━━━ Element-wise comparison ━━━━━━━━━━┓
    return (scores >= threshold).astype(int)


def select_threshold_fbeta(y_true: np.ndarray,
                           scores: np.ndarray,
                           thresholds: np.ndarray,
                           beta: float) -> tuple[float, float]:
    """
    Sweep candidate thresholds and pick the best by Fβ.
    """
    # ┏━━━━━━━━━━ Initialise trackers ━━━━━━━━━━┓
    best_tau = float(thresholds[0])
    best_score = -1.0
    # ┏━━━━━━━━━━ Iterate over τ candidates ━━━━━━━━━━┓
    for tau in thresholds:
        preds = apply_threshold(scores, tau)
        # ┏━━━━━━━━━━# If no positives are predicted, Fβ = 0 to avoid division errors ━━━━━━━━━━┓
        if preds.sum() == 0:
            score = 0.0
        else:
            score = fbeta_score(y_true, 
                                preds, 
                                beta = beta, 
                                zero_division = 0)  
        # ┏━━━━━━━━━━ Track the threshold that yields the highest Fβ score ━━━━━━━━━━┓
        if score > best_score:
            best_score = score
            best_tau = float(tau)
    return best_tau, best_score


def evaluate_threshold(y_true: np.ndarray, 
                       scores: np.ndarray, 
                       threshold: float) -> Dict[str, float]:
    """
    Compute risk/coverage metrics at a fixed τ.
    """
    # ┏━━━━━━━━━━ Binary decisions ━━━━━━━━━━┓
    preds = apply_threshold(scores, threshold)
    selected = preds == 1

    # ┏━━━━━━━━━━ Fraction and count of selected samples ━━━━━━━━━━┓
    coverage = float(np.mean(selected))
    selected_count = int(np.sum(selected))
    # ┏━━━━━━━━━━ Risk over selected subset ━━━━━━━━━━┓
    if selected_count == 0:
        risk = np.nan
    else:
        risk = float(np.mean(y_true[selected] == 0))
        
    return {"risk": risk,
            "coverage": coverage,
            "predictions": preds,
            "selected_count": selected_count}


def task_features(cfg: dict, prefix: str, task: str) -> list:
    """
    Return the feature list for the given task, falling back to shared entries.

    An empty list is allowed to explicitly disable a feature family (e.g. context channels).
    """
    # ┏━━━━━━━━━━ Resolve config key ━━━━━━━━━━┓
    key = f"{prefix}_{task.lower()}"
    if key not in cfg:
        raise KeyError(f"Missing '{key}' in configuration for task '{task}'.")
    values = cfg[key]
    
    # ┏━━━━━━━━━━ Normalise value ━━━━━━━━━━┓
    if values is None:
        return []
    if not isinstance(values, list):
        raise ValueError(f"Configuration key '{key}' must be a list.")
    return values

# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ INITIALIZATION OF SEEDS                                               ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ EPOCH LOOPS: TRAINING & TESTING                                       ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
def epoch_loop(model: torch.nn.Module,
               loader: torch.utils.data.DataLoader,
               criterion: nn.Module,
               device: torch.device,
               *,
               mode: Literal["train", "val", "test"],
               task_name: str,
               optimizer: Optional[torch.optim.Optimizer] = None,
               bce_thr: float = 0.5,
               amp: bool = True,
               clip_grad: Optional[float] = 1.0,
               beta: float = 1.15,
               return_raw: bool = False) -> Dict[str, Any]:
    """
    Unified pass over a dataloader.

    Parameters
    ----------
    mode : {"train","val","test"}
        Controls gradient tracking and optimiser usage.
    task_name : str
        Either "UP" or "DN" to select the proper label tensor.
    optimizer : torch.optim.Optimizer, optional
        Required when mode == "train".
    bce_thr : float
        Threshold applied to sigmoid outputs when using BCE loss.
    amp : bool
        Enable autocast + GradScaler when running on CUDA.
    clip_grad : Optional[float]
        Max-norm for gradient clipping. Set to None to disable.
    beta : float
        β parameter for the Fβ-score.
    return_raw : bool
        When True, returns numpy arrays for predictions, probabilities and targets
        in addition to aggregate metrics.
    """
    # ┏━━━━━━━━━━ Training/Val/Testing Cases ━━━━━━━━━━┓
    mode = mode.lower()
    if mode not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported epoch_loop mode: {mode}")

    # ┏━━━━━━━━━━ Training Mode Case ━━━━━━━━━━┓
    is_train = mode == "train"
    model.train() if is_train else model.eval()
    if is_train and optimizer is None:
        raise ValueError("Optimizer must be provided when mode = 'train'")
    
    # ┏━━━━━━━━━━ AMP ━━━━━━━━━━┓
    amp_enabled = amp and device.type == "cuda"
    scaler = GradScaler('cuda') if amp_enabled else None
    
    # ┏━━━━━━━━━━ To store predictions, probabilities and logits ━━━━━━━━━━┓
    losses: List[float] = []
    preds_cpu: List[torch.Tensor] = []
    targets_cpu: List[torch.Tensor] = []
    probs_cpu: Optional[List[torch.Tensor]] = [] if return_raw else None
    logits_cpu: Optional[List[torch.Tensor]] = [] if return_raw else None

    # ┏━━━━━━━━━━ Requested Task ━━━━━━━━━━┓
    task_name = task_name.upper()

    with torch.set_grad_enabled(is_train):
        # ┏━━━━━━━━━━ Main Loop ━━━━━━━━━━┓
        for xb, y_up, y_dn in loader:
            # ┏━━━━━━━━━━ For faster transfer ━━━━━━━━━━┓
            xb = xb.to(device, non_blocking = True)
            tgt_raw = y_up if task_name == "UP" else y_dn
            tgt_raw = tgt_raw.to(device, non_blocking=True)

            # ┏━━━━━━━━━━ Criterion Instantiation ━━━━━━━━━━┓
            is_bce = isinstance(criterion, nn.BCEWithLogitsLoss)

            # ┏━━━━━━━━━━ Forward Pass (+ Autocast) + Predictions ━━━━━━━━━━┓
            with autocast('cuda', enabled = amp_enabled):
                if is_bce:
                    logits = model(xb).squeeze(1)
                    loss = criterion(logits, tgt_raw.float())
                    prob_tensor = torch.sigmoid(logits)
                    pred_tensor = (prob_tensor > bce_thr).long()
                else:
                    logits = model(xb)
                    loss = criterion(logits, tgt_raw.long())
                    prob_tensor = torch.softmax(logits, dim=1)[:, 1] if return_raw else None
                    pred_tensor = logits.argmax(dim=1)
            
            # ┏━━━━━━━━━━ Back-prop ━━━━━━━━━━┓
            if is_train:
                # ┏━━━━━━━━━━ Cheaper ━━━━━━━━━━┓
                optimizer.zero_grad(set_to_none = True)

                if amp_enabled:
                    # ┏━━━━━━━━━━ Scale + Backward ━━━━━━━━━━┓
                    scaler.scale(loss).backward()
                    
                    # ┏━━━━━━━━━━ Unscale if we want gradient clipping ━━━━━━━━━━┓
                    if clip_grad is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                    
                    # ┏━━━━━━━━━━ Step & Update ━━━━━━━━━━┓
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # ┏━━━━━━━━━━ Pure FP32 ━━━━━━━━━━┓
                    loss.backward()
                    if clip_grad is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                    optimizer.step()

            # ┏━━━━━━━━━━ Store Losses, Predictions and Targets ━━━━━━━━━━┓
            losses.append(loss.detach().item())
            preds_cpu.append(pred_tensor.detach().cpu())
            targets_cpu.append(tgt_raw.detach().long().cpu())

            # ┏━━━━━━━━━━ Probabilities & Logits ━━━━━━━━━━┓
            if return_raw:
                if probs_cpu is not None:
                    if is_bce:
                        probs_cpu.append(prob_tensor.detach().cpu())
                    else:
                        probs_cpu.append(prob_tensor.detach().cpu())
                if logits_cpu is not None:
                    logits_cpu.append(logits.detach().cpu())
    
    # ┏━━━━━━━━━━ Format of Predictions, Targets and Mean Loss ━━━━━━━━━━┓
    preds = torch.cat(preds_cpu).numpy()
    targets = torch.cat(targets_cpu).numpy()
    mean_loss = float(np.mean(losses)) if losses else 0.0

    # ┏━━━━━━━━━━ Compute Metrics ━━━━━━━━━━┓
    acc   = accuracy_score(targets, preds) if targets.size else 0.0
    prec  = precision_score(targets, preds, zero_division = 0) if targets.size else 0.0
    rec   = recall_score(targets, preds, zero_division = 0) if targets.size else 0.0
    f1    = f1_score(targets, preds, zero_division = 0) if targets.size else 0.0
    fbeta = fbeta_score(targets, preds, beta = beta, zero_division = 0) if targets.size else 0.0

    # ┏━━━━━━━━━━ Dictionary to store Results ━━━━━━━━━━┓
    result: Dict[str, Any] = {"loss": mean_loss,
                              "acc": acc,
                              "prec": prec,
                              "rec": rec,
                              "f1": f1,
                              "fbeta": fbeta}

    # ┏━━━━━━━━━━ Additional Information ━━━━━━━━━━┓
    if return_raw:
        result["preds"] = preds
        result["targets"] = targets
        if probs_cpu is not None and probs_cpu:
            result["probs"] = torch.cat(probs_cpu).numpy()
        if logits_cpu is not None and logits_cpu:
            result["logits"] = torch.cat(logits_cpu).numpy()

    return result


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ CLASSIFICATION METRICS                                                ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
def compute_basic_metrics(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          beta: float) -> dict:
    """Compute accuracy/precision/recall/F1/Fβ for masked subsets."""
    # ┏━━━━━━━━━━ If empty ━━━━━━━━━━┓
    if y_true.size == 0:
        return {"accuracy":  0.0,
                "precision": 0.0,
                "recall":    0.0,
                "f1":        0.0,
                "fbeta":     0.0}

    return {"accuracy":  accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division = 0),
            "recall":    recall_score(y_true, y_pred, zero_division = 0),
            "f1":        f1_score(y_true, y_pred, zero_division = 0),
            "fbeta":     fbeta_score(y_true, y_pred, beta = beta, zero_division = 0)}


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ MODEL TRAIN                                                           ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
def model_train(model, 
                optimizer, 
                criterion, 
                early_stopper,
                train_loader, 
                val_loader, 
                writer, 
                device,
                MAX_EPOCHS, 
                task_name,
                scheduler,
                bce_thr,
                beta
                ):
    """
    Full training loop with early stopping on validation loss.
    Logs train/val losses to TensorBoard as '{task_name}/Loss/{Train,Val}'.
    """
    # ┏━━━━━━━━━━ Storage of Losses ━━━━━━━━━━┓
    train_losses = []
    val_losses   = []

    # ┏━━━━━━━━━━ Header ━━━━━━━━━━┓
    header = f"▶️▶️▶️ Starting {task_name} training for {MAX_EPOCHS} epochs ◀️◀️◀️"
    print("\n" + "="*len(header))
    print(header)
    print("="*len(header) + "\n")

    for epoch in range(1, MAX_EPOCHS+1):
        # ┏━━━━━━━━━━ Training ━━━━━━━━━━┓
        train_metrics = epoch_loop(model,
                                   train_loader,
                                   criterion,
                                   device,
                                   mode = "train",
                                   task_name = task_name,
                                   optimizer = optimizer,
                                   bce_thr = bce_thr,
                                   amp = True,
                                   clip_grad = 1.0,
                                   beta = beta,
                                   return_raw = False)

        # ┏━━━━━━━━━━ Validation ━━━━━━━━━━┓
        val_metrics = epoch_loop(model,
                                 val_loader,
                                 criterion,
                                 device,
                                 mode = "val",
                                 task_name = task_name,
                                 optimizer = None,
                                 bce_thr = bce_thr,
                                 amp = True,
                                 clip_grad = 0.0,
                                 beta = beta,
                                 return_raw = False)
        
        # ┏━━━━━━━━━━ Record losses ━━━━━━━━━━┓
        tr_loss = train_metrics["loss"]
        val_loss = val_metrics["loss"]
        train_losses.append(tr_loss)
        val_losses.append(val_loss)

        # ┏━━━━━━━━━━ TensorBoard logging ━━━━━━━━━━┓
        writer.add_scalar(f"{task_name}/Loss/Train", tr_loss, epoch)
        writer.add_scalar(f"{task_name}/Loss/Val",   val_loss, epoch)

        # ┏━━━━━━━━━━ LR-scheduler step ━━━━━━━━━━┓
        step_scheduler(scheduler, epoch, val_loss)

        # ┏━━━━━━━━━━ Early-stopping ━━━━━━━━━━┓
        early_stopper(val_loss, model)
        if early_stopper.early_stop:
            print(f"🚨 Early stopping triggered at epoch {epoch} 🚨\n")
            break
    
    # ┏━━━━━━━━━━ Compute and print averages ━━━━━━━━━━┓
    avg_train = sum(train_losses) / len(train_losses)
    avg_val   = sum(val_losses)   / len(val_losses)
    print(f"   ▶︎ Average Train Loss: {avg_train:.4f}")
    print(f"   ▶︎ Average Val   Loss: {avg_val:.4f}\n")

    # ┏━━━━━━━━━━ Footer ━━━━━━━━━━┓
    footer = f"▶️▶️▶️ Finished {task_name} training for {MAX_EPOCHS} epochs ◀️◀️◀️"
    print("="*len(footer))
    print(footer)
    print("="*len(footer) + "\n")

    return val_loss


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ MODEL TEST                                                            ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
def model_test(model, 
               criterion, 
               test_loader, 
               writer, 
               device,
               step, 
               task_name,
               bce_thr,
               beta
            ):
    """
    Run one pass over 'test_loader', print metrics, and log test-loss at given 'step'.
    """ 
    # ┏━━━━━━━━━━ Header ━━━━━━━━━━┓
    header = f"▶️▶️▶️ Starting {task_name} Testing ◀️◀️◀️"
    print("\n" + "="*len(header))
    print(header)
    print("="*len(header) + "\n")
    
    # ┏━━━━━━━━━━ Testing ━━━━━━━━━━┓
    test_metrics = epoch_loop(model,
                              test_loader,
                              criterion,
                              device,
                              mode = "test",
                              task_name = task_name,
                              optimizer = None,
                              bce_thr = bce_thr,
                              amp = True,
                              clip_grad = 0.0,
                              beta = beta,
                              return_raw = False)

    # ┏━━━━━━━━━━ Extracting Loss & Metrics ━━━━━━━━━━┓
    loss = test_metrics["loss"]
    acc = test_metrics["acc"]
    prec = test_metrics["prec"]
    rec = test_metrics["rec"]
    f1 = test_metrics["f1"]
    fbeta = test_metrics["fbeta"]

    #  ━━━━━━━━━━┓ Pretty test banner ━━━━━━━━━━┓
    print("#"*23)
    print(f"🌟 {task_name} TEST RESULTS 🌟")
    print("#"*23)
    print(f"{'Metric':<10}{'Value':>10}")
    print(f"{'Loss':<10}{loss:>10.4f}")
    print(f"{'Accuracy':<10}{acc:>10.4f}")
    print(f"{'Precision':<10}{prec:>10.4f}")
    print(f"{'Recall':<10}{rec:>10.4f}")
    print(f"{'F1':<10}{f1:>10.4f}")
    print("#"*23 + "\n")
    
    # ┏━━━━━━━━━━ Footer ━━━━━━━━━━┓
    footer = f"▶️▶️▶️ Finished {task_name} Testing ◀️◀️◀️"
    print("="*len(footer))
    print(footer)
    print("="*len(footer) + "\n")

    return loss, acc, prec, rec, f1, fbeta
