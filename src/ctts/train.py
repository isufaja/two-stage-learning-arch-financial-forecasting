# ┏━━━━━━━━━━ General Imports ━━━━━━━━━━┓
import os
import argparse
import datetime
import yaml
import torch
import copy
import warnings
import numpy as np
from pathlib import Path
from typing import Sequence
from tqdm.auto import tqdm
from torch.utils.data import TensorDataset
from torch.utils.tensorboard import SummaryWriter

# ┏━━━━━━━━━━ Data Preprocessing utils ━━━━━━━━━━┓
from .paths import PROJECT_ROOT, dataset_path, resolve_project_path
from .data_preprocessing import (merge_meta_targets, 
                                build_loaders, 
                                prepare_dataset, 
                                count_meta_targets)

# ┏━━━━━━━━━━ Optimization utils ━━━━━━━━━━┓
from .utils.optim_utils import (get_optimizer, 
                               make_scheduler, 
                               step_scheduler)

# ┏━━━━━━━━━━ Training utils ━━━━━━━━━━┓
from .utils.train_utils import (epoch_loop,
                               seed_everything,
                               task_features,
                               build_scores,
                               select_threshold_fbeta,
                               evaluate_threshold,
                               build_m1_window,
                               compute_basic_metrics)

# ┏━━━━━━━━━━ Model Class ━━━━━━━━━━┓
from .model import CTTSModel, EarlyStopping

# ┏━━━━━━━━━━ Evaluation utils ━━━━━━━━━━┓
from .utils.test_utils import (plot_cm_with_metrics,
                              export_predictions,
                              plot_meta_labeling_consensus)

# ┏━━━━━━━━━━ Selective Classification ━━━━━━━━━━┓
from .selective_classification import (save_metrics,
                                      coverage_at_risk,
                                      collect_risk_coverage_curve,
                                      area_under_risk_coverage,
                                      plot_coverage_risk_curve)

# ┏━━━━━━━━━━ Ignoring Warning Message ━━━━━━━━━━┓
os.environ.setdefault("TORCH_CUDA_NVML_DISABLE_WARNING", "1")
warnings.filterwarnings("ignore", message="Can't initialize NVML")


DEFAULT_SEED = 1493583942

def run_training(cfg: dict, seed: int = DEFAULT_SEED) -> None:
    cfg = copy.deepcopy(cfg)
    paths_cfg = cfg.setdefault("paths", {})
    data_root = resolve_project_path(paths_cfg.get("data_root", "data"))
    output_root = resolve_project_path(paths_cfg.get("output_root", "outputs"))
    paths_cfg["data_root"] = str(data_root)
    paths_cfg["output_root"] = str(output_root)

    # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    # ┃ 1) CONFIG & ASSET SELECTION                                           ┃
    # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    # ┏━━━━━━━━━━ 1.a) For the model architecture parameters i.e. CNN, Transformer, MLP ━━━━━━━━━━┓
    architecture_cfg = {"UP": cfg["model_up"], "DN": cfg["model_dn"]}

    # ┏━━━━━━━━━━ 1.b) For the training hyper-parameters, i.e. learning rate, batch size, etc. ━━━━━━━━━━┓
    train_cfg = {"UP": cfg["train_up"], "DN": cfg["train_dn"]}

    # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    # ┃ 2) CONFIGURATION PARAMETERS & DIRECTORY STRUCTURE                     ┃
    # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    # ┏━━━━━━━━━━ 2.a) Data Source & Asset Configuration ━━━━━━━━━━┓
    granularity_usual = cfg["training_mode"]["granularity_usual"]
    granularity_slug  = granularity_usual.replace(" ", "").replace("-", "").lower()
    loss_type         = cfg["training_mode"]["loss_function"].lower()
    provider          = cfg["dataset"]["source"].capitalize()
    cv_usual          = cfg["training_mode"]["cv_usual"]
    seq_len           = cfg["sequence_length"]
    meta_label_usual  = cfg["training_mode"]["meta_label_usual"].lower()
    meta_suffix       = "FP" if meta_label_usual == "fp" else "TP"
    meta_dir_suffix   = "og" if meta_label_usual == "original" else meta_label_usual
    granularity_slug_with_meta = f"{granularity_slug}_{meta_dir_suffix}"

    # ┏━━━━━━━━━━ 2.b) Selective Classification Configuration ━━━━━━━━━━┓
    threshold_cfg    = cfg["training_mode"].get("threshold", {})
    policy           = threshold_cfg["policy"].lower()
    alpha_cfg        = float(threshold_cfg["alpha"])
    fbeta_cfg        = float(threshold_cfg["fbeta"])
    gating_mode      = threshold_cfg["gating"].lower()
    min_coverage_cfg = float(threshold_cfg["min_coverage"])
    min_selected_cfg = float(threshold_cfg["min_selected_count"])
    # ┏━━━━━━━━━━ 2.c) Data Paths ━━━━━━━━━━┓
    csv_path = dataset_path(cfg["dataset"]["source"],
                            cfg["dataset"]["type"].capitalize(),
                            cfg["dataset"]["symbol"],
                            "up",
                            granularity = granularity_usual,
                            meta_label_mode = meta_label_usual,
                            data_root = data_root)


    # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    # ┃ 3) DEVICE & SEEDS                                                     ┃
    # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    # ┏━━━━━━━━━━ Select GPU if available, otherwise CPU ━━━━━━━━━━┓
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    # For Personal Mac
    # device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    # ┏━━━━━━━━━━ Reproducibility ━━━━━━━━━━┓
    seed_everything(seed)

    # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    # ┃ 4) DATA PREPARATION                                                   ┃
    # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    normal_task = cfg["training_mode"]["normal_task"].upper()

    # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    # ┃ 5) DATALOADERS & CRITERION                                            ┃
    # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    for task in [normal_task]:
        # ┏━━━━━━━━━━ Accessing Configuration per Task ━━━━━━━━━━┓
        model_cfg = architecture_cfg[task]
        trainer_cfg = train_cfg[task]
        cm_labels = (f'No_{meta_suffix}_{task}', f'{meta_suffix}_{task}')
        beta_metric = trainer_cfg["fbeta"]

        # ┏━━━━━━━━━━ Extracting Column & Context Features ━━━━━━━━━━┓
        column_features = task_features(cfg, "column_features", task)
        context_features = task_features(cfg, "context_features", task)

        # ┏━━━━━━━━━━ Merge of Data [UP & DN CSVs from M1] ━━━━━━━━━━┓
        df_asset = merge_meta_targets(asset_type       = cfg["dataset"]["type"],
                                      asset            = cfg["dataset"]["symbol"],
                                      data_dir         = str(Path(csv_path).parent),
                                      output_dir       = None,
                                      column_features  = column_features,
                                      context_features = context_features,
                                      meta_label_mode  = meta_label_usual)
        

        # ┏━━━━━━━━━━ Counting Meta-Labels [Positives & Negatives] ━━━━━━━━━━┓
        meta_columns = (f"is{meta_suffix}_UP", f"is{meta_suffix}_DN")
        meta_up_col, meta_dn_col = meta_columns
        meta_counts = count_meta_targets(df_asset, columns = meta_columns, task = task)
        print("Meta-Label counts:")
        for column, stats in meta_counts.items():
            formatted = ", ".join(f"{label}: {count}" for label, count in stats.items())
            print(f"  {column}: {formatted}")
        
        # ┏━━━━━━━━━━ 5.a) Preparation of Dataloaders and Training/Validation/Testing Splits with corresponding Criterion ━━━━━━━━━━┓
        # ┏━━━━━━━━━━ Preparation of Dataloaders ━━━━━━━━━━┓
        dataset_tensor = prepare_dataset(df_asset, 
                                         seq_len          = seq_len,
                                         column_features  = column_features,
                                         context_features = context_features,
                                         meta_label_mode  = meta_label_usual,
                                         task             = task)
        window_dates = np.asarray(dataset_tensor.window_dates)

        
        # ┏━━━━━━━━━━ 5.b) Sanity-check window count vs raw rows ━━━━━━━━━━┓
        expected_samples = len(df_asset) - seq_len + 1
        total_samples = len(dataset_tensor)    
        if total_samples != expected_samples:
            # This means prepare_dataset does something more than “raw sliding window”
            # (e.g. drops rows, filters NaNs, merges context). Downstream we must be
            # careful when aligning M1 or original indices.
            print(f"[WARN] Dataset tensor contains {total_samples} windows but expected {expected_samples}."
                  " Check for downstream alignment issues.")

        # ┏━━━━━━━━━━ 5.c) Building M1 Mask Aligned to Dataset Windows ━━━━━━━━━━┓
        m1_window = build_m1_window(df_asset, task, seq_len, total_samples)

        def _policy_mask_from_indices(indices: Sequence[int]) -> np.ndarray:
            """Return the policy mask aligned either via indices or timestamps."""
            if m1_window is None:
                return None
            idx_array = np.asarray(indices, dtype=int)
            if meta_label_usual in {"fp", "tp"}:
                meta_col = f"is{meta_suffix}_{task}"
                if meta_col not in df_asset.columns:
                    raise KeyError(f"Column '{meta_col}' not found in merged dataset for task '{task}'.")
                subset_dates = window_dates[idx_array]
                meta_vals = df_asset.loc[subset_dates, meta_col].to_numpy(dtype=float)
                return np.isfinite(meta_vals)
            return m1_window[idx_array].astype(bool)

        # ┏━━━━━━━━━━ 5.d) Splits and Criterion ━━━━━━━━━━┓
        seed_everything(seed)  # Re-seed for reproducibility
        print(f"\n────────── {task} model ──────────")
        train_folds, test_loader = build_loaders(ds               = dataset_tensor,
                                                 cross_validation = cv_usual,
                                                 target           = task,
                                                 props            = cfg["training_mode"]["cross_val_props"],
                                                 train_frac       = cfg["splits"]["train"],
                                                 val_frac         = cfg["splits"]["val"],
                                                 test_frac        = cfg["splits"]["test"],
                                                 batch_size       = trainer_cfg["batch_size"],
                                                 loss_type        = loss_type,
                                                 focal_gamma      = cfg["training_mode"]["focal_gamma"],
                                                 focal_alpha      = cfg["training_mode"]["focal_alpha"],
                                                 device           = device)
        
        # ┏━━━━━━━━━━ 5.e) Get the model parameters ━━━━━━━━━━┓
        model_kwargs = dict(
            # ┏━━━━━━━━━━ CNN Parameters ━━━━━━━━━━┓
            cnn_embed_dim = model_cfg["cnn_embed_dim"],
            cnn_kernel    = model_cfg["cnn_kernel"],
            cnn_stride    = model_cfg["cnn_stride"],
            p_pos_drop    = model_cfg["p_pos_drop"],
            nb_features   = len(column_features),
            
            # ┏━━━━━━━━━━ Transformer Parameters ━━━━━━━━━━┓
            trans_heads   = model_cfg["transformer"]["heads"],
            trans_ff      = model_cfg["transformer"]["ffn_dim"] * model_cfg["cnn_embed_dim"][-1],
            trans_layers  = model_cfg["transformer"]["layers"],
            trans_dropout = model_cfg["transformer"]["dropout"],
            trans_activ   = model_cfg["transformer"]["activation"],
            
            # ┏━━━━━━━━━━ Classifier Parameters ━━━━━━━━━━┓
            mlp_hidden    = model_cfg["classifier"]["mlp_hidden"],
            mlp_dropout   = model_cfg["classifier"]["mlp_dropout"],
            mlp_activ     = model_cfg["classifier"]["mlp_activation"],
            mlp_pooling   = model_cfg["classifier"]["mlp_pooling"],
            

            # ┏━━━━━━━━━━ Training Mode Parameters ━━━━━━━━━━┓
            num_classes   = 1 if cfg["training_mode"]["loss_function"] == "bce" else 2,
            padding       = cfg["training_mode"]["padding"],
            context_len   = cfg["sequence_length"] + len(context_features)
        )
        
        # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        # ┃ 6) LOOP OVER FOLDS: MODEL, OPTIMIZER, SCHEDULER & EARLY STOPPING      ┃
        # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
        n_folds        = len(train_folds)
        fold_losses    = []

        # ┏━━━━━━━━━━ 6.a) Store validation losses & Paths ━━━━━━━━━━┓
        #val_losses_all_folds = []
        task_root = (output_root / "Usual" / provider
                                                                / cfg["dataset"]["symbol"] 
                                                               / task 
                                                              / granularity_slug_with_meta)
        
        tb_root = (output_root / "Usual" / "Tensorboard" / provider 
                                                                              / cfg["dataset"]["symbol"])
        # ┏━━━━━━━━━━ Main Loop for Training & Validation ━━━━━━━━━━┓
        for fold_idx, (train_loader, val_loader, crit) in enumerate(train_folds):
            # ┏━━━━━━━━━━ 6.b) Checkpoints & Runs and Tensorboard Folders Creation ━━━━━━━━━━┓
            run_stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            checkpoint_dir = task_root / f"Run_{run_stamp}"
            tensorboard_dir = tb_root / f"{task}" / granularity_slug_with_meta / f"Run_{run_stamp}"

            checkpoint_dir.mkdir(parents = True, exist_ok = True)
            tensorboard_dir.mkdir(parents = True, exist_ok = True)

            # ┏━━━━━━━━━━ 6.c) Instantiate the model (fresh for each fold) ━━━━━━━━━━┓
            seed_everything(seed)
            model = CTTSModel(**model_kwargs).to(device)

            # ┏━━━━━━━━━━ 6.d) Optimizer ━━━━━━━━━━┓
            optim_cfg = trainer_cfg
            optimizer = get_optimizer(model.parameters(), optim_cfg)

            # ┏━━━━━━━━━━ 6.e) Scheduler (optional - remove it when scheduler chosen) ━━━━━━━━━━┓
            sch_cfg = trainer_cfg['scheduler']
            scheduler = make_scheduler(optimizer, sch_cfg, trainer_cfg["max_epochs"])

            # ┏━━━━━━━━━━ 6.f) Early Stoppings ━━━━━━━━━━┓
            checkpoint_path = checkpoint_dir / f"{task}_best.pt"
            stopper = EarlyStopping(patience   = trainer_cfg["patience"],
                                    verbose    = False,
                                    delta      = 1e-4,
                                    path       = str(checkpoint_path),
                                    just_count = True)
            
            # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
            # ┃ 7) TENSORBOARD WRITERS                                                ┃
            # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
            # ┏━━━━━━━━━━ TensorBoard Terminal Command  ━━━━━━━━━━┓
            # ┏━━━━━━━━━━ 1) ls ~/miniconda3/envs/CTTS/bin/tensorboard  ━━━━━━━━━━┓
            # ┏━━━━━━━━━━ 2) ~/miniconda3/envs/CTTS/bin/tensorboard --logdir Output/runs  ━━━━━━━━━━┓
            writer = SummaryWriter(str(tensorboard_dir))
            
            # ┏━━━━━━━━━━ Best Metrics (temporary) ━━━━━━━━━━┓
            best_loss  = float('inf')
            best_state = None        

            # ┏━━━━━━━━━━ Train & Validation ━━━━━━━━━━┓
            epoch_iter = tqdm(range(trainer_cfg["max_epochs"]),
                              desc          = f"🌀 {task} Fold {fold_idx + 1}/{n_folds}",
                              leave         = True,
                              dynamic_ncols = True)

            for epoch in epoch_iter:
                # ┏━━━━━━━━━━ Train ━━━━━━━━━━┓
                train_result = epoch_loop(model,
                                          train_loader,
                                          crit,
                                          device,
                                          mode = "train",
                                          task_name = task,
                                          optimizer = optimizer,
                                          bce_thr = 0.5,
                                          amp = True,
                                          clip_grad = 1.0,
                                          beta = trainer_cfg["fbeta"],
                                          return_raw = False)
                
                # ┏━━━━━━━━━━ Train Loss ━━━━━━━━━━┓
                train_loss = train_result["loss"]
                
                # ┏━━━━━━━━━━ Validation ━━━━━━━━━━┓
                val_result = epoch_loop(model,
                                        val_loader,
                                        crit,
                                        device,
                                        mode = "val",
                                        task_name = task,
                                        optimizer = None,
                                        bce_thr = 0.5,
                                        amp = False,
                                        clip_grad = 0.0,
                                        beta = trainer_cfg["fbeta"],
                                        return_raw = False)
                
                # ┏━━━━━━━━━━ Validation Loss & Metrics ━━━━━━━━━━┓
                val_loss = val_result["loss"]
                vacc     = val_result["acc"]
                vprec    = val_result["prec"]
                vrec     = val_result["rec"]
                vf1      = val_result["f1"]
                vfbeta   = val_result["fbeta"]
                            
                # ┏━━━━━━━━━━ Log to TensorBoard if last fold ━━━━━━━━━━┓
                if writer is not None:
                    writer.add_scalar("Loss/train",      train_loss, epoch)
                    writer.add_scalar("Loss/validation", val_loss  , epoch)
                
                epoch_iter.set_postfix(train = f"{train_loss:.4f}", val = f"{val_loss:.4f}")

                # ┏━━━━━━━━━━ Early Stopping ━━━━━━━━━━┓
                stopper(val_loss, model)
                step_scheduler(scheduler, epoch, val_loss)
                if stopper.early_stop:
                    break
                
                # ┏━━━━━━━━━━ If this epoch is the best so far, record its metrics ━━━━━━━━━━┓
                if val_loss < best_loss:
                    best_loss  = val_loss
                    best_acc   = vacc
                    best_prec  = vprec
                    best_f1    = vf1
                    best_rec   = vrec
                    best_fbeta = vfbeta
                    best_state = copy.deepcopy(model.state_dict())
            
            epoch_iter.close()
            
            # ┏━━━━━━━━━━ Close writer after last fold training ━━━━━━━━━━┓
            if writer is not None:
                writer.close()
                writer = None

            # ┏━━━━━━━━━━ Store Best Validation Loss (per fold) ━━━━━━━━━━┓
            fold_losses.append(best_loss)

            # ┏━━━━━━━━━━ Store Best Precision in Test ━━━━━━━━━━┓
            if fold_idx == n_folds - 1:
                # ┏━━━━━━━━━━ Reload the best state into model ━━━━━━━━━━┓
                assert best_state is not None, "No Model saved during Training"
                model.load_state_dict(best_state)

                # ┏━━━━━━━━━━ Save that best state-dict to disk ━━━━━━━━━━┓
                ckpt_path = checkpoint_dir / f'{task}_best.pt'
                torch.save(best_state, ckpt_path)

                # ┏━━━━━━━━━━ Validation evaluation (raw outputs cached) ━━━━━━━━━━┓
                val_eval = epoch_loop(model,
                                      val_loader,
                                      crit,
                                      device,
                                      mode = "val",
                                      task_name = task,
                                      optimizer = None,
                                      bce_thr = 0.5,
                                      amp = False,
                                      clip_grad = 0.0,
                                      beta = trainer_cfg["fbeta"],
                                      return_raw = True)
                
                # ┏━━━━━━━━━━ Extraction of raw predictions & probabilities ━━━━━━━━━━┓
                vpreds_raw = val_eval["preds"]
                vtargets   = val_eval["targets"]
                vprobs     = val_eval["probs"]
                if vprobs is None:
                    raise ValueError("Validation probabilities were not produced; ensure the criterion supports probability extraction.")

                # ┏━━━━━━━━━━ Adapt raw probabilities to Gating Policy & Threshold Uniqueness with endpoints ━━━━━━━━━━┓ 
                val_scores = build_scores(vprobs, gating_mode)

                # ┏━━━━━━━━━━ Aligning Validation Indices for M1 Masking (Aware of Subsets) ━━━━━━━━━━┓
                if hasattr(val_loader.dataset, "indices"):
                    val_indices = np.asarray(val_loader.dataset.indices, dtype=int)
                else:
                    print("[WARN] Validation dataset has no 'indices' attribute; "
                          "falling back to np.arange(len(scores)). Alignment with M1 window may be incorrect.")
                    val_indices = np.arange(val_scores.shape[0], dtype=int)

                # ┏━━━━━━━━━━ Extracting Validation Indices for M1 Masking (Aware of Subsets) ━━━━━━━━━━┓
                val_m1_mask = _policy_mask_from_indices(val_indices)

                # ┏━━━━━━━━━━ Sanity Check ━━━━━━━━━━┓
                if val_m1_mask is None:
                    raise ValueError("M1 mask unavailable for risk_budget policy during validation.")

                # ┏━━━━━━━━━━ Align M1 Window to M2 Validation Subset ━━━━━━━━━━┓
                if not val_m1_mask.any():
                    # ┏━━━━━━━━━━ Protects against empty Val splits ━━━━━━━━━━┓
                    raise ValueError("No M1-positive samples in validation split for risk_budget policy.")
                
                # ┏━━━━━━━━━━ Apply mask to both targets (GT) and scores (probs) to keep them aligned ━━━━━━━━━━┓
                val_targets_policy = vtargets[val_m1_mask]
                val_scores_policy  = val_scores[val_m1_mask]
                
                # ┏━━━━━━━━━━ Validation Metrics [w/o Optimized Threshold] ━━━━━━━━━━┓
                val_preds_default   = vpreds_raw[val_m1_mask]
                val_metrics_default = compute_basic_metrics(val_targets_policy, val_preds_default, beta_metric)

                # ┏━━━━━━━━━━ Sanity Alignment Check ━━━━━━━━━━┓
                if val_targets_policy.shape[0] != val_scores_policy.shape[0]:
                    raise AssertionError(f"Validation targets ({val_targets_policy.shape[0]}) and scores ({val_scores_policy.shape[0]}) mismatch.")
                
                # ┏━━━━━━━━━━ Gating Policy Probabilities & Threshold Uniqueness with endpoints ━━━━━━━━━━┓ 
                finite_scores     = val_scores_policy[np.isfinite(val_scores_policy)]
                if finite_scores.size == 0:
                    # No usable scores, at least create a trivial curve
                    thresholds_unique = np.array([0.0, 1.0])
                    print("[WARN] No finite validation scores available; using trivial thresholds {0, 1}.")
                else:
                    thresholds_unique = np.unique(finite_scores)
                    thresholds_unique = np.unique(np.concatenate([thresholds_unique, np.array([0.0, 1.0])]))
                
                # ┏━━━━━━━━━━ Risk/Coverage Curve Creation with Sweeping Thresholds ━━━━━━━━━━┓ 
                val_curve = collect_risk_coverage_curve(y_true = val_targets_policy,
                                                        y_score = val_scores_policy, 
                                                        thresholds = thresholds_unique, 
                                                        include_error_counts = True)

                # ┏━━━━━━━━━━ Area under the validation risk–coverage curve ━━━━━━━━━━┓ 
                val_aurc = area_under_risk_coverage(curve = val_curve)

                # ┏━━━━━━━━━━ Threshold Policy Selection ━━━━━━━━━━┓
                if policy == "risk_budget":
                    # ┏━━━━━━━━━━ Coverage associated to user-defined Risk ━━━━━━━━━━┓ 
                    selection = coverage_at_risk(curve        = val_curve,
                                                 max_risk     = alpha_cfg,
                                                 min_coverage = min_coverage_cfg,
                                                 min_selected = min_selected_cfg)
                     
                    # ┏━━━━━━━━━━ Best Threshold (according to Policy) ━━━━━━━━━━┓ 
                    selected_tau = float(selection["threshold"])
                    
                    # ┏━━━━━━━━━━ Risk & Coverage corresponding metrics to Selected Threshold ━━━━━━━━━━┓ 
                    selection_metric = {"Risk_Constraint_Satisfied": bool(selection["constraint_satisfied"])}
                    
                    if not selection_metric["Risk_Constraint_Satisfied"]:
                        print(f"[WARN] Risk Budget infeasible at Alpha - {alpha_cfg:.3f}. Then, using lowest-risk fallback threshold.")

                elif policy == "f_beta":
                    # ┏━━━━━━━━━━ Optimized Threshold with its FBeta Metric ━━━━━━━━━━┓ 
                    selected_tau, best_metric = select_threshold_fbeta(val_targets_policy,
                                                                       val_scores_policy,
                                                                       thresholds_unique,
                                                                       fbeta_cfg)
                else:
                    raise ValueError(f"Unknown threshold policy: {policy}")

                # ┏━━━━━━━━━━ Optimized/Best Threshold for Validation Predictions [Last Fold] ━━━━━━━━━━┓ 
                eval_val      = evaluate_threshold(val_targets_policy, val_scores_policy, selected_tau)
                val_preds_tau = eval_val.pop("predictions")

                # ┏━━━━━━━━━━ Validation Metrics [Optimized Threshold] ━━━━━━━━━━┓
                val_metrics_tau = compute_basic_metrics(val_targets_policy, val_preds_tau, beta_metric)

                # ┏━━━━━━━━━━ Validation Confusion Matrix & Metrics [Not Optimized Threshold & Masked] ━━━━━━━━━━┓
                plot_cm_with_metrics(vpreds_raw[val_m1_mask],
                                     vtargets[val_m1_mask],
                                     labels         = cm_labels,
                                     title          = f'M2_{task} — Val',
                                     out_dir        = checkpoint_dir,
                                     best_threshold = None,
                                     cmap           = "Oranges")

                # ┏━━━━━━━━━━ Validation Confusion Matrix & Metrics [Optimized Threshold & Masked] ━━━━━━━━━━┓
                plot_cm_with_metrics(val_preds_tau,
                                     vtargets[val_m1_mask],
                                     labels         = cm_labels,
                                     title          = f'M2_{task} — Val',
                                     out_dir        = checkpoint_dir,
                                     best_threshold = selected_tau,
                                     cmap           = "Greens")

                # ┏━━━━━━━━━━ Plotting Risk & Coverage Curve ━━━━━━━━━━┓
                val_rc_png = checkpoint_dir / f"M1+M2_{task}_Val_RiskCoverage.png"
                plot_coverage_risk_curve(curve = val_curve, 
                                         label = None, #f"Gating: {gating_mode}", 
                                         save_path = str(val_rc_png), 
                                         show = False,
                                         asset_name = cfg["dataset"]["symbol"],
                                         #highlight_point = (eval_val["coverage"], eval_val["risk"]),
                                         highlight_point = None,
                                         #highlight_text = f"τ* = {selected_tau:.3f}",
                                         highlight_text = None,
                                         smooth = True,
                                         smooth_method = "spline",
                                         smooth_points = 300,
                                         smooth_s_factor = 0.1 * len(val_curve["coverage"]))
                
                # ┏━━━━━━━━━━ Summary of Risk & Coverage Analysis ━━━━━━━━━━┓
                curve_rows = [{"Threshold": float(t),
                               "Coverage": float(c),
                               "Risk": float(r),
                               "Selected_Count": int(s),
                               "Error_Count": int(e) if "error_count" in val_curve else None} for t, c, r, s, e in zip(val_curve["thresholds"],
                                                                                                                       val_curve["coverage"],
                                                                                                                       val_curve["risk"],
                                                                                                                       val_curve["selected_count"],
                                                                                                                       val_curve.get("error_count", [0] * len(val_curve["thresholds"])))
                              ]

                dataset_id = f"{cfg['dataset']['source']}_{cfg['dataset']['symbol']}_{granularity_usual}"
                payload = {"Dataset":            dataset_id,
                           "Task":               task,
                           "Policy":             policy,
                           "Gating":             gating_mode,
                           "Val_AURC":           val_aurc,
                           "Val_Tau":            selected_tau,
                           "Val_Coverage@Tau":   eval_val["coverage"],
                           "Val_Risk@Tau":       eval_val["risk"],
                           "Selected_Count@Tau": eval_val["selected_count"]}
                
                # ┏━━━━━━━━━━ Adding Data to Summary of Risk & Coverage Analysis ━━━━━━━━━━┓
                if policy == "risk_budget":
                    payload["Alpha"] = alpha_cfg
                    payload.update(selection_metric)
                    payload["Min_Coverage"] = min_coverage_cfg
                    payload["Min_Selected_Count"] = min_selected_cfg
                
                # ┏━━━━━━━━━━ Adding Data to Summary of FBeta Analysis ━━━━━━━━━━┓
                elif policy == "f_beta":
                    payload["FBeta"]     = fbeta_cfg
                    payload["FBeta@Tau"] = best_metric

                # ┏━━━━━━━━━━ Original split's Model Evaluation on Test Set & Confusion Matrix ━━━━━━━━━━┓
                test_eval = epoch_loop(model,
                                       test_loader,
                                       crit,
                                       device,
                                       mode = "test",
                                       task_name = task,
                                       optimizer = None,
                                       bce_thr = 0.5,
                                       amp = False,
                                       clip_grad = 0.0,
                                       beta = trainer_cfg["fbeta"],
                                       return_raw = True)

                # ┏━━━━━━━━━━ Test Predictions & Probabilities [Not Optimized Threshold] ━━━━━━━━━━┓
                tpreds   = test_eval["preds"]
                ttargets = test_eval["targets"]
                tprobs   = test_eval.get("probs")
                if tprobs is None:
                    raise ValueError("Test probabilities were not produced; ensure the criterion supports probability extraction.")

                # ┏━━━━━━━━━━ Risk & Coverage Scores [Optimized Threshold] ━━━━━━━━━━┓
                test_scores    = build_scores(tprobs, gating_mode)
                
                # ┏━━━━━━━━━━ Aligning Test Indices for M1 Masking (Aware of Subsets) ━━━━━━━━━━┓
                if hasattr(test_loader.dataset, "indices"):
                    test_indices = np.asarray(test_loader.dataset.indices, dtype=int)
                else:
                    print("[WARN] Test dataset has no 'indices' attribute; "
                          "falling back to np.arange(len(scores)). Alignment with M1 window may be incorrect.")
                    test_indices = np.arange(test_scores.shape[0], dtype=int)
                
                if test_indices.size != test_scores.shape[0]:
                    # ┏━━━━━━━━━━ Protects against different lengths ━━━━━━━━━━┓
                    raise ValueError("Test loader indices do not match the number of test scores."
                                     f" Expected {test_scores.shape[0]} entries, got {test_indices.size}.")

                # ┏━━━━━━━━━━ M1 Predictions Aligned to Test Set & Test Mask ━━━━━━━━━━┓
                test_m1_mask = _policy_mask_from_indices(test_indices)

                # ┏━━━━━━━━━━ Sanity Check ━━━━━━━━━━┓
                if test_m1_mask is None:
                    raise ValueError("M1 mask unavailable for risk_budget policy during test.")
                if not test_m1_mask.any():
                    raise ValueError("No M1-positive samples in test split for risk_budget policy.")

                # ┏━━━━━━━━━━ Apply mask to both targets (GT) and scores (probs) to keep them aligned ━━━━━━━━━━┓
                ttargets_policy = ttargets[test_m1_mask]
                test_scores_policy = test_scores[test_m1_mask]

                # ┏━━━━━━━━━━ Test Metrics [w/o Optimized Threshold] ━━━━━━━━━━┓
                test_preds_default   = tpreds[test_m1_mask]
                test_metrics_default = compute_basic_metrics(ttargets_policy, test_preds_default, beta_metric)

                 # ┏━━━━━━━━━━ Final safety: targets and scores must stay aligned ━━━━━━━━━━┓
                if ttargets_policy.shape[0] != test_scores_policy.shape[0]:
                    raise AssertionError(f"Test targets ({ttargets_policy.shape[0]}) and scores ({test_scores_policy.shape[0]}) mismatch.")

                # ┏━━━━━━━━━━ Optimized/Best Threshold for Test Predictions ━━━━━━━━━━┓ 
                eval_test_tau = evaluate_threshold(ttargets_policy, test_scores_policy, selected_tau)
                test_preds_tau = eval_test_tau.pop("predictions")

                # ┏━━━━━━━━━━ Test Metrics [Optimized Threshold] ━━━━━━━━━━┓
                test_metrics_tau = compute_basic_metrics(ttargets_policy, test_preds_tau, beta_metric)

                # ┏━━━━━━━━━━ Ensure predictions length matches test length ━━━━━━━━━━┓
                if test_preds_tau.shape[0] != ttargets[test_m1_mask].shape[0]:
                    raise AssertionError("Length of test predictions at τ does not match test targets.")
                
    
                # ┏━━━━━━━━━━ Additional Information for Payload: R&C, Validation and Test ━━━━━━━━━━┓
                payload["Test_Coverage@Tau"]       = eval_test_tau["coverage"]
                payload["Test_Risk@Tau"]           = eval_test_tau["risk"]
                payload["Test_Selected_Count@Tau"] = eval_test_tau["selected_count"]
                
                payload.update({"Val_Accuracy":        val_metrics_default["accuracy"],
                                "Val_Precision":       val_metrics_default["precision"],
                                "Val_Recall":          val_metrics_default["recall"],
                                "Val_F1":              val_metrics_default["f1"],
                                "Val_FBeta":           val_metrics_default["fbeta"],
                                "Val_Accuracy@Tau":    val_metrics_tau["accuracy"],
                                "Val_Precision@Tau":   val_metrics_tau["precision"],
                                "Val_Recall@Tau":      val_metrics_tau["recall"],
                                "Val_F1@Tau":          val_metrics_tau["f1"],
                                "Val_FBeta@Tau":       val_metrics_tau["fbeta"],
                                "Test_Accuracy":       test_metrics_default["accuracy"],
                                "Test_Precision":      test_metrics_default["precision"],
                                "Test_Recall":         test_metrics_default["recall"],
                                "Test_F1":             test_metrics_default["f1"],
                                "Test_FBeta":          test_metrics_default["fbeta"],
                                "Test_Accuracy@Tau":   test_metrics_tau["accuracy"],
                                "Test_Precision@Tau":  test_metrics_tau["precision"],
                                "Test_Recall@Tau":     test_metrics_tau["recall"],
                                "Test_F1@Tau":         test_metrics_tau["f1"],
                                "Test_FBeta@Tau":      test_metrics_tau["fbeta"]})

                # ┏━━━━━━━━━━ Test Confusion Matrix & Metrics [Not Optimized Threshold] ━━━━━━━━━━┓
                plot_cm_with_metrics(tpreds[test_m1_mask],
                                     ttargets[test_m1_mask],
                                     labels         = cm_labels,
                                     title          = f'M2_{task} — Test',
                                     out_dir        = checkpoint_dir,
                                     best_threshold = None,
                                     cmap           = "Blues")
                
                # ┏━━━━━━━━━━ Test Confusion Matrix & Metrics [Optimized Threshold] ━━━━━━━━━━┓
                plot_cm_with_metrics(test_preds_tau,
                                     ttargets[test_m1_mask],
                                     labels         = cm_labels,
                                     title          = f'M2_{task} — Test',
                                     out_dir        = checkpoint_dir,
                                     best_threshold = selected_tau,
                                     cmap           = "Purples")

                # ┏━━━━━━━━━━ Export into CSV Test Predictions ━━━━━━━━━━┓
                print("\nEnriched Results: ")
                tpreds_tau_aligned = np.zeros_like(tpreds)
                tpreds_tau_aligned[test_m1_mask] = test_preds_tau
                _, consensus_metrics = export_predictions(df_asset        = df_asset,
                                                          dataset_tensor  = dataset_tensor,
                                                          tpreds          = tpreds,
                                                          tpreds_tau      = tpreds_tau_aligned,
                                                          cfg             = cfg,
                                                          checkpoint_dir  = checkpoint_dir,
                                                          tprobs          = tprobs,
                                                          meta_label_mode = meta_label_usual)
                
                # ┏━━━━━━━━━━ Additional Information for Payload: Consensus Metrics from M1+M2 and R&C Curves ━━━━━━━━━━┓
                payload.update(consensus_metrics)
                payload["Val_Curves"] = curve_rows

                # ┏━━━━━━━━━━ Path & Save Summary ━━━━━━━━━━┓
                payload_json = checkpoint_dir / f"M2_{task}_R&C_Analysis.json"
                save_metrics(payload, str(payload_json))

                # ┏━━━━━━━━━━ Meta-Labeling Consensus ━━━━━━━━━━┓
                plot_meta_labeling_consensus(cfg = cfg,
                                             checkpoint_dir = checkpoint_dir,
                                             best_threshold = selected_tau)

            # ┏━━━━━━━━━━ Empty Caché ━━━━━━━━━━┓
            torch.cuda.empty_cache()
                        

def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description = "Train CTTS model")
    
    # ┏━━━━━━━━━━ Config Argument ━━━━━━━━━━┓
    parser.add_argument("--config",
                        type    = str,
                        default = str(PROJECT_ROOT / "configs" / "btc" / "chronos_up.yaml"),
                        help    = "Path to the YAML configuration file")
    
    # ┏━━━━━━━━━━ Seed Argument ━━━━━━━━━━┓
    parser.add_argument("--seed",
                        type    = int,
                        default = DEFAULT_SEED,
                        help    = "Random seed for data splits and model init")
    
    # ┏━━━━━━━━━━ Extracting Arguments & Paths ━━━━━━━━━━┓
    args = parser.parse_args(argv)
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = (Path.cwd() / cfg_path).resolve()

    # ┏━━━━━━━━━━ Config Path & Load ━━━━━━━━━━┓
    if not cfg_path.exists():
        raise FileNotFoundError(f"Could not find config file at {cfg_path}")

    with cfg_path.open("r") as fh:
        cfg = yaml.safe_load(fh)

    # ┏━━━━━━━━━━ Run Training ━━━━━━━━━━┓
    run_training(cfg, seed = args.seed)


if __name__ == "__main__":
    main()
