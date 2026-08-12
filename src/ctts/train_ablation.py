
#!/usr/bin/env python
"""Run an ablation study over context features using the train.py pipeline."""
from __future__ import annotations

# ┏━━━━━━━━━━ Standard libs ━━━━━━━━━━┓
import argparse
import copy
import yaml

# ┏━━━━━━━━━━ Project imports ━━━━━━━━━━┓
from pathlib import Path
from .paths import PROJECT_ROOT, resolve_project_path
from .train import run_training
from .utils.ablation_utils import (resolve_baseline_features,
                                  ensure_clean_dir,
                                  reorganize_outputs,
                                  slugify,
                                  feature_groups,
                                  write_metadata)


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ BASELINE DEFINITIONS                                                  ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
# For each ablation task, define the minimal/always-on feature set.
BASELINE_FEATURES_BY_TASK = {"UP": ["m1_prediction"],
                             "DN": ["m1_prediction", "m1_dn"]}

# Human-friendly group labels by combo size (used for folder structure).
GROUP_LABELS = {0: "0_Baseline",
                1: "1_Singles",
                2: "2_Pairs",
                3: "3_Trios",
                4: "4_Quartets",
                5: "5_Quintets",
                6: "6_Sextets"}


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ MAIN ENTRYPOINT                                                       ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
def main() -> None:
    # ┏━━━━━━━━━━ CLI: config path ━━━━━━━━━━┓
    parser = argparse.ArgumentParser(description="Run context-feature ablation study")
    parser.add_argument("--config",
                        type = str,
                        default = str(PROJECT_ROOT / "configs" / "btc" / "chronos_up.yaml"),
                        help = "Base configuration file to use as template")
    args = parser.parse_args()

    # ┏━━━━━━━━━━ Resolve and validate config file ━━━━━━━━━━┓
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = (Path.cwd() / cfg_path).resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Could not find config file at {cfg_path}")

    # ┏━━━━━━━━━━ Load base configuration ━━━━━━━━━━┓
    base_cfg = yaml.safe_load(cfg_path.open())

    # ┏━━━━━━━━━━ Read ablation knobs from config ━━━━━━━━━━┓
    training_mode = base_cfg.get("training_mode", {})
    task = training_mode["ablation_task"]
    # Prefer task-specific context key; fallback to generic if absent.
    context_key = f"context_features_{task.lower()}"
    context_features = base_cfg.get(context_key) or base_cfg.get("context_features", [])

    # ┏━━━━━━━━━━ Determine baseline vs. additional features ━━━━━━━━━━┓
    # Baseline = minimal features we always include for this task.
    baseline_features = resolve_baseline_features(task, context_features, BASELINE_FEATURES_BY_TASK)
    # Additional = candidates to ablate/combine on top of baseline.
    additional_features = [feat for feat in context_features if feat not in baseline_features]

    # ┏━━━━━━━━━━ Prepare ablation output directories ━━━━━━━━━━┓
    output_root = resolve_project_path(base_cfg["paths"]["output_root"])
    
    provider = base_cfg["dataset"]["source"].capitalize()
    ablation_root = output_root / "Ablation" / provider / task
    ablation_root.mkdir(parents=True, exist_ok=True)
    
    # Useful for metadata/renaming after each run.
    symbol = base_cfg["dataset"]["symbol"]
    
    # ┏━━━━━━━━━━ Iterate through feature groups (by size & combos) ━━━━━━━━━━┓
    # feature_groups(additional_features) yields (size, combination_iterable
    for size, combo in feature_groups(additional_features):
        # Group folder name (e.g., "2_Pairs")
        group_name = GROUP_LABELS.get(size, f"{size}_Combos")

        # Actual feature set for this run = baseline + current combo
        combo_features = list(baseline_features) + list(combo)

        # Slug for filesystem-safe run naming
        combo_slug = slugify(combo)
        run_dir = ablation_root / group_name / combo_slug

        print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"Group: {group_name} | Features: {combo_features}")
        print(f"Output directory: {run_dir}")

        # ┏━━━━━━━━━━ Ensure a clean run directory ━━━━━━━━━━┓
        ensure_clean_dir(run_dir)

        # ┏━━━━━━━━━━ Build a config variant for this combo ━━━━━━━━━━┓
        cfg_variant = copy.deepcopy(base_cfg)
        
        # Overwrite task-specific context with the combo features
        cfg_variant[context_key] = combo_features
        
        # Keep generic context_features in sync if present
        if "context_features" in cfg_variant:
            cfg_variant["context_features"] = combo_features
        
        # ┏━━━━━━━━━━ Normalize training_mode fields ━━━━━━━━━━┓
        cfg_variant.setdefault("training_mode", {})
        cfg_variant["training_mode"]["normal_task"] = task

        # Carry through ablation toggles into explicit *usual* fields for bookkeeping
        meta_label_ablation = cfg_variant["training_mode"]["meta_label_ablation"]
        cfg_variant["training_mode"]["meta_label_usual"] = meta_label_ablation
        
        # Transfering granularity to training usage
        granularity_ablation = cfg_variant["training_mode"]["granularity_ablation"]
        if granularity_ablation is not None:
            cfg_variant["training_mode"]["granularity_usual"] = granularity_ablation
        
        # Point outputs to this run's folder
        cfg_variant["paths"]["output_root"] = str(run_dir)

        # ┏━━━━━━━━━━ Execute training for this feature combo ━━━━━━━━━━┓
        run_training(cfg_variant)

        # ┏━━━━━━━━━━ Reorganize outputs and persist metadata ━━━━━━━━━━┓
        reorganize_outputs(run_dir, provider, symbol)

        # Persist the exact columns used (context may map to actual columns)
        column_key = f"column_features_{task.lower()}"
        column_features = cfg_variant.get(column_key) or cfg_variant.get("column_features", [])
        write_metadata(run_dir / "metadata.json",
                       task            = task,
                       column_features = column_features,
                       baseline        = baseline_features,
                       extras          = combo,
                       context         = combo_features)


if __name__ == "__main__":
    main()
