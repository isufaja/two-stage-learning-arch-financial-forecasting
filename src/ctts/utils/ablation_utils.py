"""Helpers for ablation studies."""
from __future__ import annotations

import itertools
import json
import shutil
from pathlib import Path
from typing import Iterable, Tuple


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ BASELINE FEATURE UTILITIES                                            ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

def resolve_baseline_features(task: str, available: Iterable[str], baseline_map: dict[str, list[str]]) -> list[str]:
    """Validate baseline context features for a task and return them."""
    task = task.upper()
    if task not in baseline_map:
        raise ValueError(f"Unsupported task '{task}'. Expected one of {list(baseline_map)}")

    baseline = baseline_map[task]
    missing = [feat for feat in baseline if feat not in available]
    if missing:
        raise ValueError(
            "Baseline context features are missing from config for task "
            f"{task}: {', '.join(missing)}"
        )
    return baseline


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ FILESYSTEM HELPERS                                                    ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

def ensure_clean_dir(path: Path) -> None:
    """Remove existing directory and recreate it empty."""
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def reorganize_outputs(run_root: Path, provider: str, symbol: str) -> None:
    """Reorganize training outputs created under the standard Usual/ layout."""
    usual_root = run_root / "Usual"
    tensorboard_root = usual_root / "Tensorboard"
    provider_root = usual_root / provider

    if provider_root.exists():
        for symbol_dir in provider_root.iterdir():
            if not symbol_dir.is_dir():
                continue
            for task_dir in symbol_dir.iterdir():
                if not task_dir.is_dir():
                    continue
                for granularity_dir in task_dir.iterdir():
                    if not granularity_dir.is_dir():
                        continue
                    dest_base = run_root / provider / symbol_dir.name / task_dir.name / granularity_dir.name
                    legacy_dir = granularity_dir / "Run"
                    run_dirs = []
                    if legacy_dir.exists():
                        run_dirs.append(legacy_dir)
                    run_dirs.extend(
                        sorted(
                            (
                                p
                                for p in granularity_dir.iterdir()
                                if p.is_dir() and p.name.startswith("Run_")
                            ),
                            key=lambda path: path.name,
                        )
                    )

                    for run_dir in run_dirs:
                        dest_run = dest_base if run_dir.name == "Run" else dest_base / run_dir.name
                        dest_run.mkdir(parents=True, exist_ok=True)
                        for item in run_dir.iterdir():
                            target = dest_run / item.name
                            if target.exists():
                                if target.is_dir():
                                    shutil.rmtree(target)
                                else:
                                    target.unlink()
                            shutil.move(str(item), target)
                        shutil.rmtree(run_dir)

                    shutil.rmtree(granularity_dir)
                shutil.rmtree(task_dir)
            shutil.rmtree(symbol_dir)
        shutil.rmtree(provider_root)

    if tensorboard_root.exists():
        tb_provider_root = tensorboard_root / provider
        if tb_provider_root.exists():
            dest_tb = run_root / "Tensorboard" / provider
            dest_tb.mkdir(parents=True, exist_ok=True)
            for item in tb_provider_root.iterdir():
                target = dest_tb / item.name
                if target.exists():
                    if target.is_dir():
                        shutil.rmtree(target)
                    else:
                        target.unlink()
                shutil.move(str(item), target)
            shutil.rmtree(tb_provider_root)
        if tensorboard_root.exists() and not any(tensorboard_root.iterdir()):
            shutil.rmtree(tensorboard_root)

    if usual_root.exists() and not any(usual_root.iterdir()):
        shutil.rmtree(usual_root)


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ FEATURE GROUPING UTILITIES                                            ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

def slugify(features: Iterable[str]) -> str:
    """Create a simple slug identifier from a list of feature names."""
    features = list(features)
    return "baseline" if not features else "_".join(features)


def feature_groups(additional_features: Iterable[str]) -> list[tuple[int, Tuple[str, ...]]]:
    """Return [(size, combination)] for all non-empty combinations plus baseline."""
    additional_features = list(additional_features)
    groups: list[tuple[int, Tuple[str, ...]]] = [(0, tuple())]
    for size in range(1, len(additional_features) + 1):
        for combo in itertools.combinations(additional_features, size):
            groups.append((size, combo))
    return groups


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ METADATA UTILITIES                                                    ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

def write_metadata(path: Path, *, task: str, column_features: list[str], baseline: list[str], extras: Iterable[str], context: list[str]) -> None:
    """Persist ablation metadata as JSON."""
    payload = {
        "task": task,
        "column_features": column_features,
        "baseline_context_features": baseline,
        "additional_features": list(extras),
        "context_features": context,
    }
    path.write_text(json.dumps(payload, indent=2))


__all__ = [
    "resolve_baseline_features",
    "ensure_clean_dir",
    "reorganize_outputs",
    "slugify",
    "feature_groups",
    "write_metadata",
]
