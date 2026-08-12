"""Portable filesystem helpers for CTTS datasets and outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union


PathLike = Union[str, Path]
PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def resolve_project_path(path: PathLike, *, base_dir: Optional[PathLike] = None) -> Path:
    """Resolve a user path, treating relative paths as project-root relative."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    base = Path(base_dir).expanduser() if base_dir is not None else PROJECT_ROOT
    return (base / candidate).resolve()


def dataset_path(
    provider: str,
    market: str,
    symbol: str,
    split: str = "merge",
    granularity: Optional[str] = None,
    meta_label_mode: Optional[str] = None,
    *,
    data_root: Optional[PathLike] = None,
    must_exist: bool = True,
) -> Path:
    """Build the path to one M1-derived CTTS CSV.

    The expected layout is::

        data/<provider>/<market>/<symbol>/<granularity>_<mode>/<symbol>_<split>.csv

    Relative ``data_root`` values are resolved from the repository root.
    """
    root = resolve_project_path(data_root or DATA_DIR)
    root = root / provider / market / symbol
    if granularity:
        root = root / granularity
    if meta_label_mode:
        mode = meta_label_mode.lower()
        suffix = "og" if mode == "original" else mode
        root = root.with_name(f"{root.name}_{suffix}")

    path = root / f"{symbol}_{split}.csv"
    if must_exist and not path.is_file():
        raise FileNotFoundError(
            f"CTTS dataset file not found: {path}. "
            "Set paths.data_root in the YAML configuration or create the documented data layout."
        )
    return path


def bolt_path(*args, **kwargs) -> Path:
    return dataset_path("Bolt", *args, **kwargs)


def chronos_path(*args, **kwargs) -> Path:
    return dataset_path("Chronos", *args, **kwargs)


def tirex_path(*args, **kwargs) -> Path:
    return dataset_path("Tirex", *args, **kwargs)

