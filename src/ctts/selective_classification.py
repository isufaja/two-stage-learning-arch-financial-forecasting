import json
import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt
from typing import Any, Dict, Iterable, Optional
from scipy.interpolate import PchipInterpolator, UnivariateSpline

def collect_risk_coverage_curve(y_true,
                                y_score,
                                thresholds = None,
                                empty_selection_value = np.nan,
                                include_error_counts: bool = False):

    """Compute risk-coverage curve: coverage = fraction selected,
       risk = error rate among selected samples.
       NOTE: In M2 for meta-labeling, the coverage of M2 are the selected samples over M1's 
       positive predictions. Because M2, has a chance to execute only on M1's positive predictions.
       """
    
    # ┏━━━━━━━━━━ Convert inputs to NumPy arrays ━━━━━━━━━━┓
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    # ┏━━━━━━━━━━ Validate input lengths and non-emptiness ━━━━━━━━━━┓
    if y_true.shape[0] != y_score.shape[0]:
        raise ValueError("y_true and y_score must have the same length")
    total = y_score.shape[0]
    if total == 0:
        raise ValueError("Inputs must contain at least one sample")
    
    # ┏━━━━━━━━━━ Convert thresholds to array for iteration ━━━━━━━━━━┓
    if thresholds is None:
        finite_scores = y_score[np.isfinite(y_score)]
        if finite_scores.size == 0:
            raise ValueError("y_score must contain at least one finite value")
        thresholds = np.unique(np.concatenate(([0.0], finite_scores, [1.0])))
    else:
        thresholds = np.asarray(thresholds, dtype=float)
    if thresholds.ndim != 1 or thresholds.size == 0:
        raise ValueError("thresholds must be a non-empty one-dimensional array")

    # ┏━━━━━━━━━━ Preallocate arrays for results ━━━━━━━━━━┓
    coverages       = np.empty(thresholds.shape, dtype=float)
    risks           = np.empty(thresholds.shape, dtype=float)
    selected_counts = np.empty(thresholds.shape, dtype=int)
    error_counts    = np.empty(thresholds.shape, dtype=int) if include_error_counts else None

    # ┏━━━━━━━━━━ Compute coverage (fraction selected) and risk (error rate) for each threshold ━━━━━━━━━━┓
    for idx, threshold in enumerate(thresholds):
        selected = y_score >= threshold
        selected_count = int(np.sum(selected))

        coverages[idx] = selected_count / total
        selected_counts[idx] = selected_count

        if selected_count == 0: # No samples selected
            risks[idx] = empty_selection_value
            if include_error_counts:
                error_counts[idx] = 0
        else:
            errors = (y_true[selected] == 0) # Count wrong predictions among selected samples
            err_count = int(np.sum(errors))
            risks[idx] = err_count / selected_count
            if include_error_counts:
                error_counts[idx] = err_count
        
    # ┏━━━━━━━━━━ Store computed arrays in a dictionary ━━━━━━━━━━┓
    curve = {"thresholds": thresholds,
             "coverage": coverages,
             "risk": risks,
             "selected_count": selected_counts}
    
    # ┏━━━━━━━━━━ Curve's Additional Data ━━━━━━━━━━┓
    if include_error_counts:
        curve["error_count"] = error_counts

    return curve


def coverage_at_risk(y_true = None,
                     y_score = None,
                     *,  
                     max_risk,
                     thresholds = None,
                     empty_selection_value = np.nan,
                     min_coverage: float = 0.0,
                     min_selected: int = 0,
                     curve: Optional[Dict[str, np.ndarray]] = None):

    """Return the best threshold whose risk is ≤ max_risk.
    The max_risk, is defined by the user, in the config.yaml file.

    Notes: 
    1. Coverage is the fraction of samples whose score ≥ threshold.
    2. Risk is the error rate among the selected samples when predicting 1 for score ≥ threshold.    
    """
    
    # ┏━━━━━━━━━━ Check that the allowed risk is a valid probability ━━━━━━━━━━┓
    if not 0.0 <= max_risk <= 1.0:
        raise ValueError("max_risk must be within [0, 1]")

    # ┏━━━━━━━━━━ Recompute Curve if not Provided; build it from y_true / y_score ━━━━━━━━━━┓
    if curve is None:
        if y_true is None or y_score is None:
            raise ValueError("Provide either `curve` or y_true/y_score arrays")
        curve = collect_risk_coverage_curve(y_true=y_true,
                                            y_score=y_score,
                                            thresholds=thresholds,
                                            empty_selection_value=empty_selection_value)
        
    # ┏━━━━━━━━━━ If a curve is supplied, thresholds must come from that curve ━━━━━━━━━━┓
    elif thresholds is not None:
        raise ValueError("When `curve` is supplied, `thresholds` must be omitted")

    # ┏━━━━━━━━━━ Extract Information Curve ━━━━━━━━━━┓
    risks = curve["risk"]
    coverages = curve["coverage"]
    selected_counts = curve["selected_count"]
    thresholds_arr = curve["thresholds"]

    # ┏━━━━━━━━━━ Mask thresholds that satisfy [Minimum Selected & Finite Risk & Min Coverage & Max Risk] ━━━━━━━━━━┓
    valid_mask = (~np.isnan(risks) 
                & (risks <= max_risk)
                & (coverages >= min_coverage) 
                & (selected_counts >= min_selected))

    # ┏━━━━━━━━━━ All Conditions Satisfied, choose the one with maximum coverage ━━━━━━━━━━┓
    if np.any(valid_mask):
        best_idx = int(np.argmax(coverages[valid_mask])) # Index of the highest coverage among valid entries
        valid_indices = np.flatnonzero(valid_mask)       # Map back to original indices
        chosen_idx = valid_indices[best_idx]             # Final chosen index in original arrays

        return {"threshold":            float(thresholds_arr[chosen_idx]),
                "coverage":             float(coverages[chosen_idx]),
                "risk":                 float(risks[chosen_idx]),
                "selected_count":       int(selected_counts[chosen_idx]),
                "constraint_satisfied": True}

    # ┏━━━━━━━━━━ Minimum Conditions Satisfied [Only Minimum Selected & Finite Risk] ━━━━━━━━━━┓

    # If no thresholds satisfy all the risk and coverage constraints,
    # then pick the threshold with the smallest risk among all thresholds 
    # that have at least one selected sample and a finite risk value.
    fallback_mask = (~np.isnan(risks)) & (selected_counts >= max(min_selected, 1))
    if np.any(fallback_mask):
        fallback_indices = np.flatnonzero(fallback_mask)
        min_risk_idx = fallback_indices[int(np.argmin(risks[fallback_mask]))]
        
        return {"threshold":            float(thresholds_arr[min_risk_idx]),
                "coverage":             float(coverages[min_risk_idx]),
                "risk":                 float(risks[min_risk_idx]),
                "selected_count":       int(selected_counts[min_risk_idx]),
                "constraint_satisfied": False}

    # ┏━━━━━━━━━━ No Condition Satisfied; return a degenerate solution ━━━━━━━━━━┓
    return {"threshold": float(thresholds_arr.max()) if thresholds_arr.size else np.nan,
            "coverage": 0.0,
            "risk": np.nan,
            "selected_count": 0,
            "constraint_satisfied": False}


def area_under_risk_coverage(y_true = None,
                             y_score = None,
                             thresholds = None,
                             empty_selection_value = np.nan,
                             curve: Optional[Dict[str, np.ndarray]] = None):

    """Compute Area Under the Risk-Coverage Curve (AURC).

    AURC = ∫ risk(coverage) d(coverage), with coverage on the x-axis
    and risk (error rate among selected samples) on the y-axis.
    """
    
    # ┏━━━━━━━━━━ Recompute Curve if not Provided; build it from y_true / y_score ━━━━━━━━━━┓
    if curve is None:
        if y_true is None or y_score is None:
            raise ValueError("Provide either `curve` or y_true/y_score arrays")
        
        curve = collect_risk_coverage_curve(y_true = y_true,
                                            y_score = y_score,
                                            thresholds = thresholds,
                                            empty_selection_value = empty_selection_value)

    # ┏━━━━━━━━━━ Extract coverage and risk arrays from the curve ━━━━━━━━━━┓
    coverage = np.asarray(curve["coverage"])
    risk = np.asarray(curve["risk"])

    # ┏━━━━━━━━━━ Keep only points where both coverage and risk are defined ━━━━━━━━━━┓
    valid_mask = (~np.isnan(coverage)) & (~np.isnan(risk))
    coverage = coverage[valid_mask]
    risk = risk[valid_mask]

    # ┏━━━━━━━━━━ Minimum Requirement for AUR&C Curve ━━━━━━━━━━┓
    
    # For a correct trapezoidal integration (np.trapezoid(y, x)), the x values should be monotonic; 
    # Sorting ensures coverage is nondecreasing.
    if coverage.size < 2:
        return 0.0

    # ┏━━━━━━━━━━ Sorting Coverages & Risk for computing AUR&C Curve ━━━━━━━━━━┓
    order = np.argsort(coverage)
    coverage_sorted = coverage[order]
    risk_sorted = risk[order]

    # ┏━━━━━━━━━━ Area of Risk & Coverage Curve ━━━━━━━━━━┓
    area = float(np.trapezoid(risk_sorted, coverage_sorted))
    return area


def plot_coverage_risk_curve(y_true = None,
                             y_score = None,
                             *,
                             curve: Optional[Dict[str, np.ndarray]] = None,
                             thresholds = None,
                             empty_selection_value = 0.0,
                             label: Optional[str] = None,
                             ax: Optional[plt.Axes] = None,
                             save_path: Optional[str] = None,
                             show: bool = True,
                             highlight_point: Optional[tuple] = None,
                             highlight_text: Optional[str] = None,
                             asset_name: Optional[str] = None,
                             smooth: bool = False,
                             smooth_points: int = 200,
                             smooth_method: str = "spline",
                             smooth_s_factor: Optional[float] = None,
                             coverage_min: Optional[float] = None,
                             coverage_max: Optional[float] = None):

    """
    Plot a risk-coverage curve (risk on y-axis, coverage on x-axis). 
    """

    # ┏━━━━━━━━━━ Recompute Curve if not Provided; build it from y_true / y_score ━━━━━━━━━━┓
    if curve is None:
        if y_true is None or y_score is None:
            raise ValueError("Provide either `curve` or y_true/y_score arrays")

        curve = collect_risk_coverage_curve(
            y_true=y_true,
            y_score=y_score,
            thresholds=thresholds,
            empty_selection_value=empty_selection_value,
        )

    # ┏━━━━━━━━━━ Extract Coverage & Risk from Curve ━━━━━━━━━━┓
    coverage = np.asarray(curve["coverage"], dtype=float)
    risk = np.asarray(curve["risk"], dtype=float)

    # ┏━━━━━━━━━━ Sort points by coverage so the curve is monotone in x ━━━━━━━━━━┓
    order = np.argsort(coverage)
    coverage_sorted = coverage[order]
    risk_sorted = risk[order]

    # ┏━━━━━━━━━━ Optional smoothing using PCHIP or spline ━━━━━━━━━━┓
    coverage_plot = coverage_sorted
    risk_plot = risk_sorted
    if smooth and coverage_sorted.size >= 2:
        unique_mask = np.concatenate(([True], np.diff(coverage_sorted) != 0))
        cov_unique = coverage_sorted[unique_mask]
        risk_unique = risk_sorted[unique_mask]
        finite_mask = np.isfinite(risk_unique)
        cov_unique = cov_unique[finite_mask]
        risk_unique = risk_unique[finite_mask]
        if cov_unique.size >= 2:
            grid = np.linspace(cov_unique.min(), cov_unique.max(), smooth_points)
            method = (smooth_method or "pchip").lower()
            if method == "spline":
                s_factor = smooth_s_factor
                if s_factor is None:
                    s_factor = max(1.0, 0.001 * cov_unique.size)
                try:
                    interpolator = UnivariateSpline(cov_unique, risk_unique, s=s_factor)
                    risk_interp = interpolator(grid)
                except Exception:
                    interpolator = PchipInterpolator(cov_unique, risk_unique, extrapolate=False)
                    risk_interp = interpolator(grid)
            else:
                interpolator = PchipInterpolator(cov_unique, risk_unique, extrapolate=False)
                risk_interp = interpolator(grid)
            valid = np.isfinite(risk_interp)
            coverage_plot = grid[valid]
            risk_plot = risk_interp[valid]

    if coverage_min is not None or coverage_max is not None:
        clip_mask = np.ones_like(coverage_plot, dtype=bool)
        if coverage_min is not None:
            clip_mask &= coverage_plot >= coverage_min
        if coverage_max is not None:
            clip_mask &= coverage_plot <= coverage_max
        coverage_plot = coverage_plot[clip_mask]
        risk_plot = risk_plot[clip_mask]


    # ┏━━━━━━━━━━ Optional smoothing using PCHIP or spline ━━━━━━━━━━┓
    coverage_plot = coverage_sorted
    risk_plot = risk_sorted
    if smooth and coverage_sorted.size >= 2:
        unique_mask = np.concatenate(([True], np.diff(coverage_sorted) != 0))
        cov_unique = coverage_sorted[unique_mask]
        risk_unique = risk_sorted[unique_mask]
        finite_mask = np.isfinite(risk_unique)
        cov_unique = cov_unique[finite_mask]
        risk_unique = risk_unique[finite_mask]
        if cov_unique.size >= 2:
            grid = np.linspace(cov_unique.min(), cov_unique.max(), smooth_points)
            method = (smooth_method or "pchip").lower()
            if method == "spline":
                s_factor = smooth_s_factor
                if s_factor is None:
                    s_factor = max(1.0, 0.001 * cov_unique.size)
                try:
                    interpolator = UnivariateSpline(cov_unique, risk_unique, s=s_factor)
                    risk_interp = interpolator(grid)
                except Exception:
                    interpolator = PchipInterpolator(cov_unique, risk_unique, extrapolate=False)
                    risk_interp = interpolator(grid)
            else:
                interpolator = PchipInterpolator(cov_unique, risk_unique, extrapolate=False)
                risk_interp = interpolator(grid)
            valid = np.isfinite(risk_interp)
            coverage_plot = grid[valid]
            risk_plot = risk_interp[valid]

    if coverage_min is not None or coverage_max is not None:
        clip_mask = np.ones_like(coverage_plot, dtype=bool)
        if coverage_min is not None:
            clip_mask &= coverage_plot >= coverage_min
        if coverage_max is not None:
            clip_mask &= coverage_plot <= coverage_max
        coverage_plot = coverage_plot[clip_mask]
        risk_plot = risk_plot[clip_mask]


    # ┏━━━━━━━━━━ Optional smoothing using PCHIP or spline ━━━━━━━━━━┓
    coverage_plot = coverage_sorted
    risk_plot = risk_sorted
    if smooth and coverage_sorted.size >= 2:
        unique_mask = np.concatenate(([True], np.diff(coverage_sorted) != 0))
        cov_unique = coverage_sorted[unique_mask]
        risk_unique = risk_sorted[unique_mask]
        finite_mask = np.isfinite(risk_unique)
        cov_unique = cov_unique[finite_mask]
        risk_unique = risk_unique[finite_mask]
        if cov_unique.size >= 2:
            grid = np.linspace(cov_unique.min(), cov_unique.max(), smooth_points)
            method = (smooth_method or "pchip").lower()
            if method == "spline":
                s_factor = smooth_s_factor
                if s_factor is None:
                    s_factor = max(1.0, 0.001 * cov_unique.size)
                try:
                    interpolator = UnivariateSpline(cov_unique, risk_unique, s=s_factor)
                    risk_interp = interpolator(grid)
                except Exception:
                    interpolator = PchipInterpolator(cov_unique, risk_unique, extrapolate=False)
                    risk_interp = interpolator(grid)
            else:
                interpolator = PchipInterpolator(cov_unique, risk_unique, extrapolate=False)
                risk_interp = interpolator(grid)
            valid = np.isfinite(risk_interp)
            coverage_plot = grid[valid]
            risk_plot = risk_interp[valid]

    if coverage_min is not None or coverage_max is not None:
        clip_mask = np.ones_like(coverage_plot, dtype=bool)
        if coverage_min is not None:
            clip_mask &= coverage_plot >= coverage_min
        if coverage_max is not None:
            clip_mask &= coverage_plot <= coverage_max
        coverage_plot = coverage_plot[clip_mask]
        risk_plot = risk_plot[clip_mask]

    # ┏━━━━━━━━━━ Create Skeleton Plot ━━━━━━━━━━┓
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        created_fig = True
    else:
        fig = ax.figure

    # ┏━━━━━━━━━━ Legend of Plot ━━━━━━━━━━┓
    ax.plot(coverage_plot, risk_plot, label=label)
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Risk (Error Rate)")
    title = "Risk-Coverage Curve"
    if asset_name:
        title = f"{title} ({asset_name})"
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)

    # ┏━━━━━━━━━━ Optional highlight of a selected operating point (coverage, risk) ━━━━━━━━━━┓
    if highlight_point is not None:
        try:
            hx, hy = float(highlight_point[0]), float(highlight_point[1])
            if not (np.isnan(hx) or np.isnan(hy)):
                ax.scatter([hx], [hy], color="red", marker="x", s=80, zorder=5)
                if highlight_text:
                    ax.annotate(highlight_text,
                                xy=(hx, hy), xytext=(6, -6), textcoords="offset points",
                                fontsize=9, color="red")
        except Exception:
            pass
    
    # ┏━━━━━━━━━━ Optional Shortening Interval ━━━━━━━━━━┓
    if coverage_min is not None or coverage_max is not None:
        ax.set_xlim(left=coverage_min, right=coverage_max)
        
    handles, labels = ax.get_legend_handles_labels()
    filtered = [(h, l) for h, l in zip(handles, labels) if l]
    if filtered:
        ax.legend(*zip(*filtered))

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path)
    if created_fig:
        if show:
            plt.show()
        else:
            plt.close(fig)


def _to_jsonable(value: Any) -> Any:
    """
    It converts NumPy objects and nested structures (arrays, scalars, dicts, lists) 
    into standard Python types (list, float, int, etc.) that the JSON encoder can understand
    """

    
    # ┏━━━━━━━━━━ # Convert NumPy arrays to lists ━━━━━━━━━━┓
    if isinstance(value, np.ndarray):
        return value.tolist()
    # ┏━━━━━━━━━━ # Convert NumPy scalar types to native Python types ━━━━━━━━━━┓
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    # ┏━━━━━━━━━━ # Recursively convert dictionaries, lists, and tuples ━━━━━━━━━━┓
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    # ┏━━━━━━━━━━ # Convert NumPy scalar types to native Python types ━━━━━━━━━━┓
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    # ┏━━━━━━━━━━ # Base case: leave native JSON-safe types unchanged ━━━━━━━━━━┓
    return value


def save_metrics(metrics: Dict[str, Any], save_path: str, convert_numpy: bool = True) -> None:
    """
    Save a dictionary of metrics to a JSON file.

    If `convert_numpy=True`, NumPy objects (arrays, scalars) are converted
    into standard Python types so they can be safely serialized.
    """
    
    # ┏━━━━━━━━━━ Convert NumPy types to JSON-compatible objects if requested ━━━━━━━━━━┓
    if convert_numpy:
        payload = _to_jsonable(metrics)
    else:
        payload = metrics
    # ┏━━━━━━━━━━ Write metrics to JSON file with human-readable formatting ━━━━━━━━━━┓
    with open(save_path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=4)
