"""Cumulative dose-volume histograms and summary statistics.

The arithmetic is standard and applies to any dose grid on the image grid. The
meaning of the result depends entirely on where the dose came from: for the
synthetic illustration the values are relative and non-clinical.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def cumulative_dvh(dose: np.ndarray, mask: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """Fraction of the structure receiving at least each dose level in ``bins``."""
    values = np.sort(dose[mask].ravel())
    if values.size == 0:
        return np.zeros_like(bins, dtype=float)
    return 1.0 - np.searchsorted(values, bins, side="left") / values.size


def dose_at_volume(dose: np.ndarray, mask: np.ndarray, volume_fraction: float) -> float:
    """D_x: the minimum dose received by the hottest ``volume_fraction`` of the structure."""
    return float(np.quantile(dose[mask], 1.0 - volume_fraction))


def volume_at_dose(dose: np.ndarray, mask: np.ndarray, level: float) -> float:
    return float(np.mean(dose[mask] >= level))


def structure_stats(dose: np.ndarray, mask: np.ndarray, voxel_cc: float, *, is_target: bool) -> dict[str, Any]:
    d = dose[mask]
    stats = {
        "volume_cc": round(float(mask.sum()) * voxel_cc, 3),
        "mean": round(float(d.mean()), 4),
        "max_D2": round(dose_at_volume(dose, mask, 0.02), 4),
        "D50": round(dose_at_volume(dose, mask, 0.50), 4),
        "V5": round(volume_at_dose(dose, mask, 0.05), 4),
        "V20": round(volume_at_dose(dose, mask, 0.20), 4),
        "V50": round(volume_at_dose(dose, mask, 0.50), 4),
    }
    if is_target:
        stats.update({
            "D95": round(dose_at_volume(dose, mask, 0.95), 4),
            "D98": round(dose_at_volume(dose, mask, 0.98), 4),
            "V95": round(volume_at_dose(dose, mask, 0.95), 4),
            "homogeneity_index": round((dose_at_volume(dose, mask, 0.02) - dose_at_volume(dose, mask, 0.98))
                                       / max(dose_at_volume(dose, mask, 0.50), 1e-6), 4),
        })
    return stats


def paddick_conformity(dose: np.ndarray, target: np.ndarray, level: float = 0.95) -> float:
    piv = dose >= level
    tv_piv = float(np.sum(piv & target))
    denom = float(target.sum()) * float(piv.sum())
    return 0.0 if denom == 0 else tv_piv ** 2 / denom


def dvh_report(dose: np.ndarray, structures: dict[str, np.ndarray], target_name: str, voxel_cc: float,
               *, max_level: float | None = None, n_bins: int = 241) -> dict[str, Any]:
    top = max_level or float(max(1.2, np.quantile(dose[dose > 0], 0.999) if (dose > 0).any() else 1.2))
    bins = np.linspace(0, top, n_bins)
    curves, stats, skipped = {}, {}, []
    for name, mask in structures.items():
        if not mask.any():
            skipped.append(name)
            continue
        curves[name] = np.round(cumulative_dvh(dose, mask, bins), 5).tolist()
        stats[name] = structure_stats(dose, mask, voxel_cc, is_target=(name == target_name))
    out = {"dose_bins": np.round(bins, 4).tolist(), "curves": curves, "stats": stats, "skipped_empty": skipped}
    if target_name in structures and structures[target_name].any():
        out["paddick_conformity_index_95"] = round(paddick_conformity(dose, structures[target_name]), 4)
    return out
