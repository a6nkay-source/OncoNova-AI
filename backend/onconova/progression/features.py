"""Features for the progression model, derived from longitudinal analysis output.

Imaging features are required. Without at least two scans there is no growth
information, and the model returns an explicit unavailable state rather than
guessing. Structured clinical fields are optional and imputed when missing, and
every imputation is reported back to the user.
"""

from __future__ import annotations

from typing import Any

import numpy as np

IMAGING_FEATURES = [
    "baseline_total_volume_cc",
    "log_volume_ratio_per_100d",
    "new_lesion_count",
    "node_region_lesion_observed",
    "primary_sphericity",
]
CLINICAL_FEATURES = ["age_years"]
FEATURES = IMAGING_FEATURES + CLINICAL_FEATURES
REQUIRED = set(IMAGING_FEATURES)


def from_longitudinal(result: dict[str, Any], clinical: dict[str, Any] | None = None) -> dict[str, Any]:
    tps = result["timepoints"]
    base, last = tps[0], tps[-1]
    interval = max(int(last["days_from_baseline"]) - int(base["days_from_baseline"]), 1)
    v0, v1 = float(base["total_lesion_volume_cc"]), float(last["total_lesion_volume_cc"])
    log_ratio = np.log(max(v1, 1e-3) / max(v0, 1e-3)) / (interval / 100.0) if v0 > 0 else None

    primary = next((t for t in result["tracks"] if t["class"] == "primary_tumor" and base["tp_id"] in t["measurements"]), None)
    sph = primary["measurements"][base["tp_id"]].get("sphericity") if primary else None
    node = any(t["class"] == "lymph_node_region_lesion" for t in result["tracks"])
    clinical = clinical or {}
    return {
        "baseline_total_volume_cc": v0 if v0 > 0 else None,
        "log_volume_ratio_per_100d": None if log_ratio is None else float(log_ratio),
        "new_lesion_count": int(result.get("new_lesion_count", 0)),
        "node_region_lesion_observed": int(node),
        "primary_sphericity": sph,
        "age_years": clinical.get("age_years"),
        "interval_days": interval,
    }


def missing_required(features: dict[str, Any]) -> list[str]:
    return [k for k in IMAGING_FEATURES if features.get(k) is None]


def to_row(features: dict[str, Any]) -> np.ndarray:
    return np.array([[np.nan if features.get(k) is None else float(features[k]) for k in FEATURES]], dtype=float)
