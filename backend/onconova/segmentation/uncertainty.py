"""Voxel-wise uncertainty summaries and unreliable-region detection."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage

from onconova.imaging.volume import Volume


def binary_entropy(p: np.ndarray) -> np.ndarray:
    """Entropy of a Bernoulli probability, scaled to [0, 1]."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return (-(p * np.log2(p) + (1 - p) * np.log2(1 - p))).astype(np.float32)


def categorical_entropy(probs: np.ndarray) -> np.ndarray:
    """Predictive entropy over the class axis (axis 0), scaled to [0, 1]."""
    p = np.clip(probs, 1e-6, 1.0)
    return (-(p * np.log(p)).sum(0) / np.log(probs.shape[0])).astype(np.float32)


def component_confidence(foreground_prob: np.ndarray, labels: np.ndarray) -> list[dict[str, Any]]:
    comps, n = ndimage.label(labels > 0)
    out = []
    for i in range(1, n + 1):
        region = comps == i
        out.append({"component": i, "mean_foreground_probability": round(float(foreground_prob[region].mean()), 4)})
    return out


def unreliable_regions(uncertainty: np.ndarray, volume: Volume, labels: np.ndarray | None = None, *,
                       threshold: float = 0.6, min_volume_cc: float = 0.02, min_remote_cc: float = 1.0,
                       limit: int = 20) -> list[dict[str, Any]]:
    """Connected uncertain regions, largest first.

    With ``labels``, regions touching a predicted lesion (within 3 voxels) are always
    reported; regions elsewhere only if they exceed ``min_remote_cc``, so diffuse
    boundary noise across normal anatomy does not swamp the list.
    """
    high = uncertainty >= threshold
    near = ndimage.binary_dilation(labels > 0, iterations=3) if labels is not None and labels.any() else None
    comps, n = ndimage.label(high, structure=ndimage.generate_binary_structure(3, 3))
    voxel_cc = volume.voxel_volume_mm3 / 1000.0
    regions = []
    for i, sl in enumerate(ndimage.find_objects(comps), start=1):
        if sl is None:
            continue
        region = comps[sl] == i
        vol = float(region.sum()) * voxel_cc
        touches = bool(near is not None and (near[sl] & region).any())
        if vol < min_volume_cc or (labels is not None and not touches and vol < min_remote_cc):
            continue
        idx = np.argwhere(region).mean(0) + np.array([s.start for s in sl])
        regions.append({
            "volume_cc": round(vol, 3),
            "mean_uncertainty": round(float(uncertainty[sl][region].mean()), 3),
            "location": "at or near a predicted lesion" if touches else "away from predicted lesions",
            "centroid_index_zyx": [round(float(v), 1) for v in idx],
            "centroid_mm": [round(float(v), 1) for v in volume.index_to_physical(idx)],
        })
    regions.sort(key=lambda r: r["volume_cc"], reverse=True)
    return regions[:limit]


def summarize(uncertainty: np.ndarray, labels: np.ndarray, threshold: float = 0.6) -> dict[str, Any]:
    pred = labels > 0
    return {
        "mean_uncertainty_in_prediction": round(float(uncertainty[pred].mean()), 4) if pred.any() else None,
        "high_uncertainty_fraction_of_prediction": round(float((uncertainty[pred] >= threshold).mean()), 4) if pred.any() else None,
        "high_uncertainty_volume_voxels": int((uncertainty >= threshold).sum()),
        "threshold": threshold,
    }
