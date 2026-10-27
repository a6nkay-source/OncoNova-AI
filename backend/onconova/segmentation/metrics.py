"""Segmentation agreement metrics computed in physical units.

Conventions for empty masks: when both masks are empty, overlap metrics are 1.0
and distances are 0. When exactly one is empty, overlap metrics are 0.0 and the
surface distances are undefined (None), because there is no surface to measure.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure


def _counts(pred: np.ndarray, ref: np.ndarray, roi: np.ndarray | None = None) -> tuple[int, int, int, int]:
    p, r = pred.astype(bool), ref.astype(bool)
    if roi is not None:
        p, r = p & roi, r & roi
        total = int(roi.sum())
    else:
        total = p.size
    tp = int(np.sum(p & r))
    fp = int(np.sum(p & ~r))
    fn = int(np.sum(~p & r))
    return tp, fp, fn, total - tp - fp - fn


def dice(pred: np.ndarray, ref: np.ndarray) -> float:
    tp, fp, fn, _ = _counts(pred, ref)
    denom = 2 * tp + fp + fn
    return 1.0 if denom == 0 else 2 * tp / denom


def iou(pred: np.ndarray, ref: np.ndarray) -> float:
    tp, fp, fn, _ = _counts(pred, ref)
    denom = tp + fp + fn
    return 1.0 if denom == 0 else tp / denom


def sensitivity(pred: np.ndarray, ref: np.ndarray) -> float | None:
    tp, _, fn, _ = _counts(pred, ref)
    return None if tp + fn == 0 else tp / (tp + fn)


def specificity(pred: np.ndarray, ref: np.ndarray, roi: np.ndarray | None = None) -> float | None:
    """Voxel specificity. Pass an ROI (e.g. the body) or the value is dominated by air."""
    _, fp, _, tn = _counts(pred, ref, roi)
    return None if tn + fp == 0 else tn / (tn + fp)


def _surface(mask: np.ndarray) -> np.ndarray:
    return mask & ~binary_erosion(mask, structure=generate_binary_structure(3, 1), border_value=0)


def surface_distances(pred: np.ndarray, ref: np.ndarray, spacing_zyx: tuple[float, float, float]
                      ) -> tuple[np.ndarray, np.ndarray] | None:
    p, r = pred.astype(bool), ref.astype(bool)
    if not p.any() or not r.any():
        return None
    # Crop to the union bounding box: every surface voxel lies inside it, so distances are exact.
    coords = np.argwhere(p | r)
    lo = np.maximum(coords.min(0) - 2, 0)
    hi = coords.max(0) + 3
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    ps, rs = _surface(np.pad(p[sl], 1)), _surface(np.pad(r[sl], 1))
    dist_to_ref = distance_transform_edt(~rs, sampling=spacing_zyx)
    dist_to_pred = distance_transform_edt(~ps, sampling=spacing_zyx)
    return dist_to_ref[ps], dist_to_pred[rs]


def hausdorff(pred: np.ndarray, ref: np.ndarray, spacing_zyx: tuple[float, float, float],
              percentile: float = 100.0) -> float | None:
    if not pred.any() and not ref.any():
        return 0.0
    d = surface_distances(pred, ref, spacing_zyx)
    if d is None:
        return None
    return float(max(np.percentile(d[0], percentile), np.percentile(d[1], percentile)))


def mean_surface_distance(pred: np.ndarray, ref: np.ndarray, spacing_zyx: tuple[float, float, float]) -> float | None:
    if not pred.any() and not ref.any():
        return 0.0
    d = surface_distances(pred, ref, spacing_zyx)
    if d is None:
        return None
    return float((d[0].sum() + d[1].sum()) / (d[0].size + d[1].size))


def evaluate_pair(pred: np.ndarray, ref: np.ndarray, spacing_zyx: tuple[float, float, float],
                  roi: np.ndarray | None = None) -> dict[str, Any]:
    voxel_cc = float(np.prod(spacing_zyx)) / 1000.0
    return {
        "dice": dice(pred, ref),
        "iou": iou(pred, ref),
        "hausdorff_mm": hausdorff(pred, ref, spacing_zyx),
        "hd95_mm": hausdorff(pred, ref, spacing_zyx, 95),
        "mean_surface_distance_mm": mean_surface_distance(pred, ref, spacing_zyx),
        "sensitivity": sensitivity(pred, ref),
        "specificity": specificity(pred, ref, roi),
        "specificity_roi": "body" if roi is not None else "whole volume",
        "pred_volume_cc": float(pred.astype(bool).sum()) * voxel_cc,
        "ref_volume_cc": float(ref.astype(bool).sum()) * voxel_cc,
    }
