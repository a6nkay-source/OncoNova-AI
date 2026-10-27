"""Modality-aware intensity normalisation.

CT is in Hounsfield units, so fixed windows are meaningful. MRI intensities have
no absolute scale, so they are standardised within the body foreground. PET is
scaled by percentile because SUV conversion needs decay and dose metadata that
this prototype does not yet handle.
"""

from __future__ import annotations

import numpy as np

from onconova.errors import ModelInputError
from onconova.imaging.volume import Volume

# (level, width) in HU
CT_WINDOWS = {
    "lung": (-600.0, 1500.0),
    "mediastinum": (50.0, 350.0),
    "soft_tissue": (40.0, 400.0),
    "bone": (400.0, 2000.0),
}
MODEL_CT_RANGE = (-1000.0, 400.0)


def window(array: np.ndarray, level: float, width: float) -> np.ndarray:
    lo, hi = level - width / 2, level + width / 2
    return ((np.clip(array, lo, hi) - lo) / (hi - lo)).astype(np.float32)


def foreground_mask(array: np.ndarray) -> np.ndarray:
    threshold = np.percentile(array, 20)
    mask = array > threshold
    return mask if mask.any() else np.ones_like(array, dtype=bool)


def normalize_for_model(volume: Volume) -> tuple[np.ndarray, str]:
    """Return a float32 array scaled for model input and a description of the method."""
    arr = volume.array.astype(np.float32, copy=False)
    if volume.modality == "CT":
        lo, hi = MODEL_CT_RANGE
        return ((np.clip(arr, lo, hi) - lo) / (hi - lo)).astype(np.float32), f"CT clipped to [{lo:.0f}, {hi:.0f}] HU, scaled to [0, 1]"
    if volume.modality == "MR":
        fg = foreground_mask(arr)
        lo, hi = np.percentile(arr[fg], [0.5, 99.5])
        clipped = np.clip(arr, lo, hi)
        mean, std = float(clipped[fg].mean()), float(clipped[fg].std()) or 1.0
        return ((clipped - mean) / std).astype(np.float32), "MRI clipped to foreground 0.5-99.5th percentile, z-scored"
    if volume.modality == "PT":
        lo, hi = np.percentile(arr, [1, 99.9])
        return ((np.clip(arr, lo, hi) - lo) / max(hi - lo, 1e-6)).astype(np.float32), "PET percentile-scaled (relative uptake, not SUV)"
    raise ModelInputError("unknown_modality", "Set the modality (CT, MR or PT) before normalising this volume.")


def intensity_profile(volume: Volume) -> dict[str, float]:
    """Summary statistics used for out-of-distribution checks."""
    arr = volume.array
    fg = foreground_mask(arr)
    p1, p50, p99 = np.percentile(arr[fg], [1, 50, 99])
    return {"p1": float(p1), "p50": float(p50), "p99": float(p99), "min": float(arr.min()), "max": float(arr.max())}
