"""Shared types for segmentation models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

import numpy as np

from onconova.imaging.normalize import intensity_profile
from onconova.imaging.volume import Volume
from onconova.safety import DEMO_MODEL_LABEL

LESION_CLASS_NAMES = {1: "primary_tumor", 2: "lymph_node_region_lesion", 3: "other_lesion"}


@dataclass
class ModelCard:
    id: str
    name: str
    version: str
    kind: str
    modalities: list[str]
    classes: dict[int, str]
    trained_on: str
    intended_use: str
    limitations: list[str]
    label: str = DEMO_MODEL_LABEL
    validated_for_clinical_use: bool = False
    status: str = "available"
    unavailable_reason: str | None = None
    uncertainty_method: str = ""
    checkpoint_sha256: str | None = None
    training_manifest: str | None = None
    # Ranges seen in the tuning/training data, used for out-of-distribution warnings.
    input_profile: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["classes"] = {str(k): v for k, v in self.classes.items()}
        return out


@dataclass
class SegmentationResult:
    labels: np.ndarray            # uint8 class map, same shape as the image
    foreground_probability: np.ndarray  # float32 in [0, 1]
    uncertainty: np.ndarray       # float32 in [0, 1]; 1 = maximally uncertain
    assessed_region: np.ndarray   # bool; where the model actually looked
    warnings: list[str] = field(default_factory=list)
    saliency: np.ndarray | None = None


class Segmenter(Protocol):
    card: ModelCard

    def predict(self, volume: Volume, **kwargs: Any) -> SegmentationResult: ...


def ood_warnings(volume: Volume, profile: dict[str, Any]) -> list[str]:
    """Compare a scan with the model's input profile; warn about anything outside it."""
    if not profile:
        return ["No input profile is recorded for this model; distribution shift cannot be assessed."]
    warnings = []
    if profile.get("modality") and volume.modality != profile["modality"]:
        warnings.append(f"Modality {volume.modality} differs from the model's training modality {profile['modality']}.")
    lo, hi = profile.get("spacing_min"), profile.get("spacing_max")
    if lo and hi:
        s = np.asarray(volume.spacing)
        if np.any(s < np.asarray(lo) * 0.8) or np.any(s > np.asarray(hi) * 1.25):
            warnings.append(f"Voxel spacing {np.round(s, 2).tolist()} mm is outside the training range {lo}-{hi} mm.")
    stats = intensity_profile(volume)
    for key in ("p1", "p50", "p99"):
        rng = profile.get(f"{key}_range")
        if rng and not (rng[0] - 0.1 * abs(rng[0]) - 25 <= stats[key] <= rng[1] + 0.1 * abs(rng[1]) + 25):
            warnings.append(f"Intensity {key} = {stats[key]:.0f} is outside the training range {rng}; treat output as out-of-distribution.")
    if profile.get("source") == "synthetic" and not volume.metadata.get("demo"):
        warnings.append("This model has only seen synthetic phantoms. Output on real scans is unvalidated and likely unreliable.")
    return warnings
