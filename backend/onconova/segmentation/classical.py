"""Classical lung-lesion heuristic: a transparent baseline that needs no training.

Pipeline: smooth, find air inside the body, fill holes slice by slice to recover
the lung envelope (lesions enclosed by lung become part of it), keep soft-tissue
density inside the envelope, reject tube-like components (vessels) by shape, and
call the largest remaining component the primary-lesion candidate.

Known blind spots, reported with every result: anything outside the lung
envelope (mediastinal or hilar nodes, chest wall), lesions touching the pleura or
mediastinum, and ground-glass opacities below the density threshold.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from onconova.errors import ModelInputError
from onconova.imaging.volume import Volume
from onconova.segmentation.base import LESION_CLASS_NAMES, ModelCard, SegmentationResult, ood_warnings
from onconova.segmentation.uncertainty import binary_entropy

PHANTOM_PROFILE = {
    "source": "synthetic",
    "modality": "CT",
    "spacing_min": [2.2, 2.2, 2.5],
    "spacing_max": [2.2, 2.2, 2.5],
    "p1_range": [-1000, -850],
    "p50_range": [-860, 60],
    "p99_range": [40, 700],
}


class ClassicalLungLesionSegmenter:
    def __init__(self, threshold_hu: float = -400.0, min_volume_cc: float = 0.03, max_elongation: float = 3.0,
                 min_lung_cc: float = 20.0) -> None:
        self.threshold_hu = threshold_hu
        self.min_volume_cc = min_volume_cc
        self.max_elongation = max_elongation
        self.min_lung_cc = min_lung_cc
        self.card = ModelCard(
            id="demo-classical",
            name="Classical lung-lesion heuristic",
            version="0.1.0",
            kind="classical-heuristic",
            modalities=["CT"],
            classes={1: LESION_CLASS_NAMES[1], 3: LESION_CLASS_NAMES[3]},
            trained_on="No training. Thresholds were chosen by hand on synthetic thoracic phantoms.",
            intended_use="Pipeline baseline and teaching example on synthetic CT.",
            limitations=[
                "Does not detect lesions outside the lung envelope (mediastinal/hilar nodes, chest wall).",
                "Lesions touching the pleura or mediastinum may be missed or truncated.",
                "Ground-glass or sub-solid lesions below the density threshold are missed.",
                "'Primary' means 'largest candidate'; it is a heuristic label, not a diagnosis.",
                "Probabilities are a sigmoid of HU distance to the threshold, not calibrated.",
            ],
            uncertainty_method="Binary entropy of a heuristic intensity score inside the lung envelope (not calibrated)",
            input_profile=PHANTOM_PROFILE,
        )

    def _lung_envelope(self, hu: np.ndarray, voxel_cc: float) -> np.ndarray:
        body = ndimage.binary_fill_holes(hu > -500)
        comps, n = ndimage.label(body)
        if n:
            sizes = ndimage.sum(body, comps, range(1, n + 1))
            body = comps == (int(np.argmax(sizes)) + 1)
        for z in range(body.shape[0]):
            body[z] = ndimage.binary_fill_holes(body[z])
        air = (hu < self.threshold_hu) & body
        envelope = np.zeros_like(air)
        for z in range(air.shape[0]):
            envelope[z] = ndimage.binary_fill_holes(air[z])
        comps, n = ndimage.label(envelope)
        if n:
            sizes = ndimage.sum(envelope, comps, range(1, n + 1)) * voxel_cc
            keep = np.isin(comps, np.nonzero(sizes >= self.min_lung_cc)[0] + 1)
            envelope = keep
        return envelope

    @staticmethod
    def _elongation(coords_mm: np.ndarray) -> float:
        if len(coords_mm) < 5:
            return 1.0
        eig = np.sort(np.linalg.eigvalsh(np.cov(coords_mm.T)))[::-1]
        return float(np.sqrt(eig[0] / max(eig[1], 1e-6)))

    def predict(self, volume: Volume, **_: object) -> SegmentationResult:
        if volume.modality != "CT":
            raise ModelInputError("unsupported_modality", "The classical heuristic needs CT in Hounsfield units.")
        voxel_cc = volume.voxel_volume_mm3 / 1000.0
        hu = ndimage.gaussian_filter(volume.array.astype(np.float32), 0.8)
        envelope = self._lung_envelope(hu, voxel_cc)
        warnings = ood_warnings(volume, self.card.input_profile)
        if not envelope.any():
            warnings.append("No lung envelope was found; the heuristic cannot assess this scan.")

        candidates = envelope & (hu > self.threshold_hu)
        comps, n = ndimage.label(candidates, structure=ndimage.generate_binary_structure(3, 2))
        spacing_zyx = np.asarray(volume.spacing_zyx)
        kept: list[tuple[int, float]] = []
        rejected: list[int] = []
        boundary = envelope & ~ndimage.binary_erosion(envelope, iterations=1)
        touches_boundary = False
        for i, sl in enumerate(ndimage.find_objects(comps), start=1):
            if sl is None:
                continue
            region = comps[sl] == i
            vol = float(region.sum()) * voxel_cc
            coords = (np.argwhere(region) + [s.start for s in sl]) * spacing_zyx
            if vol >= self.min_volume_cc and self._elongation(coords) <= self.max_elongation:
                kept.append((i, vol))
                touches_boundary |= bool((boundary[sl] & region).sum() > 0.05 * region.sum())
            else:
                rejected.append(i)

        labels = np.zeros(volume.shape, dtype=np.uint8)
        kept.sort(key=lambda kv: kv[1], reverse=True)
        for rank, (i, _) in enumerate(kept):
            labels[comps == i] = 1 if rank == 0 else 3

        score = 1.0 / (1.0 + np.exp(-(hu - self.threshold_hu) / 60.0))
        prob = np.where(envelope, score, 0.0).astype(np.float32)
        if rejected:
            prob[np.isin(comps, rejected)] *= 0.3
        # The score is meaningless at the pleural interface (lung meets chest wall), so uncertainty is
        # only reported inside the envelope, two voxels away from its boundary.
        interior = ndimage.binary_erosion(envelope, iterations=2)
        uncertainty = np.where(interior, binary_entropy(prob), 0.0).astype(np.float32)

        warnings.append("Regions outside the lung envelope were not assessed (mediastinum, hila, chest wall).")
        if touches_boundary:
            warnings.append("A candidate touches the lung boundary; its extent may be truncated.")
        if len(kept) > 1:
            warnings.append(f"{len(kept) - 1} additional candidate(s) labelled 'other lesion'; they need expert review.")
        return SegmentationResult(labels, prob, uncertainty, envelope, warnings)
