"""MedSAM (Ma et al., Nat Commun 2024, PMID 38253604) as a refiner of another model's detections.

MedSAM is a promptable 2-D model: it segments what a box points at. Here the boxes
come only from another model's 3-D segmentation (one box per connected region on each
axial slice, plus a small margin), never from reference masks, so the pipeline is
fully automatic: detector, then MedSAM. MedSAM cannot find a tumour the detector
missed, and it outlines whatever a false-positive box contains.

Preprocessing follows the original MedSAM inference code rather than the Hugging Face
processor defaults: CT lung window (W 1500, L -160, as in the paper) scaled to 0-255,
three channels, bicubic resize to 1024 x 1024, min-max scaled to [0, 1] with no
ImageNet normalisation; boxes scaled to the 1024 grid; mask logits upsampled
bilinearly to the slice and thresholded at probability 0.5.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# The weights are read from a local folder only; the Hugging Face hub is never contacted.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
from scipy import ndimage

from onconova.errors import ModelInputError
from onconova.imaging.volume import Volume
from onconova.segmentation.base import ModelCard, SegmentationResult
from onconova.segmentation.uncertainty import binary_entropy
from onconova.segmentation.unet import pick_device

try:  # optional dependency
    import torch
    import torch.nn.functional as F
    from transformers import SamModel

    MEDSAM_AVAILABLE = True
except ImportError:  # pragma: no cover
    MEDSAM_AVAILABLE = False

WINDOW = (-160.0, 1500.0)  # level, width (MedSAM paper, lung)
SIZE = 1024
BOX_MARGIN_PX = 5  # MedSAM was trained with boxes perturbed by 0-20 px at 1024 (about 0-10 px on a 512 slice)


def window_slice(hu: np.ndarray) -> np.ndarray:
    level, width = WINDOW
    lo, hi = level - width / 2, level + width / 2
    return ((np.clip(hu, lo, hi) - lo) / (hi - lo) * 255.0).astype(np.float32)


def slice_boxes(mask2d: np.ndarray, margin: int = BOX_MARGIN_PX) -> list[tuple[int, int, int, int]]:
    """(x0, y0, x1, y1) inclusive pixel boxes, one per connected region, expanded by ``margin``."""
    comps, _ = ndimage.label(mask2d)
    h, w = mask2d.shape
    boxes = []
    for sl in ndimage.find_objects(comps):
        if sl is None:
            continue
        y0, y1, x0, x1 = sl[0].start, sl[0].stop - 1, sl[1].start, sl[1].stop - 1
        boxes.append((max(0, x0 - margin), max(0, y0 - margin), min(w - 1, x1 + margin), min(h - 1, y1 + margin)))
    return boxes


class MedSAMRefiner:
    def __init__(self, weights_dir: Path, detector, *, model_id: str) -> None:
        self.weights_dir = weights_dir
        self.detector = detector
        self.device = pick_device()
        self._model = None
        self.card = ModelCard(
            id=model_id,
            name=f"MedSAM (ViT-B) refining {detector.card.id} detections",
            version="medsam-vit-base",
            kind="foundation-model-refiner",
            modalities=["CT"],
            classes={1: "primary_tumor"},
            trained_on=("MedSAM: SAM ViT-B fine-tuned on 1,570,263 public medical image-mask pairs (Ma et al. 2024). "
                        f"Boxes from {detector.card.id}."),
            intended_use="Research comparison: automatic box prompts from a detector, refined slice by slice by MedSAM.",
            limitations=[
                "Needs a box: it cannot find tumours the detector missed, and it outlines whatever a false box contains.",
                "2-D, slice by slice: no 3-D context, so outlines can jitter between slices.",
                "MedSAM's training data came from public challenge datasets; if MSD Task06 was among them, scores on "
                "MSD cases are optimistic.",
                "The probability is MedSAM's mask score, not calibrated.",
            ],
            uncertainty_method="Binary entropy of MedSAM's mask probability inside the prompt boxes (not calibrated)",
            label="RESEARCH MODEL - public foundation model, not validated for clinical use.",
            input_profile=dict(detector.card.input_profile),
        )
        if not MEDSAM_AVAILABLE:
            self.card.status, self.card.unavailable_reason = "unavailable", "transformers is not installed (pip install transformers)."
        elif not (weights_dir / "pytorch_model.bin").exists():
            self.card.status, self.card.unavailable_reason = "unavailable", f"No MedSAM weights in {weights_dir.name}."
        elif detector.card.status != "available":
            self.card.status, self.card.unavailable_reason = "unavailable", f"Its detector is unavailable: {detector.card.unavailable_reason}"

    def _sam(self):
        if self._model is None:
            self._model = SamModel.from_pretrained(str(self.weights_dir), local_files_only=True).to(self.device).eval()
        return self._model

    def _segment_slice(self, hu: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
        """Max MedSAM probability over ``boxes`` on one slice, shape (y, x)."""
        h, w = hu.shape
        img = torch.from_numpy(window_slice(hu))[None, None].to(self.device)
        img = F.interpolate(img, size=(SIZE, SIZE), mode="bicubic", align_corners=False)[0].clamp(0, 255)
        img = (img - img.min()) / (img.max() - img.min()).clamp(min=1e-8)
        pixel_values = img.expand(3, SIZE, SIZE)[None]
        scale = torch.tensor([SIZE / w, SIZE / h, SIZE / w, SIZE / h], device=self.device)
        box_t = torch.tensor(boxes, dtype=torch.float32, device=self.device) * scale
        model = self._sam()
        with torch.no_grad():
            emb = model.get_image_embeddings(pixel_values)
            out = model(image_embeddings=emb, input_boxes=box_t[None], multimask_output=False)
            logits = out.pred_masks[0, :, 0]  # (n_boxes, 256, 256)
            probs = torch.sigmoid(F.interpolate(logits[:, None], size=(h, w), mode="bilinear", align_corners=False))[:, 0]
        return probs.max(0).values.float().cpu().numpy()

    def predict(self, volume: Volume, **kwargs: Any) -> SegmentationResult:
        if self.card.status != "available":
            raise ModelInputError("model_unavailable", self.card.unavailable_reason or "Model unavailable.")
        if volume.modality != "CT":
            raise ModelInputError("unsupported_modality", "MedSAM refinement is set up for CT here.")
        det = self.detector.predict(volume, mc_samples=0, saliency=False)
        detected = det.labels > 0
        prob = np.zeros(volume.shape, np.float32)
        prompted = np.zeros(volume.shape, bool)
        n_boxes = 0
        for z in np.unique(np.nonzero(detected)[0]):
            boxes = slice_boxes(detected[z])
            n_boxes += len(boxes)
            prob[z] = self._segment_slice(volume.array[z], boxes)
            for x0, y0, x1, y1 in boxes:
                prompted[z, y0:y1 + 1, x0:x1 + 1] = True
        prob[~prompted] = 0.0  # nothing outside a prompt box
        labels = (prob >= 0.5).astype(np.uint8)
        warnings = [w for w in det.warnings]
        warnings.append(f"MedSAM refined {n_boxes} box(es) from {self.detector.card.id} on "
                        f"{len(np.unique(np.nonzero(detected)[0]))} axial slice(s); regions the detector missed stay missed.")
        uncertainty = np.where(prompted, binary_entropy(prob), 0.0).astype(np.float32)
        return SegmentationResult(labels, prob, uncertainty, np.ones(volume.shape, bool), warnings, None)
