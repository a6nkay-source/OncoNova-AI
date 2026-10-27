"""MONAI 3-D U-Net segmenter with Monte Carlo dropout uncertainty.

The checkpoint is produced by scripts/train_segmentation.py. Without PyTorch/MONAI
or without a checkpoint, the model reports itself as unavailable instead of
pretending to run.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from onconova.errors import ModelInputError
from onconova.imaging.normalize import normalize_for_model
from onconova.imaging.volume import Volume
from onconova.segmentation.base import LESION_CLASS_NAMES, ModelCard, SegmentationResult, ood_warnings
from onconova.segmentation.uncertainty import categorical_entropy

try:  # optional dependency
    import torch
    from monai.networks.nets import UNet

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without the dl extra
    TORCH_AVAILABLE = False

NETWORK_KWARGS: dict[str, Any] = {
    "spatial_dims": 3,
    "in_channels": 1,
    "out_channels": 4,
    "channels": (16, 32, 64, 128),
    "strides": (2, 2, 2),
    "num_res_units": 1,
    "dropout": 0.15,
}
DIVISOR = 8  # product of strides


def build_network():
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch and MONAI are required: pip install -e 'backend[dl]'")
    return UNet(**NETWORK_KWARGS)


def pick_device() -> str:
    if not TORCH_AVAILABLE:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


MIN_COMPONENT_CC = 0.1


def remove_small_components(labels: np.ndarray, voxel_cc: float, min_cc: float = MIN_COMPONENT_CC) -> np.ndarray:
    """Post-processing: drop connected lesion components smaller than ``min_cc``."""
    from scipy import ndimage

    comps, n = ndimage.label(labels > 0, structure=ndimage.generate_binary_structure(3, 2))
    if n == 0:
        return labels
    sizes = ndimage.sum(labels > 0, comps, range(1, n + 1)) * voxel_cc
    small = np.isin(comps, np.nonzero(sizes < min_cc)[0] + 1)
    out = labels.copy()
    out[small] = 0
    return out


def _pad_to(x: np.ndarray, divisor: int) -> tuple[np.ndarray, tuple[slice, ...]]:
    pads = [(0, (-s) % divisor) for s in x.shape]
    return np.pad(x, pads, mode="edge"), tuple(slice(0, s) for s in x.shape)


class UNetSegmenter:
    def __init__(self, checkpoint_dir: Path) -> None:
        self.checkpoint_dir = checkpoint_dir
        self._net = None
        self.device = pick_device()
        card_path = checkpoint_dir / "model_card.json"
        info = json.loads(card_path.read_text()) if card_path.exists() else {}
        self.card = ModelCard(
            id="demo-unet",
            name="3-D U-Net (MONAI), synthetic-phantom demo",
            version=info.get("version", "0.1.0"),
            kind="deep-learning",
            modalities=["CT"],
            classes=dict(LESION_CLASS_NAMES),
            trained_on=info.get("trained_on", "Not trained yet."),
            intended_use="Demonstrates the training, inference, uncertainty and evaluation workflow on synthetic CT.",
            limitations=[
                "Trained only on synthetic phantoms; it has never seen a real patient scan.",
                "Lesion classes are defined by the phantom generator, not by pathology.",
                "Uncertainty from MC dropout reflects model variance only; it is not calibrated risk.",
                f"Post-processing removes predicted components smaller than {MIN_COMPONENT_CC} cc.",
            ],
            uncertainty_method="Monte Carlo dropout: predictive entropy over stochastic forward passes",
            checkpoint_sha256=info.get("checkpoint_sha256"),
            training_manifest=info.get("training_manifest"),
            input_profile=info.get("input_profile", {}),
        )
        weights = checkpoint_dir / "model.pt"
        if not TORCH_AVAILABLE:
            self.card.status, self.card.unavailable_reason = "unavailable", "PyTorch/MONAI not installed (pip install -e 'backend[dl]')."
        elif not weights.exists():
            self.card.status, self.card.unavailable_reason = "unavailable", "No checkpoint. Run `make train-seg` to train the demo model."
        elif info.get("checkpoint_sha256") and sha256_file(weights) != info["checkpoint_sha256"]:
            self.card.status, self.card.unavailable_reason = "unavailable", "Checkpoint hash does not match its model card."

    def _network(self):
        if self._net is None:
            net = build_network()
            state = torch.load(self.checkpoint_dir / "model.pt", map_location="cpu", weights_only=True)
            net.load_state_dict(state)
            self._net = net.to(self.device)
        return self._net

    def predict(self, volume: Volume, *, mc_samples: int = 8, saliency: bool = True, **_: Any) -> SegmentationResult:
        if self.card.status != "available":
            raise ModelInputError("model_unavailable", self.card.unavailable_reason or "Model unavailable.")
        if volume.modality != "CT":
            raise ModelInputError("unsupported_modality", "The demo U-Net was trained on CT only.")
        x, _ = normalize_for_model(volume)
        padded, crop = _pad_to(x, DIVISOR)
        net = self._network()
        inp = torch.from_numpy(padded[None, None]).to(self.device)

        net.eval()
        if mc_samples > 0:  # MC dropout: re-enable dropout layers only
            for m in net.modules():
                if m.__class__.__name__.startswith("Dropout"):
                    m.train()
        samples = max(mc_samples, 1)  # 0 -> one deterministic pass
        with torch.no_grad():
            acc = None
            for _ in range(samples):
                p = torch.softmax(net(inp), dim=1)
                acc = p if acc is None else acc + p
            probs = (acc / samples)[0].float().cpu().numpy()
        probs = probs[(slice(None), *crop)]
        labels = remove_small_components(probs.argmax(0).astype(np.uint8), volume.voxel_volume_mm3 / 1000.0)
        uncertainty = categorical_entropy(probs)
        fg = (1.0 - probs[0]).astype(np.float32)

        sal = None
        if saliency and (labels > 0).any():
            sal = self._saliency(net, inp, labels, crop)
        warnings = ood_warnings(volume, self.card.input_profile)
        return SegmentationResult(labels, fg, uncertainty, np.ones(volume.shape, bool), warnings, sal)

    def _saliency(self, net, inp, labels: np.ndarray, crop) -> np.ndarray:
        """|d(mean lesion logit)/d(input)|. Shows sensitivity, not causal reasoning."""
        net.eval()
        x = inp.clone().requires_grad_(True)
        logits = net(x)
        mask = np.zeros(inp.shape[2:], dtype=np.float32)
        mask[crop] = labels > 0
        m = torch.from_numpy(mask).to(self.device)
        lesion_logit = (logits[0, 1:].max(0).values * m).sum() / m.sum().clamp(min=1)
        lesion_logit.backward()
        grad = x.grad[0, 0].abs().float().cpu().numpy()[crop]
        from scipy.ndimage import gaussian_filter

        grad = gaussian_filter(grad, 1.0)
        return (grad / (np.percentile(grad, 99.9) + 1e-8)).clip(0, 1).astype(np.float32)
