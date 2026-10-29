"""MONAI 3-D U-Net segmenter with Monte Carlo dropout uncertainty.

The checkpoint is produced by scripts/train_segmentation.py. Without PyTorch/MONAI
or without a checkpoint, the model reports itself as unavailable instead of
pretending to run.

Inference runs where the model was trained: the scan is resampled to the training
spacing (``grid.model_grid``), segmented in overlapping tiles of the training patch
size with Gaussian blending, and the class probabilities are resampled back onto the
scan's own grid before the argmax. Scans already at the training spacing are not
resampled.
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
from onconova.segmentation.grid import channels_to_grid, model_grid, needs_resampling, to_grid
from onconova.segmentation.uncertainty import categorical_entropy

try:  # optional dependency
    import torch
    from monai.inferers import sliding_window_inference
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
DEFAULT_PATCH_ZYX = (48, 80, 80)  # training patch of the demo model (configs/segmentation_unet_demo.yaml)
TILE_OVERLAP = 0.25
SALIENCY_MAX_VOXELS = 8_000_000  # whole-volume gradient on the model grid; skipped above this
MIRRORS = [(), (4,), (3,), (2,), (3, 4), (2, 4), (2, 3), (2, 3, 4)]  # tensor dims (N, C, z, y, x) to flip


def build_network(kwargs: dict[str, Any] | None = None):
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch and MONAI are required: pip install -e 'backend[dl]'")
    kw = {**NETWORK_KWARGS, **(kwargs or {})}
    return UNet(**{k: tuple(v) if isinstance(v, list) else v for k, v in kw.items()})


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
    """One checkpoint, or an ensemble whose card lists ``members`` (probabilities are averaged).

    The demo model's wording is the default; a card may override id, name, label,
    training description, intended use, limitations and classes.
    """

    def __init__(self, checkpoint_dir: Path, *, model_id: str = "demo-unet") -> None:
        self.checkpoint_dir = checkpoint_dir
        self._nets: list | None = None
        self.device = pick_device()
        card_path = checkpoint_dir / "model_card.json"
        info = json.loads(card_path.read_text()) if card_path.exists() else {}
        self.network_kwargs = info.get("network") or {}
        # Where the model works. Older cards have no ``inference`` block: a profile with a single
        # spacing is the training spacing, and the demo patch size applies.
        inference = info.get("inference", {})
        profile = info.get("input_profile", {})
        lo, hi = profile.get("spacing_min"), profile.get("spacing_max")
        spacing = inference.get("spacing_xyz") or (lo if lo and hi and np.allclose(lo, hi) else None)
        self.spacing_xyz = tuple(float(v) for v in spacing) if spacing else None
        self.patch_zyx = tuple(int(v) for v in inference.get("patch_zyx", DEFAULT_PATCH_ZYX))
        text = info.get("card", {})
        self.card = ModelCard(
            id=text.get("id", model_id),
            name=text.get("name", "3-D U-Net (MONAI), synthetic-phantom demo"),
            version=info.get("version", "0.1.0"),
            kind="deep-learning",
            modalities=["CT"],
            classes={int(k): v for k, v in text["classes"].items()} if "classes" in text else dict(LESION_CLASS_NAMES),
            trained_on=info.get("trained_on", "Not trained yet."),
            intended_use=text.get("intended_use",
                                  "Demonstrates the training, inference, uncertainty and evaluation workflow on synthetic CT."),
            limitations=text.get("limitations", [
                "Trained only on synthetic phantoms; it has never seen a real patient scan.",
                "Lesion classes are defined by the phantom generator, not by pathology.",
                "Uncertainty from MC dropout reflects model variance only; it is not calibrated risk.",
            ]) + [f"Post-processing removes predicted components smaller than {MIN_COMPONENT_CC} cc."],
            uncertainty_method="Monte Carlo dropout: predictive entropy over stochastic forward passes"
            + (" of every ensemble member" if info.get("members") else ""),
            checkpoint_sha256=info.get("checkpoint_sha256"),
            training_manifest=info.get("training_manifest"),
            input_profile=info.get("input_profile", {}),
            **({"label": text["label"]} if "label" in text else {}),
        )
        # (weights file, expected sha256) for a single model or each ensemble member.
        if info.get("members"):
            self.weights = [((checkpoint_dir / m["file"]).resolve(), m.get("sha256")) for m in info["members"]]
        else:
            self.weights = [(checkpoint_dir / "model.pt", info.get("checkpoint_sha256"))]
        if not TORCH_AVAILABLE:
            self.card.status, self.card.unavailable_reason = "unavailable", "PyTorch/MONAI not installed (pip install -e 'backend[dl]')."
        elif not all(path.exists() for path, _ in self.weights):
            self.card.status, self.card.unavailable_reason = "unavailable", "No checkpoint. Run `make train-seg` to train the demo model."
        elif any(sha and sha256_file(path) != sha for path, sha in self.weights):
            self.card.status, self.card.unavailable_reason = "unavailable", "Checkpoint hash does not match its model card."

    def _networks(self) -> list:
        if self._nets is None:
            self._nets = []
            for path, _ in self.weights:
                net = build_network(self.network_kwargs)
                net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
                self._nets.append(net.to(self.device))
        return self._nets

    def predict(self, volume: Volume, *, mc_samples: int = 8, saliency: bool = True, tta: bool = False,
                **_: Any) -> SegmentationResult:
        """``tta`` averages the 8 mirrorings over z, y and x (8x slower); off unless it is shown to help."""
        if self.card.status != "available":
            raise ModelInputError("model_unavailable", self.card.unavailable_reason or "Model unavailable.")
        if volume.modality != "CT":
            raise ModelInputError("unsupported_modality", "The demo U-Net was trained on CT only.")
        x, _ = normalize_for_model(volume)
        grid = model_grid(volume, self.spacing_xyz) if needs_resampling(volume, self.spacing_xyz) else None
        if grid is not None:
            x = to_grid(x, volume, grid)
        nets = self._networks()
        inp = torch.from_numpy(np.ascontiguousarray(x)[None, None]).to(self.device)

        samples = max(mc_samples, 1)  # 0 -> one deterministic pass
        flips = MIRRORS if tta else [()]
        acc = None
        with torch.no_grad():
            for net in nets:  # ensemble members, each with its own MC-dropout samples
                net.eval()
                if mc_samples > 0:  # MC dropout: re-enable dropout layers only
                    for m in net.modules():
                        if m.__class__.__name__.startswith("Dropout"):
                            m.train()
                for _ in range(samples):
                    for dims in flips:
                        logits = sliding_window_inference(torch.flip(inp, dims) if dims else inp, roi_size=self.patch_zyx,
                                                          sw_batch_size=2, predictor=net, overlap=TILE_OVERLAP,
                                                          mode="gaussian", padding_mode="replicate")
                        p = torch.softmax(torch.flip(logits, dims) if dims else logits, dim=1)
                        acc = p if acc is None else acc + p
        probs = (acc / (samples * len(nets) * len(flips)))[0].float().cpu().numpy()

        if grid is None:
            labels = probs.argmax(0).astype(np.uint8)
            fg = (1.0 - probs[0]).astype(np.float32)
            uncertainty = categorical_entropy(probs)
        else:
            labels, fg, uncertainty = self._back_to_scan(probs, grid, volume)
        model_labels = probs.argmax(0).astype(np.uint8)
        labels = remove_small_components(labels, volume.voxel_volume_mm3 / 1000.0)

        sal = None
        if saliency and (labels > 0).any() and model_labels.size <= SALIENCY_MAX_VOXELS:
            sal = self._saliency(nets[0], x, model_labels)
            if grid is not None:
                sal = to_grid(sal, grid, volume).clip(0, 1).astype(np.float32)
        warnings = ood_warnings(volume, self.card.input_profile)
        if grid is not None:
            warnings = [w for w in warnings if not w.startswith("Voxel spacing")]
            warnings.append(
                f"Resampled from {np.round(volume.spacing, 2).tolist()} mm to the model's "
                f"{list(self.spacing_xyz)} mm grid and back; structures smaller than about "
                f"{max(self.spacing_xyz):.1f} mm can be lost or misplaced by up to half a model voxel."
            )
        return SegmentationResult(labels, fg, uncertainty, np.ones(volume.shape, bool), warnings, sal)

    @staticmethod
    def _back_to_scan(probs: np.ndarray, grid: Volume, volume: Volume) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Argmax, foreground probability and entropy on the scan grid, one class channel at a time."""
        labels = np.zeros(volume.shape, np.uint8)
        best = entropy = fg = None
        for c, pc in enumerate(channels_to_grid(probs, grid, volume)):
            pc = np.clip(pc, 0.0, 1.0)
            term = -pc * np.log(np.clip(pc, 1e-6, 1.0))
            entropy = term if entropy is None else entropy + term
            if c == 0:
                fg, best = (1.0 - pc).astype(np.float32), pc
            else:
                better = pc > best
                labels[better] = c
                best = np.where(better, pc, best)
        return labels, fg, (entropy / np.log(probs.shape[0])).astype(np.float32)

    def _saliency(self, net, x: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """|d(mean lesion logit)/d(input)| on the model grid. Shows sensitivity, not causal reasoning."""
        net.eval()
        padded, crop = _pad_to(x, DIVISOR)
        inp = torch.from_numpy(padded[None, None]).to(self.device).requires_grad_(True)
        logits = net(inp)
        mask = np.zeros(padded.shape, dtype=np.float32)
        mask[crop] = labels > 0
        m = torch.from_numpy(mask).to(self.device)
        lesion_logit = (logits[0, 1:].max(0).values * m).sum() / m.sum().clamp(min=1)
        lesion_logit.backward()
        grad = inp.grad[0, 0].abs().float().cpu().numpy()[crop]
        from scipy.ndimage import gaussian_filter

        grad = gaussian_filter(grad, 1.0)
        return (grad / (np.percentile(grad, 99.9) + 1e-8)).clip(0, 1).astype(np.float32)
