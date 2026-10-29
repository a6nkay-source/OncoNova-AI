"""Available segmentation models."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from onconova.errors import NotFoundError
from onconova.segmentation.base import Segmenter
from onconova.segmentation.classical import ClassicalLungLesionSegmenter
from onconova.segmentation.unet import UNetSegmenter

UNET_DIR_NAME = "seg-unet-demo"
# Real-CT models (MSD Task06 Lung, scripts/train_segmentation_msd.py) are offered in the app only once they
# exist and their card records an adoption decision ("adopted": true) backed by the comparison in
# docs/SEGMENTATION_COMPARISON.md. Unadopted candidates are evaluated by path, not listed here.
MSD_UNET_DIR = "seg-unet-msd-ens"
MEDSAM_DIR = "medsam-vit-base"


@lru_cache
def _registry(models_dir: str) -> dict[str, Segmenter]:
    root = Path(models_dir)
    models: dict[str, Segmenter] = {
        "demo-classical": ClassicalLungLesionSegmenter(),
        "demo-unet": UNetSegmenter(root / UNET_DIR_NAME),
    }
    card = root / MSD_UNET_DIR / "model_card.json"
    if card.exists() and json.loads(card.read_text()).get("adopted") is True:
        models["msd-unet-ens"] = UNetSegmenter(root / MSD_UNET_DIR, model_id="msd-unet-ens")
        if (root / MEDSAM_DIR / "pytorch_model.bin").exists():
            from onconova.segmentation.medsam import MedSAMRefiner

            models["msd-unet-medsam"] = MedSAMRefiner(root / MEDSAM_DIR, models["msd-unet-ens"], model_id="msd-unet-medsam")
    return models


def segmenters(models_dir: Path) -> dict[str, Segmenter]:
    return _registry(str(models_dir))


def get_segmenter(models_dir: Path, model_id: str) -> Segmenter:
    models = segmenters(models_dir)
    if model_id not in models:
        raise NotFoundError("unknown_model", f"No segmentation model named {model_id!r}.",
                            details={"available": sorted(models)})
    return models[model_id]


def reload() -> None:
    _registry.cache_clear()
