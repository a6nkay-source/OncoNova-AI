"""Available segmentation models."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from onconova.errors import NotFoundError
from onconova.segmentation.base import Segmenter
from onconova.segmentation.classical import ClassicalLungLesionSegmenter
from onconova.segmentation.unet import UNetSegmenter

UNET_DIR_NAME = "seg-unet-demo"


@lru_cache
def _registry(models_dir: str) -> dict[str, Segmenter]:
    return {
        "demo-classical": ClassicalLungLesionSegmenter(),
        "demo-unet": UNetSegmenter(Path(models_dir) / UNET_DIR_NAME),
    }


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
