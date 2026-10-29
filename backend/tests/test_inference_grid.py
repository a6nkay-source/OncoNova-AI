"""Resampling to a model's grid and back, and tiled U-Net inference on scans of any spacing."""

import json

import numpy as np
import pytest

from onconova.imaging.volume import Volume
from onconova.segmentation.grid import channels_to_grid, model_grid, needs_resampling, to_grid
from onconova.segmentation.unet import TORCH_AVAILABLE, UNetSegmenter, sha256_file

FINE = (0.7, 0.7, 1.25)
COARSE = (2.2, 2.2, 2.5)


def ball(spacing, shape_zyx, offset_zyx, radius_mm, direction=None):
    """A ball of ``radius_mm`` centred ``offset_zyx`` voxels from the middle; returns (volume, centre in LPS mm)."""
    vol = Volume(np.zeros(shape_zyx, np.float32), spacing=spacing, origin=(-30.0, 12.0, 40.0), modality="CT",
                 **({"direction": direction} if direction else {}))
    centre = vol.index_to_physical(np.asarray(shape_zyx) / 2 + np.asarray(offset_zyx) + 0.3)  # off the voxel lattice
    idx = np.stack(np.meshgrid(*[np.arange(n) for n in shape_zyx], indexing="ij"), -1)
    d = np.linalg.norm(vol.index_to_physical(idx) - centre, axis=-1)
    return vol.with_array((d <= radius_mm).astype(np.uint8)), centre


def centroid_mm(vol, mask):
    return vol.index_to_physical(np.argwhere(mask).mean(0))


def test_model_grid_covers_the_same_patient_region():
    vol = Volume(np.zeros((40, 50, 60), np.float32), spacing=FINE, origin=(-30.0, 12.0, 40.0), modality="CT")
    grid = model_grid(vol, COARSE)
    assert grid.spacing == COARSE and grid.direction == vol.direction
    for v in (vol, grid):
        v.edges = (v.index_to_physical((0, 0, 0)) - np.asarray(v.spacing) / 2,
                   v.index_to_physical(np.asarray(v.shape) - 1) + np.asarray(v.spacing) / 2)
    np.testing.assert_allclose(grid.edges[0], vol.edges[0], atol=1e-6)  # first edges coincide exactly
    assert np.all(np.abs(grid.edges[1] - vol.edges[1]) <= np.asarray(COARSE) / 2 + 1e-6)  # last within half a model voxel
    assert needs_resampling(vol, COARSE) and not needs_resampling(grid, COARSE) and not needs_resampling(vol, None)


@pytest.mark.parametrize("direction", [None, (0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)], ids=["LPS", "rotated"])
def test_a_lesion_keeps_its_position_and_size_through_the_model_grid(direction):
    """Down to 2.2 mm and back: the round trip a segmentation takes, applied to the true mask."""
    vol, centre = ball(FINE, (64, 96, 96), (3, -7, 11), 9.0, direction)
    grid = model_grid(vol, COARSE)
    coarse = to_grid(vol.array.astype(np.float32), vol, grid)  # partial-volume fractions on the model grid
    back = to_grid(coarse, grid, vol) >= 0.5
    np.testing.assert_allclose(centroid_mm(vol, vol.array > 0), centre, atol=0.2)
    np.testing.assert_allclose(centroid_mm(vol, back), centre, atol=max(COARSE) / 4)
    assert back.sum() / (vol.array > 0).sum() == pytest.approx(1.0, abs=0.06)


def test_class_probabilities_still_sum_to_one_on_the_scan_grid():
    rng = np.random.default_rng(0)
    grid = Volume(np.zeros((10, 12, 14), np.float32), spacing=COARSE, modality="CT")
    logits = rng.normal(size=(4, *grid.shape))
    probs = np.exp(logits) / np.exp(logits).sum(0)
    scan = model_grid(grid, FINE)
    total = sum(channels_to_grid(probs.astype(np.float32), grid, scan))
    np.testing.assert_allclose(total, 1.0, atol=1e-5)


@pytest.fixture
def tiny_unet(tmp_path):
    """A randomly initialised small U-Net with a card that fixes its grid and patch size."""
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch/MONAI not installed")
    import torch

    from onconova.segmentation.unet import build_network

    kwargs = {"channels": [4, 8, 16, 32]}
    torch.manual_seed(0)
    d = tmp_path / "tiny"
    d.mkdir()
    torch.save(build_network(kwargs).state_dict(), d / "model.pt")
    card = {"network": kwargs, "checkpoint_sha256": sha256_file(d / "model.pt"),
            "inference": {"spacing_xyz": [2.0, 2.0, 2.0], "patch_zyx": [16, 32, 32]},
            "input_profile": {"source": "synthetic", "modality": "CT", "spacing_min": [2.0, 2.0, 2.0], "spacing_max": [2.0, 2.0, 2.0]}}
    (d / "model_card.json").write_text(json.dumps(card))
    return UNetSegmenter(d)


def test_unet_resamples_tiles_and_answers_on_the_scan_grid(tiny_unet):
    assert tiny_unet.card.status == "available", tiny_unet.card.unavailable_reason
    rng = np.random.default_rng(1)
    scan = Volume(rng.normal(-600, 300, (50, 70, 90)).astype(np.float32), spacing=(0.8, 0.8, 1.0), modality="CT")
    scan.metadata["demo"] = True
    res = tiny_unet.predict(scan, mc_samples=2, saliency=True)
    assert res.labels.shape == res.uncertainty.shape == res.foreground_probability.shape == scan.shape
    assert 0 <= res.foreground_probability.min() and res.foreground_probability.max() <= 1
    assert res.uncertainty.max() <= 1 + 1e-6
    assert res.saliency is None or res.saliency.shape == scan.shape
    assert any(w.startswith("Resampled from [0.8, 0.8, 1.0] mm") for w in res.warnings)
    assert not any(w.startswith("Voxel spacing") for w in res.warnings)  # replaced by the resampling note

    # A scan already on the model grid is not resampled, and larger than one patch it is tiled.
    native = Volume(rng.normal(-600, 300, (24, 40, 44)).astype(np.float32), spacing=(2.0, 2.0, 2.0), modality="CT")
    native.metadata["demo"] = True
    res = tiny_unet.predict(native, mc_samples=0, saliency=False)
    assert res.labels.shape == native.shape and not any(w.startswith("Resampled") for w in res.warnings)


def test_mirror_tta_averages_valid_probabilities(tiny_unet):
    rng = np.random.default_rng(2)
    scan = Volume(rng.normal(-600, 300, (24, 40, 44)).astype(np.float32), spacing=(2.0, 2.0, 2.0), modality="CT")
    scan.metadata["demo"] = True
    plain = tiny_unet.predict(scan, mc_samples=0, saliency=False)
    tta = tiny_unet.predict(scan, mc_samples=0, saliency=False, tta=True)
    assert tta.labels.shape == scan.shape
    assert 0 <= tta.foreground_probability.min() and tta.foreground_probability.max() <= 1
    assert not np.allclose(tta.foreground_probability, plain.foreground_probability)  # the mirrors were really used
