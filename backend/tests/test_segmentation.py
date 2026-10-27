"""Segmentation output shape, uncertainty, OOD warnings and model availability."""

import json

import numpy as np
import pytest

from onconova.errors import ModelInputError
from onconova.imaging.volume import Volume
from onconova.segmentation.classical import ClassicalLungLesionSegmenter
from onconova.segmentation.metrics import dice
from onconova.segmentation.registry import UNET_DIR_NAME
from onconova.segmentation.uncertainty import binary_entropy, categorical_entropy, unreliable_regions
from onconova.segmentation.unet import TORCH_AVAILABLE, UNetSegmenter, remove_small_components


def test_classical_output_matches_input_geometry(phantom_case):
    _, renders = phantom_case
    image, lesions, _ = renders[0]
    image.metadata["demo"] = True
    res = ClassicalLungLesionSegmenter().predict(image)
    assert res.labels.shape == image.shape and res.labels.dtype == np.uint8
    assert set(np.unique(res.labels)) <= {0, 1, 3}
    for arr in (res.foreground_probability, res.uncertainty):
        assert arr.shape == image.shape and arr.min() >= 0 and arr.max() <= 1
    assert dice(res.labels == 1, lesions.array == 1) > 0.8
    assert any("not assessed" in w for w in res.warnings)


def test_classical_rejects_non_ct():
    vol = Volume(np.zeros((16, 16, 16), np.float32), (1, 1, 1), modality="MR")
    with pytest.raises(ModelInputError) as e:
        ClassicalLungLesionSegmenter().predict(vol)
    assert e.value.code == "unsupported_modality"


def test_out_of_distribution_warnings(phantom_case):
    _, renders = phantom_case
    image = renders[0][0]
    odd = Volume(image.array, (0.6, 0.6, 5.0), image.origin, modality="CT", metadata={})
    warnings = ClassicalLungLesionSegmenter().predict(odd).warnings
    assert any("spacing" in w for w in warnings)
    assert any("synthetic phantoms" in w for w in warnings)  # not a demo case -> real-scan warning


def test_entropy_bounds():
    p = np.array([0.0, 0.5, 1.0])
    np.testing.assert_allclose(binary_entropy(p), [0, 1, 0], atol=1e-4)
    probs = np.array([[0.25, 1.0], [0.25, 0.0], [0.25, 0.0], [0.25, 0.0]])
    np.testing.assert_allclose(categorical_entropy(probs), [1.0, 0.0], atol=1e-4)


def test_unreliable_regions_ignore_remote_noise():
    vol = Volume(np.zeros((30, 30, 30), np.float32), (1, 1, 1))
    unc = np.zeros(vol.shape, np.float32)
    labels = np.zeros(vol.shape, np.uint8)
    labels[10:14, 10:14, 10:14] = 1
    unc[9:15, 9:15, 9:15] = 0.9  # band around the lesion
    unc[25, 25, 25] = 0.9  # tiny remote speck
    regions = unreliable_regions(unc, vol, labels)
    assert len(regions) == 1 and regions[0]["location"] == "at or near a predicted lesion"


def test_remove_small_components():
    labels = np.zeros((20, 20, 20), np.uint8)
    labels[2:8, 2:8, 2:8] = 1  # 216 voxels
    labels[15, 15, 15] = 3  # 1 voxel
    out = remove_small_components(labels, voxel_cc=0.001, min_cc=0.01)
    assert out[15, 15, 15] == 0 and (out == 1).sum() == 216


def test_unet_reports_unavailable_without_checkpoint(tmp_path):
    seg = UNetSegmenter(tmp_path / "missing")
    assert seg.card.status == "unavailable" and seg.card.unavailable_reason
    with pytest.raises(ModelInputError) as e:
        seg.predict(Volume(np.zeros((16, 16, 16), np.float32), (1, 1, 1), modality="CT"))
    assert e.value.code == "model_unavailable"


def test_unet_rejects_tampered_checkpoint(tmp_path):
    d = tmp_path / UNET_DIR_NAME
    d.mkdir()
    (d / "model.pt").write_bytes(b"tampered")
    (d / "model_card.json").write_text(json.dumps({"checkpoint_sha256": "0" * 64}))
    if TORCH_AVAILABLE:
        assert UNetSegmenter(d).card.unavailable_reason == "Checkpoint hash does not match its model card."


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch/MONAI not installed")
def test_unet_output_shape_if_trained(phantom_case):
    from onconova.config import get_settings

    seg = UNetSegmenter(get_settings().models_dir / UNET_DIR_NAME)
    if seg.card.status != "available":
        pytest.skip(seg.card.unavailable_reason)
    image = phantom_case[1][0][0]
    res = seg.predict(image, mc_samples=2, saliency=True)
    assert res.labels.shape == image.shape and res.uncertainty.max() <= 1.0
    assert res.saliency is None or res.saliency.shape == image.shape
