"""Physical volume calculations and segmentation metrics."""

import numpy as np
import pytest

from onconova.segmentation.measure import max_axial_diameter_mm, sphericity, volume_cc
from onconova.segmentation.metrics import dice, evaluate_pair, hausdorff, iou, sensitivity, specificity


def sphere_mask(shape, spacing_zyx, radius_mm):
    zz, yy, xx = np.meshgrid(*[(np.arange(n) - (n - 1) / 2) * s for n, s in zip(shape, spacing_zyx)], indexing="ij")
    return zz ** 2 + yy ** 2 + xx ** 2 <= radius_mm ** 2


def test_sphere_volume_with_anisotropic_spacing():
    spacing_xyz = (0.7, 0.9, 2.5)
    mask = sphere_mask((60, 80, 100), spacing_xyz[::-1], 20.0)
    expected = 4 / 3 * np.pi * 20.0 ** 3 / 1000
    assert volume_cc(mask, spacing_xyz) == pytest.approx(expected, rel=0.03)


def test_volume_scales_with_voxel_size():
    mask = np.ones((4, 5, 6), bool)
    assert volume_cc(mask, (1, 1, 1)) == pytest.approx(0.12)
    assert volume_cc(mask, (2, 1, 1)) == pytest.approx(0.24)


def test_overlap_metrics_known_values():
    a = np.zeros((10, 10, 10), bool)
    b = np.zeros_like(a)
    a[0:4] = True  # 400 voxels
    b[2:6] = True  # 400 voxels, 200 shared
    assert dice(a, b) == pytest.approx(0.5)
    assert iou(a, b) == pytest.approx(200 / 600)
    assert sensitivity(a, b) == pytest.approx(0.5)
    assert specificity(a, b) == pytest.approx(400 / 600)
    roi = np.zeros_like(a)
    roi[0:8] = True  # 800 voxels: 200 TP, 200 FP, 200 FN, 200 TN
    assert specificity(a, b, roi) == pytest.approx(0.5)


def test_empty_mask_conventions():
    e = np.zeros((5, 5, 5), bool)
    m = e.copy()
    m[2, 2, 2] = True
    assert dice(e, e) == 1.0 and hausdorff(e, e, (1, 1, 1)) == 0.0
    assert dice(m, e) == 0.0 and hausdorff(m, e, (1, 1, 1)) is None
    assert sensitivity(m, e) is None


def test_hausdorff_in_millimetres():
    a = np.zeros((20, 5, 5), bool)
    b = np.zeros_like(a)
    a[2, 2, 2] = True
    b[10, 2, 2] = True
    assert hausdorff(a, b, (2.5, 1.0, 1.0)) == pytest.approx(20.0)
    assert hausdorff(a, b, (2.5, 1.0, 1.0), 95) <= hausdorff(a, b, (2.5, 1.0, 1.0))


def test_evaluate_pair_identical_masks():
    m = sphere_mask((20, 20, 20), (1, 1, 1), 6)
    out = evaluate_pair(m, m, (1.0, 1.0, 1.0))
    assert out["dice"] == 1.0 and out["hausdorff_mm"] == 0.0 and out["pred_volume_cc"] == out["ref_volume_cc"]


def test_shape_descriptors():
    sph = sphere_mask((40, 40, 40), (1, 1, 1), 12)
    box = np.zeros((40, 40, 40), bool)
    box[18:22, 18:22, 2:38] = True
    assert sphericity(sph, (1, 1, 1)) > 0.85
    assert sphericity(box, (1, 1, 1)) < sphericity(sph, (1, 1, 1))
    assert max_axial_diameter_mm(sph, (1.0, 1.0, 1.0)) == pytest.approx(25, abs=2)
