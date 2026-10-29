"""MedSAM refinement: prompts come from a detector's output only, and missing pieces give an unavailable state."""

import numpy as np

from onconova.segmentation.medsam import BOX_MARGIN_PX, MedSAMRefiner, slice_boxes, window_slice


class _Detector:
    class card:
        id = "det"
        status = "available"
        unavailable_reason = None
        input_profile = {"modality": "CT"}


def test_one_box_per_detected_region_with_a_margin_clipped_to_the_slice():
    mask = np.zeros((40, 50), bool)
    mask[10:15, 20:26] = True  # rows 10-14, cols 20-25
    mask[0:3, 47:50] = True  # touches the top-right corner
    boxes = sorted(slice_boxes(mask))
    m = BOX_MARGIN_PX
    assert boxes == sorted([(20 - m, 10 - m, 25 + m, 14 + m), (47 - m, 0, 49, 2 + m)])
    assert slice_boxes(np.zeros((8, 8), bool)) == []


def test_lung_window_matches_the_medsam_paper():
    # W 1500, L -160: -910 HU -> 0, 590 HU -> 255.
    np.testing.assert_allclose(window_slice(np.array([-2000.0, -910.0, -160.0, 590.0, 3000.0])), [0, 0, 127.5, 255, 255])


def test_unavailable_without_weights(tmp_path):
    refiner = MedSAMRefiner(tmp_path / "missing", _Detector(), model_id="medsam-test")
    assert refiner.card.status == "unavailable"
    assert "MedSAM weights" in refiner.card.unavailable_reason or "transformers" in refiner.card.unavailable_reason
