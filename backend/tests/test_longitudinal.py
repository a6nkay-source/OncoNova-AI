"""Longitudinal alignment and observed-change analysis."""

import numpy as np
import pytest

from onconova.imaging.registration import register_rigid, resample_to
from onconova.longitudinal.compare import TimepointInput, analyze
from onconova.segmentation.metrics import dice


def test_registration_recovers_patient_shift(phantom_case):
    spec, renders = phantom_case
    base, follow = renders[0][0], renders[1][0]
    reg = register_rigid(base, follow)
    true_shift = np.array(spec.timepoints[1].shift_mm)
    assert np.linalg.norm(np.array(reg.translation_mm) - true_shift) < 1.5
    assert reg.qc == "ok"


def test_registration_aligns_anatomy(phantom_case):
    _, renders = phantom_case
    base_img, _, base_struct = renders[0]
    fu_img, _, fu_struct = renders[1]
    reg = register_rigid(base_img, fu_img)
    before = dice(base_struct.array == 3, fu_struct.array == 3)
    moved = resample_to(base_img, fu_struct, reg.transform, is_label=True)
    after = dice(base_struct.array == 3, moved == 3)
    assert after > before and after > 0.9


def test_change_analysis_separates_persisting_and_new(phantom_case):
    spec, renders = phantom_case
    tps = [TimepointInput(tp.tp_id, tp.days, r[0], r[1].array) for tp, r in zip(spec.timepoints, renders)]
    result = analyze(tps, source="reference")
    primary = next(t for t in result["tracks"] if t["class"] == "primary_tumor")
    assert primary["status"] == "persisting" and primary["volume_change_pct"] > 0
    new = [t for t in result["tracks"] if t["status"] == "new"]
    assert len(new) == 1 and new[0]["first_seen"] == "T2"
    review = [f for f in result["findings"] if f["kind"] == "requires_expert_review"]
    assert any("not by itself evidence of metastasis" in f["text"] for f in review)
    assert all(f["kind"] != "model_output" for f in result["findings"])  # observed change is not a prediction
    assert result["measurement_kind"] == "measured"


def test_model_mask_measurements_are_labelled_as_such(phantom_case):
    spec, renders = phantom_case
    tps = [TimepointInput(tp.tp_id, tp.days, r[0], r[1].array) for tp, r in list(zip(spec.timepoints, renders))[:2]]
    assert analyze(tps, source="pred:demo-classical")["measurement_kind"] == "measured_from_model_output"


def test_single_timepoint_is_rejected(phantom_case):
    spec, renders = phantom_case
    with pytest.raises(ValueError):
        analyze([TimepointInput("T0", 0, renders[0][0], renders[0][1].array)], source="reference")
