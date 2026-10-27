"""Progression metrics, uncertainty and missing-data handling."""

import numpy as np
import pytest
from sklearn.metrics import brier_score_loss, roc_auc_score

from onconova.progression import cohort
from onconova.progression.features import FEATURES, from_longitudinal
from onconova.progression.metrics import binary_report, calibration_curve, concordance_index
from onconova.progression.model import ProgressionModel

LABEL = f"progression_within_{cohort.HORIZON_DAYS}d"


@pytest.fixture(scope="module")
def model():
    rows = [r for r in cohort.generate(400, seed=3) if r[LABEL] is not None]
    X = np.array([[np.nan if r[f] is None else float(r[f]) for f in FEATURES] for r in rows])
    y = np.array([r[LABEL] for r in rows])
    return ProgressionModel.fit(X, y, seed=0, n_bootstrap=20, card={"outcome_definition": "sim", "horizon_days": 180})


def features(**override):
    base = {"baseline_total_volume_cc": 14.0, "log_volume_ratio_per_100d": 0.3, "new_lesion_count": 0,
            "node_region_lesion_observed": 0, "primary_sphericity": 0.8, "age_years": 64.0}
    return {**base, **override}


def test_concordance_index_known_values():
    t = np.array([1.0, 2.0, 3.0, 4.0])
    e = np.array([1, 1, 1, 0])
    assert concordance_index(t, e, np.array([4.0, 3.0, 2.0, 1.0])) == 1.0
    assert concordance_index(t, e, np.array([1.0, 2.0, 3.0, 4.0])) == 0.0
    assert concordance_index(t, e, np.ones(4)) == 0.5


def test_binary_report_matches_sklearn_and_has_intervals():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    p = np.clip(0.3 * y + rng.uniform(0, 0.7, 300), 0, 1)
    r = binary_report(y, p, 0.5, n_boot=200)
    assert r["auroc"]["value"] == pytest.approx(roc_auc_score(y, p), abs=1e-4)
    assert r["brier"]["value"] == pytest.approx(brier_score_loss(y, p), abs=1e-4)
    lo, hi = r["auroc"]["ci95"]
    assert lo <= r["auroc"]["value"] <= hi
    assert sum(b["n"] for b in r["calibration"]["bins"]) == 300


def test_single_class_is_reported_unavailable():
    assert binary_report(np.zeros(10), np.linspace(0, 1, 10), 0.5)["status"] == "unavailable"


def test_calibration_of_perfectly_calibrated_scores():
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, 20000)
    y = (rng.uniform(0, 1, 20000) < p).astype(int)
    assert calibration_curve(y, p)["expected_calibration_error"] < 0.03


def test_missing_required_inputs_return_unavailable(model):
    out = model.predict(features(log_volume_ratio_per_100d=None))
    assert out["status"] == "unavailable" and "log_volume_ratio_per_100d" in out["missing_inputs"]
    assert "probability" not in out


def test_missing_optional_inputs_are_imputed_and_reported(model):
    out = model.predict(features(age_years=None))
    assert out["status"] == "ok" and out["imputed_features"] == ["age_years"]


def test_prediction_interval_and_ood(model):
    out = model.predict(features())
    lo, hi = out["interval95"]
    assert 0 <= lo <= out["probability"] <= hi <= 1
    assert out["kind"] == "model_output" and out["requires_expert_review"]
    weird = model.predict(features(baseline_total_volume_cc=5000.0))
    assert weird["out_of_distribution_warnings"] and weird["uncertain"]


def test_higher_growth_raises_risk(model):
    assert model.predict(features(log_volume_ratio_per_100d=1.0))["probability"] > model.predict(
        features(log_volume_ratio_per_100d=-0.5))["probability"]


def test_save_load_and_tamper_detection(model, tmp_path):
    model.save(tmp_path / "m")
    loaded = ProgressionModel.load(tmp_path / "m")
    assert loaded is not None and loaded.predict(features())["probability"] == model.predict(features())["probability"]
    (tmp_path / "m" / "model.joblib").write_bytes(b"tampered")
    assert ProgressionModel.load(tmp_path / "m") is None


def test_features_from_longitudinal():
    result = {
        "timepoints": [{"tp_id": "T0", "days_from_baseline": 0, "total_lesion_volume_cc": 10.0},
                       {"tp_id": "T1", "days_from_baseline": 100, "total_lesion_volume_cc": 20.0}],
        "tracks": [{"class": "primary_tumor", "measurements": {"T0": {"sphericity": 0.8}}}],
        "new_lesion_count": 1,
    }
    f = from_longitudinal(result, {"age_years": None})
    assert f["log_volume_ratio_per_100d"] == pytest.approx(np.log(2))
    assert f["new_lesion_count"] == 1 and f["primary_sphericity"] == 0.8 and f["age_years"] is None
