"""Patient-level splitting and leakage checks."""

from collections import defaultdict
from datetime import date

import pytest

from onconova.data.manifest import build_synthetic_segmentation_dataset
from onconova.data.splits import LeakageError, check_no_leakage, patient_level_split, pseudonymous_id, temporal_split


def test_patient_split_is_disjoint_and_deterministic():
    patients = [f"P{i}" for i in range(100)]
    a = patient_level_split(patients, seed=1)
    assert a == patient_level_split(patients, seed=1)
    assert a != patient_level_split(patients, seed=2)
    counts = defaultdict(int)
    for s in a.values():
        counts[s] += 1
    assert counts == {"train": 70, "val": 15, "test": 15}


def test_all_scans_of_a_patient_share_a_split():
    records = [{"patient_id": f"P{i // 3}", "scan": i} for i in range(60)]  # 3 scans each
    split_of = patient_level_split([r["patient_id"] for r in records], seed=0)
    for r in records:
        r["split"] = split_of[r["patient_id"]]
    check_no_leakage(records)


def test_leakage_is_detected():
    with pytest.raises(LeakageError):
        check_no_leakage([{"patient_id": "A", "split": "train"}, {"patient_id": "A", "split": "test"}])


def test_temporal_split_uses_first_scan_date():
    records = [
        {"patient_id": "early", "date": "2019-05-01"},
        {"patient_id": "early", "date": "2022-01-01"},  # later scan of an early patient stays in train
        {"patient_id": "late", "date": "2021-06-01"},
    ]
    assert temporal_split(records, date_key="date", cutoff=date(2021, 1, 1)) == {"early": "train", "late": "test"}


def test_pseudonymous_ids_hide_source():
    pid = pseudonymous_id("MRN-123", "salt")
    assert "123" not in pid and pid == pseudonymous_id("MRN-123", "salt") != pseudonymous_id("MRN-123", "other")


def test_synthetic_manifest_has_no_leakage(tmp_path):
    m = build_synthetic_segmentation_dataset(tmp_path / "ds", n_patients=6, timepoints_per_patient=2, seed=5)
    assert m["split_unit"] == "patient" and m["label"] == "DEMO DATA"
    check_no_leakage(m["records"])
    assert len(m["records"]) == 12
    assert all(not any(q.startswith("error") for q in r["quality"]) for r in m["records"])
