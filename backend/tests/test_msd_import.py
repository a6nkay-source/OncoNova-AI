"""MSD manifest builder on a mock task folder with the real directory layout."""

import json

import numpy as np

from onconova.data.msd import build_msd_manifest
from onconova.imaging.io import save_nifti
from onconova.imaging.volume import Volume


def make_task(root, n=8, bad_label=False):
    (root / "imagesTr").mkdir(parents=True)
    (root / "labelsTr").mkdir()
    training = []
    for i in range(n):
        img = Volume(np.full((16, 20, 20), -800.0, np.float32), (0.8, 0.8, 1.5), modality="CT")
        lab = np.zeros((16, 20, 20), np.uint8)
        lab[6:10, 8:12, 8:12] = 1
        if bad_label and i == 0:
            lab[0, 0, 0] = 7
        save_nifti(img, root / "imagesTr" / f"lung_{i:03d}.nii.gz")
        save_nifti(img.with_array(lab), root / "labelsTr" / f"lung_{i:03d}.nii.gz")
        training.append({"image": f"./imagesTr/lung_{i:03d}.nii.gz", "label": f"./labelsTr/lung_{i:03d}.nii.gz"})
    (root / "dataset.json").write_text(json.dumps({
        "name": "Lung", "description": "Lung and cancer segmentation", "licence": "CC-BY-SA 4.0",
        "modality": {"0": "CT"}, "labels": {"0": "background", "1": "cancer"}, "training": training, "test": []}))


def test_msd_manifest_is_validated_and_patient_split(tmp_path):
    make_task(tmp_path / "Task06_Lung", n=10)
    m = build_msd_manifest(tmp_path / "Task06_Lung", manifest_id="msd-test")
    assert m["n_patients"] == 10 and m["modality"] == "CT" and m["split_unit"] == "patient"
    assert m["label_definitions"] == {"1": "cancer"} and not m["excluded"]
    assert all(r["lesion_voxels"] == 64 and r["spacing_xyz_mm"] == [0.8, 0.8, 1.5] for r in m["records"])
    assert sum(m["scan_counts_by_split"].values()) == 10


def test_unexpected_label_values_are_excluded(tmp_path):
    make_task(tmp_path / "T", n=4, bad_label=True)
    m = build_msd_manifest(tmp_path / "T", manifest_id="x")
    assert m["n_patients"] == 3 and m["excluded"][0]["reason"].startswith("unexpected_label_values")
