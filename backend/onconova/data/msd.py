"""Build a validated, patient-level manifest for a Medical Segmentation Decathlon task.

Works on an already-downloaded, extracted task folder (``dataset.json``,
``imagesTr/``, ``labelsTr/``). Nothing is downloaded here. MSD training cases have
one scan per patient, so a case-level split is a patient-level split within MSD;
it can still overlap with other TCIA lung collections, which matters only if you
combine datasets.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from onconova.data.splits import check_no_leakage, patient_level_split
from onconova.errors import ImagingError
from onconova.imaging.io import load_nifti

# MSD label value -> OncoNova lesion class. Task06 has {0: background, 1: cancer}.
LABEL_MAP = {"cancer": 1, "tumour": 1, "tumor": 1}


def build_msd_manifest(task_dir: Path, *, manifest_id: str, split_seed: int = 42,
                       check_images: bool = True) -> dict[str, Any]:
    spec_path = task_dir / "dataset.json"
    if not spec_path.exists():
        raise FileNotFoundError(f"{spec_path} not found; point at the extracted MSD task folder.")
    spec = json.loads(spec_path.read_text())
    modality = {"CT": "CT", "MRI": "MR", "MR": "MR"}.get(str(spec.get("modality", {}).get("0", "")).upper(), "UNKNOWN")
    labels = {int(k): v for k, v in spec.get("labels", {}).items()}
    mapping = {k: LABEL_MAP.get(v.lower(), None) for k, v in labels.items() if k != 0}

    pairs = spec.get("training", [])
    patients = [Path(p["image"]).name.split(".")[0] for p in pairs]
    split_of = patient_level_split(patients, seed=split_seed)
    records, excluded = [], []
    for pid, pair in zip(patients, pairs):
        image, label = (task_dir / pair["image"]).resolve(), (task_dir / pair["label"]).resolve()
        record = {"patient_id": pid, "scan_id": pid, "split": split_of[pid],
                  "image": str(image.relative_to(task_dir.resolve())), "lesions": str(label.relative_to(task_dir.resolve())),
                  "quality": []}
        if check_images:
            try:
                vol = load_nifti(image, modality=modality)
                lab = load_nifti(label, is_label=True)
            except ImagingError as exc:
                excluded.append({"patient_id": pid, "reason": exc.code})
                continue
            if lab.shape != vol.shape or not np.allclose(lab.spacing, vol.spacing, atol=1e-3):
                excluded.append({"patient_id": pid, "reason": "label_geometry_mismatch"})
                continue
            values = set(np.unique(lab.array).tolist())
            if not values <= set(labels):
                excluded.append({"patient_id": pid, "reason": f"unexpected_label_values {sorted(values - set(labels))}"})
                continue
            record.update({"shape_zyx": list(vol.shape), "spacing_xyz_mm": [round(s, 4) for s in vol.spacing],
                           "lesion_voxels": int((lab.array > 0).sum())})
            record["quality"] = [f"warning: {w}" for w in vol.metadata.get("warnings", [])]
            if record["lesion_voxels"] == 0:
                record["quality"].append("warning: no labelled lesion voxels")
        records.append(record)
    counts = check_no_leakage(records)
    return {
        "manifest_id": manifest_id,
        "label": "PUBLIC RESEARCH DATA",
        "description": f"Medical Segmentation Decathlon {spec.get('name', '')} ({spec.get('description', '')}).",
        "source": {"dataset": spec.get("name"), "licence": spec.get("licence") or spec.get("license"),
                   "reference": spec.get("reference"), "release": spec.get("release")},
        "root": str(task_dir.resolve()),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "modality": modality,
        "split_unit": "patient",
        "split_seed": split_seed,
        "n_patients": len(records),
        "scan_counts_by_split": counts,
        "patient_counts_by_split": counts,
        "label_definitions": {str(v): labels[k] for k, v in mapping.items() if v is not None},
        "label_mapping_from_source": {str(k): v for k, v in mapping.items()},
        "ground_truth": "Manual tumour segmentation from the MSD release (single rater); not pathology-confirmed.",
        "excluded": excluded,
        "records": records,
    }
