"""Build reproducible synthetic datasets with manifests and quality checks."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from onconova import __version__
from onconova.data.splits import check_no_leakage, patient_level_split
from onconova.imaging.io import save_nifti, validate_volume
from onconova.synthetic.phantom import GENERATOR_VERSION, random_patient_spec, render_timepoint
from onconova.safety import DEMO_LABEL

SYNTHETIC_SEED_OFFSET = 10_000  # keeps training seeds disjoint from the demo cases (<1000)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quality_checks(image, lesions, structures) -> list[str]:
    problems = []
    warnings = validate_volume(image)
    problems += [f"warning: {w}" for w in warnings]
    if not (lesions.array == 1).any():
        problems.append("error: no primary lesion voxels")
    if lesions.shape != image.shape or structures.shape != image.shape:
        problems.append("error: label geometry differs from image")
    return problems


def build_synthetic_segmentation_dataset(out_dir: Path, *, n_patients: int, seed: int = 2026,
                                         timepoints_per_patient: int = 2, split_seed: int = 42,
                                         manifest_id: str = "synthetic-seg-v1") -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    patient_seeds = (SYNTHETIC_SEED_OFFSET + rng.permutation(90_000)[:n_patients]).tolist()
    patients = [f"SYN-{i:04d}" for i in range(n_patients)]
    split_of = patient_level_split(patients, seed=split_seed)

    records = []
    for pid, pseed in zip(patients, patient_seeds):
        spec = random_patient_spec(pid, int(pseed), n_timepoints=timepoints_per_patient)
        for tp in spec.timepoints:
            image, lesions, structures = render_timepoint(spec, tp)
            scan_dir = out_dir / pid / tp.tp_id
            paths = {"image": scan_dir / "image.nii.gz", "lesions": scan_dir / "lesions.nii.gz",
                     "structures": scan_dir / "structures.nii.gz"}
            save_nifti(image, paths["image"])
            save_nifti(lesions, paths["lesions"])
            save_nifti(structures, paths["structures"])
            records.append({
                "patient_id": pid,
                "scan_id": f"{pid}-{tp.tp_id}",
                "timepoint": tp.tp_id,
                "days_from_baseline": tp.days,
                "split": split_of[pid],
                "seed": int(pseed),
                "trajectory": spec.trajectory,
                "image": str(paths["image"].relative_to(out_dir)),
                "lesions": str(paths["lesions"].relative_to(out_dir)),
                "structures": str(paths["structures"].relative_to(out_dir)),
                "image_sha256": file_sha256(paths["image"]),
                "quality": quality_checks(image, lesions, structures),
            })
    counts = check_no_leakage(records)
    manifest = {
        "manifest_id": manifest_id,
        "label": DEMO_LABEL,
        "description": "Synthetic thoracic CT phantoms with generator ground truth. Not patient data.",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "onconova_version": __version__,
        "generator": GENERATOR_VERSION,
        "dataset_seed": seed,
        "split_seed": split_seed,
        "split_unit": "patient",
        "n_patients": n_patients,
        "scan_counts_by_split": counts,
        "patient_counts_by_split": {s: sum(1 for v in split_of.values() if v == s) for s in ("train", "val", "test")},
        "label_definitions": {"1": "primary_tumor", "2": "lymph_node_region_lesion", "3": "other_lesion"},
        "records": records,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text())
    check_no_leakage(manifest["records"])
    return manifest
