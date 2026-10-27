"""The DEMO DATA cases shown in the UI. Deterministic; regenerated if missing."""

from __future__ import annotations

from datetime import datetime, timezone

from onconova.imaging.io import save_nifti
from onconova.safety import DEMO_LABEL
from onconova.store import CaseStore
from onconova.synthetic.phantom import GENERATOR_VERSION, STRUCTURE_CLASSES, random_patient_spec, render_timepoint

DEMO_CASES = [
    {"case_id": "DEMO-001", "seed": 101, "n_tp": 3, "trajectory": "growing", "node": False, "new_lesion": True,
     "title": "Growing primary; new contralateral nodule at second follow-up"},
    {"case_id": "DEMO-002", "seed": 202, "n_tp": 3, "trajectory": "responding", "node": True, "new_lesion": False,
     "title": "Shrinking primary with a node-region lesion"},
    {"case_id": "DEMO-003", "seed": 303, "n_tp": 2, "trajectory": "stable", "node": False, "new_lesion": False,
     "title": "Stable primary"},
    {"case_id": "DEMO-004", "seed": 404, "n_tp": 3, "trajectory": "growing", "node": True, "new_lesion": False,
     "title": "Growing primary with an enlarging node-region lesion"},
    {"case_id": "DEMO-005", "seed": 505, "n_tp": 1, "trajectory": "stable", "node": False, "new_lesion": False,
     "title": "Single baseline scan (longitudinal and progression analysis unavailable)"},
]
TP_LABELS = {0: "Baseline", 1: "Follow-up 1", 2: "Follow-up 2"}


def materialize_demo_cases(store: CaseStore, *, force: bool = False) -> list[str]:
    created = []
    for spec_def in DEMO_CASES:
        case_id = spec_def["case_id"]
        if store.exists(case_id) and not force:
            continue
        spec = random_patient_spec(case_id, spec_def["seed"], n_timepoints=spec_def["n_tp"],
                                   trajectory=spec_def["trajectory"], node=spec_def["node"],
                                   new_lesion=spec_def["new_lesion"])
        timepoints = []
        for i, tp in enumerate(spec.timepoints):
            image, lesions, structures = render_timepoint(spec, tp)
            tp_dir = store.case_dir(case_id) / tp.tp_id
            save_nifti(image, tp_dir / "image.nii.gz")
            save_nifti(lesions, tp_dir / "reference_lesions.nii.gz")
            save_nifti(structures, tp_dir / "structures.nii.gz")
            timepoints.append({"tp_id": tp.tp_id, "label": TP_LABELS.get(i, f"Follow-up {i}"), "days_from_baseline": tp.days,
                               "modality": "CT", "geometry": image.geometry(), "ground_truth": image.metadata["ground_truth"]})
        store.write_case({
            "case_id": case_id,
            "demo": True,
            "label": DEMO_LABEL,
            "title": spec_def["title"],
            "cancer_type": "Synthetic thoracic lesion (NSCLC-like phantom)",
            "body_region": "Thorax",
            "modality": "CT",
            "timepoints": timepoints,
            "reference": {"available": True, "kind": "synthetic generator ground truth",
                          "note": "Exact masks from the phantom generator, not an expert annotation."},
            "structures": {"available": True, "classes": {str(k): v for k, v in STRUCTURE_CLASSES.items()},
                           "kind": "synthetic generator ground truth"},
            "clinical": {"age_years": None, "note": "No clinical data exists for synthetic cases."},
            "provenance": {"generator": GENERATOR_VERSION, "seed": spec_def["seed"], "trajectory": spec.trajectory,
                           "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")},
        })
        created.append(case_id)
    return created
