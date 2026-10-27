"""Evaluate every available segmentation model on the held-out synthetic test patients.

    python scripts/evaluate_segmentation.py

Uses only the manifest's ``test`` split (patients never seen in training or
validation). Metrics are computed per scan, averaged per patient, then
summarised across patients with a patient-level bootstrap 95% CI. Writes one
report per model to artifacts/evaluations/.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from onconova import __version__  # noqa: E402
from onconova.config import get_settings  # noqa: E402
from onconova.data.manifest import load_manifest  # noqa: E402
from onconova.imaging.io import load_nifti  # noqa: E402
from onconova.segmentation.base import LESION_CLASS_NAMES  # noqa: E402
from onconova.segmentation.measure import components  # noqa: E402
from onconova.segmentation.metrics import evaluate_pair  # noqa: E402
from onconova.segmentation.registry import segmenters  # noqa: E402

METRICS = ("dice", "iou", "hausdorff_mm", "hd95_mm", "mean_surface_distance_mm", "sensitivity", "specificity")


def lesion_detection(pred: np.ndarray, ref: np.ndarray) -> dict:
    ref_c, n_ref = components(ref)
    pred_c, n_pred = components(pred)
    detected = sum(1 for i in range(1, n_ref + 1) if pred[ref_c == i].any())
    false_pos = sum(1 for j in range(1, n_pred + 1) if not ref[pred_c == j].any())
    return {"reference_lesions": n_ref, "detected": detected, "false_positive_components": false_pos}


def summarise(per_patient: dict[str, list[float]], seed: int = 0, n_boot: int = 2000) -> dict:
    out = {}
    rng = np.random.default_rng(seed)
    for metric, values in per_patient.items():
        v = np.array([x for x in values if x is not None and np.isfinite(x)], dtype=float)
        if v.size == 0:
            out[metric] = {"n_patients": 0, "mean": None, "median": None, "ci95": [None, None]}
            continue
        boots = [rng.choice(v, v.size).mean() for _ in range(n_boot)]
        out[metric] = {"n_patients": int(v.size), "mean": round(float(v.mean()), 4), "median": round(float(np.median(v)), 4),
                       "std": round(float(v.std()), 4), "ci95": [round(float(np.quantile(boots, 0.025)), 4),
                                                                 round(float(np.quantile(boots, 0.975)), 4)]}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-id", default="synthetic-seg-v1")
    args = parser.parse_args()
    settings = get_settings()
    root = settings.data_dir / "synthetic" / args.manifest_id
    manifest = load_manifest(root / "manifest.json")
    test = [r for r in manifest["records"] if r["split"] == "test"]
    print(f"test scans={len(test)} patients={len({r['patient_id'] for r in test})}")

    for model_id, model in segmenters(settings.models_dir).items():
        if model.card.status != "available":
            print(f"skip {model_id}: {model.card.unavailable_reason}")
            continue
        per_scan = []
        confusion = np.zeros((4, 4), dtype=np.int64)  # rows: reference class, cols: predicted class (voxels)
        t0 = time.time()
        for r in test:
            vol = load_nifti(root / r["image"], modality="CT")
            vol.metadata["demo"] = True
            ref = load_nifti(root / r["lesions"], is_label=True).array
            body = load_nifti(root / r["structures"], is_label=True).array > 0
            res = model.predict(vol, mc_samples=0, saliency=False)
            np.add.at(confusion, (ref.astype(np.int64).ravel(), res.labels.astype(np.int64).ravel()), 1)
            entry = {"patient_id": r["patient_id"], "scan_id": r["scan_id"],
                     "any_lesion": evaluate_pair(res.labels > 0, ref > 0, vol.spacing_zyx, body),
                     "detection": lesion_detection(res.labels > 0, ref > 0)}
            for k, name in LESION_CLASS_NAMES.items():
                if (ref == k).any():  # class-wise metrics only where the class exists in the reference
                    entry[name] = evaluate_pair(res.labels == k, ref == k, vol.spacing_zyx, body)
            per_scan.append(entry)
        runtime = time.time() - t0

        results = {}
        for target in ("any_lesion", *LESION_CLASS_NAMES.values()):
            by_patient: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
            for e in per_scan:
                if target in e:
                    for m in METRICS:
                        by_patient[e["patient_id"]][m].append(e[target][m])
            if not by_patient:
                continue
            patient_means = {m: [np.mean([x for x in d[m] if x is not None]) if any(x is not None for x in d[m]) else None
                                 for d in by_patient.values()] for m in METRICS}
            results[target] = summarise(patient_means) | {"n_scans": sum(1 for e in per_scan if target in e)}
        det = [e["detection"] for e in per_scan]
        n_ref = sum(d["reference_lesions"] for d in det)
        results["lesion_detection"] = {
            "reference_lesions": n_ref,
            "detected": sum(d["detected"] for d in det),
            "lesion_sensitivity": round(sum(d["detected"] for d in det) / n_ref, 4) if n_ref else None,
            "false_positive_components_per_scan": round(sum(d["false_positive_components"] for d in det) / len(det), 4),
        }
        names = ["background", *LESION_CLASS_NAMES.values()]
        results["class_confusion_voxels"] = {
            "rows_reference_cols_predicted": names,
            "matrix": confusion.tolist(),
            "note": "Voxel counts over all test scans. Off-diagonal entries show missed or mislabelled lesion voxels.",
        }
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        report = {
            "report_id": f"segmentation-{model_id}-{stamp}",
            "task": "segmentation",
            "model": {"id": model_id, "version": model.card.version, "kind": model.card.kind,
                      "checkpoint_sha256": model.card.checkpoint_sha256, "label": model.card.label},
            "dataset": {"id": manifest["manifest_id"], "label": manifest["label"], "generator": manifest["generator"],
                        "split": "test", "n_patients": len({r["patient_id"] for r in test}), "n_scans": len(test),
                        "patient_counts_by_split": manifest["patient_counts_by_split"]},
            "protocol": ("Patient-separated test split from the manifest. Per-scan metrics against generator ground truth, "
                         "averaged per patient; mean, median and 2000-sample patient bootstrap 95% CI across patients. "
                         "Specificity is computed inside the body mask. Class metrics only on scans where the class exists."),
            "results": results,
            "per_scan": per_scan,
            "runtime_s": round(runtime, 1),
            "onconova_version": __version__,
            "caveat": "Synthetic phantoms with exact ground truth. These numbers do not describe performance on patients.",
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        out = settings.evaluations_dir / f"{report['report_id']}.json"
        out.write_text(json.dumps(report, indent=2))
        a = results["any_lesion"]
        print(f"{model_id:16s} any-lesion Dice {a['dice']['mean']} {a['dice']['ci95']}  HD95 {a['hd95_mm']['mean']} mm  "
              f"lesion sens {results['lesion_detection']['lesion_sensitivity']}  -> {out.name}")


if __name__ == "__main__":
    main()
