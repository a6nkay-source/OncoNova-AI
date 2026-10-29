"""Evaluate segmentation models on a patient-separated split of a dataset manifest.

    python scripts/evaluate_segmentation.py                                   # synthetic test split
    python scripts/evaluate_segmentation.py --dataset msd-task06-lung --split val --model demo-unet

Metrics are computed per scan, averaged per patient, then summarised across
patients with a patient-level bootstrap 95% CI. Each scan's inference time is
recorded, and the process's peak memory (run one model per process to attribute
it to that model). Writes one report per model to artifacts/evaluations/.

Synthetic data has exact generator truth and lesion classes. External (real)
datasets are evaluated as "any lesion" against their released masks, and their
``test`` split is locked: it is refused unless ``--unlock-test`` is passed, so
model choices are made on ``val``.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from onconova import __version__  # noqa: E402
from onconova.config import get_settings  # noqa: E402
from onconova.data.manifest import load_manifest  # noqa: E402
from onconova.imaging.io import load_nifti  # noqa: E402
from onconova.segmentation.base import LESION_CLASS_NAMES  # noqa: E402
from onconova.segmentation.measure import components  # noqa: E402
from onconova.segmentation.metrics import evaluate_pair  # noqa: E402
from onconova.segmentation.registry import segmenters  # noqa: E402
from onconova.segmentation.unet import UNetSegmenter  # noqa: E402

METRICS = ("dice", "iou", "hausdorff_mm", "hd95_mm", "mean_surface_distance_mm", "sensitivity", "specificity")


def lesion_detection(pred: np.ndarray, ref: np.ndarray) -> dict:
    ref_c, n_ref = components(ref)
    pred_c, n_pred = components(pred)
    detected = sum(1 for i in range(1, n_ref + 1) if pred[ref_c == i].any())
    false_pos = sum(1 for j in range(1, n_pred + 1) if not ref[pred_c == j].any())
    return {"reference_lesions": n_ref, "detected": detected, "false_positive_components": false_pos}


def body_mask(hu: np.ndarray) -> np.ndarray:
    """Largest connected region above -500 HU with holes filled: the ROI for specificity on real CT."""
    body = hu > -500
    comps, n = ndimage.label(body)
    if n:
        body = comps == (int(np.argmax(ndimage.sum(body, comps, range(1, n + 1)))) + 1)
    return ndimage.binary_fill_holes(body)


def peak_rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(peak / 1e6 if sys.platform == "darwin" else peak / 1e3, 1)  # bytes on macOS, KiB on Linux


def accelerator_mb() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 1e6, 1)
        if torch.backends.mps.is_available():
            return round(torch.mps.driver_allocated_memory() / 1e6, 1)
    except ImportError:
        pass
    return None


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
    parser.add_argument("--dataset", default="synthetic-seg-v1", help="manifest id under data/synthetic or data/external")
    parser.add_argument("--manifest-id", dest="dataset", help=argparse.SUPPRESS)  # old name
    parser.add_argument("--split", default=None, help="default: test for synthetic data, val for external data")
    parser.add_argument("--model", default=None, help="evaluate one model (default: every available model)")
    parser.add_argument("--checkpoint-dir", type=Path, default=None, help="evaluate an unregistered U-Net checkpoint folder")
    parser.add_argument("--medsam", action="store_true", help="with --checkpoint-dir: refine its detections with MedSAM")
    parser.add_argument("--unlock-test", action="store_true", help="allow the locked test split of an external dataset")
    args = parser.parse_args()
    settings = get_settings()
    synthetic_root = settings.data_dir / "synthetic" / args.dataset
    external = not (synthetic_root / "manifest.json").exists()
    manifest_path = (settings.data_dir / "external" / args.dataset if external else synthetic_root) / "manifest.json"
    manifest = load_manifest(manifest_path)
    root = Path(manifest["root"]) if external else synthetic_root
    split = args.split or ("val" if external else "test")
    if external and split == "test" and not args.unlock_test:
        sys.exit("The external test split is locked; choose models on --split val (or pass --unlock-test once, at the end).")
    records = [r for r in manifest["records"] if r["split"] == split]
    print(f"{args.dataset} {split}: scans={len(records)} patients={len({r['patient_id'] for r in records})}")

    models = segmenters(settings.models_dir)
    if args.checkpoint_dir:
        unet = UNetSegmenter(args.checkpoint_dir, model_id=args.checkpoint_dir.name)
        models = {unet.card.id: unet}
        if args.medsam:
            from onconova.segmentation.medsam import MedSAMRefiner

            refiner = MedSAMRefiner(settings.models_dir / "medsam-vit-base", unet, model_id=f"{unet.card.id}-medsam")
            models = {refiner.card.id: refiner}
    elif args.model:
        models = {args.model: models[args.model]}
    for model_id, model in models.items():
        if model.card.status != "available":
            print(f"skip {model_id}: {model.card.unavailable_reason}")
            continue
        per_scan = []
        confusion = np.zeros((4, 4), dtype=np.int64)  # rows: reference class, cols: predicted class (voxels)
        accel_peak = None
        t0 = time.time()
        for r in records:
            vol = load_nifti(root / r["image"], modality="CT")
            vol.metadata["demo"] = not external
            ref = load_nifti(root / r["lesions"], is_label=True).array
            body = (load_nifti(root / r["structures"], is_label=True).array > 0) if "structures" in r else body_mask(vol.array)
            t_case = time.time()
            res = model.predict(vol, mc_samples=0, saliency=False)
            runtime = time.time() - t_case
            mem = accelerator_mb()
            accel_peak = mem if accel_peak is None or (mem or 0) > accel_peak else accel_peak
            entry = {"patient_id": r["patient_id"], "scan_id": r["scan_id"], "runtime_s": round(runtime, 2),
                     "shape_zyx": list(vol.shape), "spacing_xyz_mm": [round(v, 3) for v in vol.spacing],
                     "any_lesion": evaluate_pair(res.labels > 0, ref > 0, vol.spacing_zyx, body),
                     "detection": lesion_detection(res.labels > 0, ref > 0)}
            if not external:
                np.add.at(confusion, (ref.astype(np.int64).ravel(), res.labels.astype(np.int64).ravel()), 1)
                for k, name in LESION_CLASS_NAMES.items():
                    if (ref == k).any():  # class-wise metrics only where the class exists in the reference
                        entry[name] = evaluate_pair(res.labels == k, ref == k, vol.spacing_zyx, body)
            per_scan.append(entry)
            print(f"  {r['scan_id']}: Dice {entry['any_lesion']['dice']:.3f}  {runtime:.1f} s", flush=True)
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
        times = [e["runtime_s"] for e in per_scan]
        results["resources"] = {
            "inference_s_per_scan": {"mean": round(float(np.mean(times)), 2), "median": round(float(np.median(times)), 2),
                                     "max": round(float(np.max(times)), 2)},
            "peak_process_rss_mb": peak_rss_mb(),
            "accelerator_mb": accel_peak,
            "device": getattr(model, "device", "cpu"),
            "note": "Inference only (loading and metrics excluded). Peak RSS covers the whole evaluation process.",
        }
        if not external:
            names = ["background", *LESION_CLASS_NAMES.values()]
            results["class_confusion_voxels"] = {
                "rows_reference_cols_predicted": names,
                "matrix": confusion.tolist(),
                "note": "Voxel counts over all test scans. Off-diagonal entries show missed or mislabelled lesion voxels.",
            }
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        caveat = ("Synthetic phantoms with exact ground truth. These numbers do not describe performance on patients."
                  if not external else
                  f"Real CT, {manifest['label']}, against single-rater manual masks ({manifest.get('ground_truth', '')}). "
                  f"{len({r['patient_id'] for r in records})} patients: confidence intervals are wide.")
        report = {
            "report_id": f"segmentation-{model_id}-{args.dataset}-{split}-{stamp}" if external else f"segmentation-{model_id}-{stamp}",
            "task": "segmentation",
            "model": {"id": model_id, "version": model.card.version, "kind": model.card.kind,
                      "checkpoint_sha256": model.card.checkpoint_sha256, "label": model.card.label},
            "dataset": {"id": manifest["manifest_id"], "label": manifest["label"], "generator": manifest.get("generator"),
                        "source": manifest.get("source"), "split": split,
                        "n_patients": len({r["patient_id"] for r in records}), "n_scans": len(records),
                        "patient_counts_by_split": manifest["patient_counts_by_split"]},
            "protocol": (f"Patient-separated {split} split from the manifest. Per-scan metrics against "
                         f"{'generator ground truth' if not external else 'the released manual masks'}, averaged per patient; "
                         "mean, median and 2000-sample patient bootstrap 95% CI across patients. Specificity is computed "
                         f"inside the body mask{'' if not external else ' (largest region above -500 HU, holes filled)'}. "
                         + ("Class metrics only on scans where the class exists." if not external else
                            "Any-lesion only: the masks have one tumour label. One deterministic pass (no MC dropout).")),
            "results": results,
            "per_scan": per_scan,
            "runtime_s": round(runtime, 1),
            "onconova_version": __version__,
            "caveat": caveat,
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        out = settings.evaluations_dir / f"{report['report_id']}.json"
        out.write_text(json.dumps(report, indent=2))
        a = results["any_lesion"]
        print(f"{model_id:16s} any-lesion Dice {a['dice']['mean']} {a['dice']['ci95']}  HD95 {a['hd95_mm']['mean']} mm  "
              f"lesion sens {results['lesion_detection']['lesion_sensitivity']}  {results['resources']['inference_s_per_scan']['mean']} s/scan "
              f"peak RSS {results['resources']['peak_process_rss_mb']} MB  -> {out.name}")


if __name__ == "__main__":
    main()
