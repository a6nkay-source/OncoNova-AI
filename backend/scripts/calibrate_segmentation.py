"""Does test-time augmentation or MC dropout make a U-Net's probabilities better calibrated?

    python scripts/calibrate_segmentation.py --dataset msd-task06-lung --split val --checkpoint-dir ../artifacts/models/seg-unet-msd-ens

For each configuration (one pass, 8-mirror TTA, MC dropout x8) it records Dice and the
calibration of the voxel foreground probability: expected calibration error (ECE,
15 equal-width bins) and Brier score, pooled over voxels of all patients, in two
regions fixed per patient and identical for every configuration:

- ``near``: within 20 mm of the reference tumour, where the boundary is decided;
- ``body``: the whole body, which also counts confident false positives far away.

A configuration is recommended for a confidence overlay only if it lowers the near-
tumour ECE without lowering mean Dice by more than 0.01. Writes
artifacts/evaluations/calibration-<model>-<dataset>-<split>-<stamp>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from onconova import __version__  # noqa: E402
from onconova.config import get_settings  # noqa: E402
from onconova.data.manifest import load_manifest  # noqa: E402
from onconova.imaging.io import load_nifti  # noqa: E402
from onconova.segmentation.metrics import dice  # noqa: E402
from onconova.segmentation.unet import UNetSegmenter  # noqa: E402

from evaluate_segmentation import body_mask  # noqa: E402

BINS = 15
NEAR_MM = 20.0
CONFIGS = {"single pass": {"mc_samples": 0}, "mirror TTA x8": {"mc_samples": 0, "tta": True},
           "MC dropout x8": {"mc_samples": 8}}


def near_region(ref: np.ndarray, spacing_zyx, mm: float) -> np.ndarray:
    """Voxels within ``mm`` of the reference mask (computed on its bounding box plus the margin)."""
    out = np.zeros(ref.shape, bool)
    if not ref.any():
        return out
    pad = np.ceil(mm / np.asarray(spacing_zyx)).astype(int) + 1
    pts = np.argwhere(ref)
    a, b = np.maximum(pts.min(0) - pad, 0), np.minimum(pts.max(0) + pad + 1, ref.shape)
    sl = tuple(slice(int(i), int(j)) for i, j in zip(a, b))
    out[sl] = ndimage.distance_transform_edt(~ref[sl], sampling=spacing_zyx) <= mm
    return out


class Calibration:
    """Pooled reliability counts: per bin, voxels, summed probability and summed outcome."""

    def __init__(self) -> None:
        self.n = np.zeros(BINS)
        self.p = np.zeros(BINS)
        self.y = np.zeros(BINS)
        self.sq = 0.0

    def add(self, prob: np.ndarray, truth: np.ndarray) -> None:
        idx = np.minimum((prob * BINS).astype(int), BINS - 1)
        self.n += np.bincount(idx, minlength=BINS)
        self.p += np.bincount(idx, weights=prob, minlength=BINS)
        self.y += np.bincount(idx, weights=truth.astype(float), minlength=BINS)
        self.sq += float(((prob - truth) ** 2).sum())

    def summary(self) -> dict:
        total = self.n.sum()
        nz = self.n > 0
        conf, acc = self.p[nz] / self.n[nz], self.y[nz] / self.n[nz]
        return {"voxels": int(total), "ece": round(float((self.n[nz] / total * np.abs(conf - acc)).sum()), 5),
                "brier": round(self.sq / total, 6),
                "reliability": [{"bin": [round(i / BINS, 3), round((i + 1) / BINS, 3)], "voxels": int(self.n[i]),
                                 "mean_probability": round(float(self.p[i] / self.n[i]), 4) if self.n[i] else None,
                                 "fraction_tumour": round(float(self.y[i] / self.n[i]), 4) if self.n[i] else None}
                                for i in range(BINS)]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="msd-task06-lung")
    parser.add_argument("--split", default="val")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.split == "test":
        sys.exit("The test split is locked; calibrate on val.")
    settings = get_settings()
    manifest = load_manifest(settings.data_dir / "external" / args.dataset / "manifest.json")
    root = Path(manifest["root"])
    records = [r for r in manifest["records"] if r["split"] == args.split]
    model = UNetSegmenter(args.checkpoint_dir, model_id=args.checkpoint_dir.name)
    if model.card.status != "available":
        sys.exit(model.card.unavailable_reason)

    stats = {name: {"near": Calibration(), "body": Calibration(), "dice": [], "seconds": []} for name in CONFIGS}
    for r in records:
        vol = load_nifti(root / r["image"], modality="CT")
        ref = load_nifti(root / r["lesions"], is_label=True).array > 0
        regions = {"near": near_region(ref, vol.spacing_zyx, NEAR_MM), "body": body_mask(vol.array)}
        for name, kwargs in CONFIGS.items():
            t0 = time.time()
            res = model.predict(vol, saliency=False, **kwargs)
            s = stats[name]
            s["seconds"].append(time.time() - t0)
            s["dice"].append(dice(res.labels > 0, ref))
            for region, mask in regions.items():
                s[region].add(res.foreground_probability[mask].astype(np.float64), ref[mask])
            print(f"  {r['scan_id']} {name:14s} Dice {s['dice'][-1]:.3f}  {s['seconds'][-1]:.0f} s", flush=True)

    results = {name: {"dice_mean": round(float(np.mean(s["dice"])), 4), "dice_per_patient": [round(v, 4) for v in s["dice"]],
                      "seconds_per_scan": round(float(np.mean(s["seconds"])), 1),
                      "near_tumour": s["near"].summary(), "body": s["body"].summary()}
               for name, s in stats.items()}
    base = results["single pass"]
    verdicts = {}
    for name, res in results.items():
        if name == "single pass":
            continue
        better = res["near_tumour"]["ece"] < base["near_tumour"]["ece"]
        keeps = res["dice_mean"] >= base["dice_mean"] - 0.01
        verdicts[name] = {"near_ece_change": round(res["near_tumour"]["ece"] - base["near_tumour"]["ece"], 5),
                          "body_ece_change": round(res["body"]["ece"] - base["body"]["ece"], 5),
                          "dice_change": round(res["dice_mean"] - base["dice_mean"], 4),
                          "recommended": bool(better and keeps)}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = {
        "report_id": f"calibration-{model.card.id}-{args.dataset}-{args.split}-{stamp}",
        "task": "segmentation_calibration",
        "model": {"id": model.card.id, "checkpoint_sha256": model.card.checkpoint_sha256},
        "dataset": {"id": args.dataset, "split": args.split, "n_patients": len(records)},
        "protocol": (f"Foreground probability (1 - p(background)) on the scan grid. ECE with {BINS} equal-width bins and "
                     f"Brier score pooled over voxels of all {len(records)} patients, within {NEAR_MM:.0f} mm of the "
                     "reference tumour ('near') and inside the body (largest region above -500 HU). Recommended only if "
                     "the near-tumour ECE drops and mean Dice falls by at most 0.01."),
        "results": results,
        "verdicts": verdicts,
        "onconova_version": __version__,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out = settings.evaluations_dir / f"{report['report_id']}.json"
    out.write_text(json.dumps(report, indent=2))
    for name, res in results.items():
        print(f"{name:14s} Dice {res['dice_mean']:.3f}  near ECE {res['near_tumour']['ece']:.4f}  body ECE "
              f"{res['body']['ece']:.5f}  {res['seconds_per_scan']} s/scan  {verdicts.get(name, '')}")
    print("->", out.name)


if __name__ == "__main__":
    main()
