"""Write docs/SEGMENTATION_COMPARISON.md: baseline vs candidates on real CT, from stored reports only.

    python scripts/build_segmentation_comparison.py [--dataset msd-task06-lung] [--split val]

Reads the newest artifacts/evaluations/segmentation-*-<dataset>-<split>-*.json per model
(written by scripts/evaluate_segmentation.py) and calibration-*.json (written by
scripts/calibrate_segmentation.py). Nothing here is typed in by hand: a model without
a report is listed as not evaluated. Differences against the baseline are paired per
patient, with a 2000-sample patient bootstrap 95% CI.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from onconova.config import get_settings  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
BASELINES = ("demo-classical", "demo-unet")  # the models OncoNova had before Stage 3
PENDING = {
    "nnunet-msd": "nnU-Net v2 `3d_fullres`, 100 epochs, trained on the same 37 fit patients in Colab "
                  "(notebooks/nnunet_msd_task06_colab.ipynb). Not evaluated until its checkpoint is in artifacts/models/nnunet-msd/.",
}


def f(v, d=3):
    return "n/a" if v is None else f"{v:.{d}f}"


def newest(paths: list[Path]) -> dict | None:
    reports = [json.loads(p.read_text()) for p in paths]
    return max(reports, key=lambda r: r["created_utc"]) if reports else None


def per_patient(report: dict, metric: str = "dice") -> dict[str, float]:
    out: dict[str, list[float]] = {}
    for e in report["per_scan"]:
        v = e["any_lesion"][metric]
        if v is not None:
            out.setdefault(e["patient_id"], []).append(v)
    return {k: float(np.mean(v)) for k, v in out.items()}


def paired_diff(a: dict[str, float], b: dict[str, float], seed: int = 0) -> tuple[float, float, float, int]:
    ids = sorted(set(a) & set(b))
    d = np.array([a[i] - b[i] for i in ids])
    rng = np.random.default_rng(seed)
    boots = [rng.choice(d, d.size).mean() for _ in range(2000)]
    return float(d.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975)), int((d > 0).sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="msd-task06-lung")
    parser.add_argument("--split", default="val")
    args = parser.parse_args()
    ev = get_settings().evaluations_dir
    by_model: dict[str, dict] = {}
    for p in ev.glob(f"segmentation-*-{args.dataset}-{args.split}-*.json"):
        r = json.loads(p.read_text())
        mid = r["model"]["id"]
        if mid not in by_model or r["created_utc"] > by_model[mid]["created_utc"]:
            by_model[mid] = r
    if not by_model:
        sys.exit("No reports for this dataset and split; run scripts/evaluate_segmentation.py first.")
    order = [m for m in BASELINES if m in by_model] + sorted(m for m in by_model if m not in BASELINES)
    any_report = by_model[order[0]]
    baseline = max((m for m in BASELINES if m in by_model), key=lambda m: by_model[m]["results"]["any_lesion"]["dice"]["mean"])

    lines = [
        "# Segmentation: baseline vs candidates on real CT",
        "",
        f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} by `backend/scripts/build_segmentation_comparison.py` "
        "from the stored evaluation reports. Do not edit by hand.",
        "",
        "OncoNova AI is an educational research prototype, not a clinically validated diagnostic system. These are "
        "research measurements on a public dataset, not evidence of clinical performance.",
        "",
        f"**Data:** `{args.dataset}`, **{args.split}** split: {any_report['dataset']['n_patients']} patients, one CT each. "
        f"{any_report['dataset'].get('source', {}).get('dataset') or ''} ({any_report['dataset'].get('source', {}).get('licence') or 'licence n/a'}). "
        f"The split is by patient: {any_report['dataset']['patient_counts_by_split']}. Candidates were trained on (part of) the train "
        "patients and chose their checkpoints on an inner split of them, so these validation patients were unseen by training and "
        "selection. The test patients stay locked for the final evaluation.",
        "",
        f"**Reference:** the released single-rater tumour masks (not pathology-confirmed). **Protocol:** {any_report['protocol']}",
        "",
        f"With {any_report['dataset']['n_patients']} patients the intervals are wide; a difference whose interval spans 0 is not shown "
        "to be real.",
        "",
        "## Results",
        "",
        "| Model | Dice mean (95% CI) | Dice median | IoU mean | HD95 mm median | Tumours found | False regions / scan | s / scan | Peak RAM MB |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for m in order:
        r = by_model[m]["results"]
        a, det, res = r["any_lesion"], r["lesion_detection"], r.get("resources", {})
        lines.append(
            f"| `{m}`{' (baseline)' if m in BASELINES else ''} | {f(a['dice']['mean'])} ({f(a['dice']['ci95'][0])}-{f(a['dice']['ci95'][1])}) | "
            f"{f(a['dice']['median'])} | {f(a['iou']['mean'])} | {f(a['hd95_mm']['median'], 1)} | {det['detected']}/{det['reference_lesions']} | "
            f"{f(det['false_positive_components_per_scan'], 1)} | {f(res.get('inference_s_per_scan', {}).get('mean'), 1)} | "
            f"{f(res.get('peak_process_rss_mb'), 0)} |")
    for m, why in PENDING.items():
        if m not in by_model:
            lines.append(f"| `{m}` | not evaluated | | | | | | | |")
    lines += ["", "Dice and IoU: overlap with the reference (1 is perfect). HD95: 95th-percentile surface distance (lower is better; "
              "undefined when a model finds nothing). Tumours found: reference tumour regions that the prediction overlaps at all. "
              "False regions: predicted regions touching no reference tumour. Timing is inference only, on this Mac "
              f"({by_model[order[-1]]['results'].get('resources', {}).get('device', 'n/a')}); peak RAM is the whole evaluation process.", ""]

    lines += [f"## Paired difference in Dice against the best baseline (`{baseline}`)", "",
              "| Model | Mean difference (95% CI) | Patients improved |", "|---|---|---|"]
    base = per_patient(by_model[baseline])
    for m in order:
        if m == baseline:
            continue
        mean, lo, hi, wins = paired_diff(per_patient(by_model[m]), base)
        lines.append(f"| `{m}` | {mean:+.3f} ({lo:+.3f} to {hi:+.3f}) | {wins}/{len(base)} |")
    lines.append("")

    lines += ["## Per patient (Dice)", "", "| Patient | " + " | ".join(f"`{m}`" for m in order) + " |",
              "|---|" + "---|" * len(order)]
    tables = {m: per_patient(by_model[m]) for m in order}
    for pid in sorted(base):
        lines.append(f"| {pid} | " + " | ".join(f(tables[m].get(pid)) for m in order) + " |")
    lines.append("")

    cal = newest(list(ev.glob(f"calibration-*-{args.dataset}-{args.split}-*.json")))
    lines += ["## Calibration of the foreground probability", ""]
    if cal:
        lines += [f"Model `{cal['model']['id']}`. {cal['protocol']}", "",
                  "| Configuration | Dice mean | ECE near tumour | Brier near tumour | ECE body | s / scan | Recommended for a confidence overlay |",
                  "|---|---|---|---|---|---|---|"]
        for name, r in cal["results"].items():
            v = cal["verdicts"].get(name)
            lines.append(f"| {name} | {f(r['dice_mean'])} | {f(r['near_tumour']['ece'], 4)} | {f(r['near_tumour']['brier'], 4)} | "
                         f"{f(r['body']['ece'], 5)} | {f(r['seconds_per_scan'], 0)} | {'reference' if v is None else ('yes' if v['recommended'] else 'no')} |")
        lines.append("")
    else:
        lines += ["Not run yet (`scripts/calibrate_segmentation.py`).", ""]

    lines += ["## Models", ""]
    for m in order:
        r = by_model[m]
        lines.append(f"- `{m}`: {r['model'].get('label') or ''} Checkpoint `{(r['model'].get('checkpoint_sha256') or 'n/a')[:16]}`, "
                     f"report `{r['report_id']}`.")
    for m, why in PENDING.items():
        if m not in by_model:
            lines.append(f"- `{m}`: {why}")
    lines.append("")
    out = ROOT / "docs" / "SEGMENTATION_COMPARISON.md"
    out.write_text("\n".join(lines))
    print(f"wrote {out} ({len(order)} models)")


if __name__ == "__main__":
    main()
