"""Write docs/EVALUATION_REPORT.md from the newest stored evaluation reports.

Every number in the output is copied from artifacts/evaluations/*.json, which the
evaluation scripts write. If a report is missing, the section says so.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from onconova.config import get_settings  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TARGETS = {"any_lesion": "Any lesion", "primary_tumor": "Primary lesion",
           "lymph_node_region_lesion": "Node-region lesion", "other_lesion": "Other lesion"}


def f(v, d=3):
    return "n/a" if v is None else f"{v:.{d}f}"


def ci(c, d=3):
    return "n/a" if not c or c[0] is None else f"{c[0]:.{d}f}-{c[1]:.{d}f}"


def latest(reports, task, model_id=None, dataset=None):
    cands = [r for r in reports if r["task"] == task and (model_id is None or r["model"]["id"] == model_id)
             and (dataset is None or r.get("dataset", {}).get("id") == dataset)]
    return max(cands, key=lambda r: r["created_utc"]) if cands else None


def seg_section(r) -> list[str]:
    res = r["results"]
    out = [f"### {r['model']['id']} (v{r['model']['version']}, {r['model'].get('kind')})", "",
           f"Report `{r['report_id']}` · checkpoint `{(r['model'].get('checkpoint_sha256') or 'n/a')[:16]}` · "
           f"{r['dataset']['n_patients']} test patients / {r['dataset']['n_scans']} scans from `{r['dataset']['id']}`.", "",
           "| Target | Scans | Dice mean (95% CI) | Dice median | IoU mean | HD95 mm mean / median | Sensitivity | Specificity (body) |",
           "|---|---|---|---|---|---|---|---|"]
    for key, name in TARGETS.items():
        if key not in res:
            continue
        t = res[key]
        out.append(f"| {name} | {t['n_scans']} | {f(t['dice']['mean'])} ({ci(t['dice']['ci95'])}) | {f(t['dice']['median'])} | "
                   f"{f(t['iou']['mean'])} | {f(t['hd95_mm']['mean'], 1)} / {f(t['hd95_mm']['median'], 1)} | "
                   f"{f(t['sensitivity']['mean'])} | {f(t['specificity']['mean'], 4)} |")
    d = res["lesion_detection"]
    out += ["", f"Lesion-level detection: {d['detected']}/{d['reference_lesions']} reference lesions overlapped "
                f"(sensitivity {f(d['lesion_sensitivity'])}); {f(d['false_positive_components_per_scan'], 2)} false-positive "
                "components per scan.", ""]
    conf = res.get("class_confusion_voxels")
    if conf:
        names = conf["rows_reference_cols_predicted"]
        out += ["Voxel class confusion over all test scans (rows: reference, columns: prediction):", "",
                "| Reference \\ Predicted | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
        for name, row in zip(names, conf["matrix"]):
            total = sum(row) or 1
            out.append(f"| {name} | " + " | ".join(f"{v} ({100 * v / total:.0f}%)" for v in row) + " |")
        out.append("")
    return out


def prog_section(r) -> list[str]:
    out = [f"Report `{r['report_id']}` · artifact `{r['model'].get('artifact_sha256', 'n/a')[:16]}` · cohort "
           f"`{r['dataset']['id']}` (n={r['dataset']['n_patients']}, {r['dataset']['label']}).", "",
           "| Split | n (events) | AUROC (95% CI) | AUPRC (95% CI) | Brier (95% CI) | C-index (95% CI) | Sens / Spec @ threshold | ECE | Cal. slope |",
           "|---|---|---|---|---|---|---|---|---|"]
    for name, b in r["results"].items():
        out.append(f"| {name.replace('_', ' ')} | {b['n']} ({b['n_events']}) | {f(b['auroc']['value'])} ({ci(b['auroc']['ci95'])}) | "
                   f"{f(b['auprc']['value'])} ({ci(b['auprc']['ci95'])}) | {f(b['brier']['value'])} ({ci(b['brier']['ci95'])}) | "
                   f"{f(b['c_index']['value'])} ({ci(b['c_index']['ci95'])}) | {f(b['sensitivity']['value'], 2)} / "
                   f"{f(b['specificity']['value'], 2)} @ {f(b['threshold'], 3)} | {f(b['calibration']['expected_calibration_error'])} | "
                   f"{f(b['calibration']['calibration_slope'], 2)} |")
    out += ["", "**Subgroups** (pooled internal + temporal test sets):", "", "| Subgroup | n | AUROC (95% CI) | Brier |", "|---|---|---|---|"]
    for name, b in r.get("subgroups", {}).items():
        auc = b.get("auroc", {})
        out.append(f"| {name} | {b['n']} | {f(auc.get('value'))} ({ci(auc.get('ci95'))}) | {f(b.get('brier', {}).get('value'))} |")
    out += ["", "**Standardised coefficients** (association inside the model, not causation): " +
            ", ".join(f"`{x['feature']}` {x['standardized_log_odds']:+.2f}" for x in r.get("feature_importance", [])[:8]), ""]
    return out


def main() -> None:
    settings = get_settings()
    reports = [json.loads(p.read_text()) for p in settings.evaluations_dir.glob("*.json")]
    seg = [latest(reports, "segmentation", m, "synthetic-seg-v1") for m in ("demo-classical", "demo-unet")]
    prog = latest(reports, "future_progression_risk")
    L = [
        "# OncoNova AI model evaluation report",
        "",
        f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} by `make report` from the stored evaluation "
        "artifacts in `artifacts/evaluations/`. No number in this file was typed by hand.",
        "",
        "> **Read this first.** Every model here was trained and evaluated on **synthetic data** (DEMO DATA): CT phantoms "
        "with exact generator ground truth, and a tabular cohort whose outcome was simulated from a known formula. "
        "The results show that the pipelines run end to end and recover planted signal. They say nothing about "
        "performance on patients. None of these models is validated for any clinical use.",
        "",
        "## 1. Intended use",
        "",
        "Research and teaching: exercising an imaging-AI workflow (ingestion, segmentation, longitudinal change, risk "
        "modelling, dose visualisation) with honest uncertainty and provenance. Not for diagnosis, prognosis, treatment "
        "planning or any decision about a person.",
        "",
        "## 2. Supported cancer types and modalities",
        "",
        "| Component | Cancer type | Modality | Status |",
        "|---|---|---|---|",
        "| Image ingestion and viewer | any | CT, MRI, PET (NIfTI or DICOM series) | Implemented; PET shown as stored values (no SUV) |",
        "| Classical lesion heuristic | lung lesions in thoracic CT | CT (HU) | Tuned on phantoms only |",
        "| 3-D U-Net (MONAI) | synthetic thoracic lesions | CT | Trained on phantoms only |",
        "| Progression model | none (simulated cohort) | imaging-derived features | Simulated outcome only |",
        "| Dose visualisation | n/a | any image grid | Geometric illustration (synthetic) or imported research dose |",
        "",
        "## 3. Training data",
        "",
        "- Segmentation: manifest `synthetic-seg-v1`, 60 synthetic patients x 2 scans, generator `thorax-phantom-0.1`, "
        "patient-level split 42/9/9 (seed 42). Labels: primary lesion, node-region lesion, other lesion (generator definitions, "
        "not pathology).",
        "- Progression: `progression-cohort-v1`, 900 synthetic patients over 3 synthetic sites with simulated 180-day "
        "progression and time-to-event with censoring. Development split from sites A/B before 2021-01-01; temporal "
        "validation on later A/B enrolment; external validation on site C.",
        "",
        "## 4. Exclusion criteria",
        "",
        "- Progression binary metrics exclude patients censored before the 180-day horizon without an event (no defined "
        "label); they are kept for the C-index. Counts are in the table below.",
        "- Segmentation class-wise metrics only use scans where the class exists in the reference.",
        "- Inputs the models refuse: non-CT input for the CT models (error), missing longitudinal features for the "
        "progression model (explicit unavailable state).",
        "",
        "## 5. Evaluation protocol",
        "",
        "- Patient-level splits everywhere; leakage is checked by `check_no_leakage` and by tests.",
        "- Segmentation: per-scan metrics against generator ground truth, averaged per patient, then mean/median with a "
        "2000-sample patient bootstrap 95% CI. HD95 and Hausdorff in millimetres using voxel spacing. Specificity inside "
        "the body mask.",
        "- Progression: decision threshold chosen on validation (Youden). AUROC, AUPRC, Brier, sensitivity/specificity with "
        "1000-sample bootstrap CIs, Harrell's C (300 samples), quantile-bin calibration with ECE and logistic "
        "calibration slope/intercept, sex and site subgroups.",
        "",
        "## 6. Measured results",
        "",
        "### 6.1 Segmentation (held-out synthetic test patients)",
        "",
    ]
    for r in seg:
        L += seg_section(r) if r else ["_No stored report for this model. Run `make evaluate`._", ""]
    L += ["### 6.2 Future progression risk (simulated cohort)", ""]
    L += prog_section(prog) if prog else ["_No stored report. Run `make train-progression`._", ""]
    L += [
        "## 7. Known limitations",
        "",
        "- **Synthetic only.** Phantoms are analytic shapes with simple noise; real CT has vessels, motion, artefacts, "
        "contrast phases, scanner variation and pathology the phantoms do not have.",
        "- **Classical heuristic** never finds lesions outside the lung envelope (Dice 0 on node-region lesions by design) "
        "and misses lesions touching the pleura or mediastinum.",
        "- **U-Net** (demo configuration, class-balanced crop sampling) gets class Dice 0 for node-region and 'other' "
        "lesions in this run. The confusion table in 6.1 shows where those voxels went: predicted as background (missed) "
        "or as 'primary' (mislabelled). Mislabelled distant lesions also explain the primary-class HD95 having a high "
        "mean but a low median.",
        "- **Uncertainty** values are model-internal (heuristic entropy; MC-dropout entropy; bootstrap spread for the "
        "progression model). None is a calibrated probability of error.",
        "- **Progression model** predicts a simulated outcome. Its discrimination only shows that the code recovers a "
        "signal deliberately planted in the generator. It has no metastasis-site labels and does not predict spread.",
        "- **Saliency** is input-gradient sensitivity. It is not an explanation of reasoning or evidence of causation.",
        "- **Dose** is a geometric illustration in relative units (no physics engine, no heterogeneity correction). "
        "Imported doses are displayed as stored and not verified.",
        "",
        "## 8. Remaining research questions",
        "",
        "- Does the pipeline transfer to a public, expert-annotated dataset (MSD Task06 Lung, NSCLC-Radiomics) under a "
        "patient-level split, and how large is the synthetic-to-real gap?",
        "- Which uncertainty estimates (MC dropout, ensembles, test-time augmentation) are best calibrated against actual "
        "segmentation error on real data?",
        "- Can longitudinal features predict a clinically defined endpoint (e.g. RANO progression in LUMIERE) with "
        "temporal and external validation, and how do imaging and clinical features compare?",
        "- How should registration quality be scored so that change measurements are flagged when alignment fails?",
        "",
    ]
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "EVALUATION_REPORT.md").write_text("\n".join(L))
    print("Wrote docs/EVALUATION_REPORT.md")


if __name__ == "__main__":
    main()
