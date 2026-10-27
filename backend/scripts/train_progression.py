"""Train and evaluate the demo progression-risk model on the synthetic cohort.

    python scripts/train_progression.py

Splits (all at the patient level, mutually disjoint):
  * development: sites A and B, enrolled before 2021-01-01 -> train / val / internal test (70/15/15)
  * temporal validation: sites A and B, enrolled on or after 2021-01-01
  * external validation: site C, any date (a site the model never sees in training)

Writes the model to artifacts/models/progression-demo/ and the measured
evaluation to artifacts/evaluations/. Every number in that report is computed
here from the held-out splits; nothing is typed in by hand.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from onconova import __version__  # noqa: E402
from onconova.config import get_settings  # noqa: E402
from onconova.data.splits import check_no_leakage, patient_level_split  # noqa: E402
from onconova.progression import cohort  # noqa: E402
from onconova.progression.features import FEATURES  # noqa: E402
from onconova.progression.metrics import binary_report, bootstrap_ci, concordance_index, youden_threshold  # noqa: E402
from onconova.progression.model import MODEL_DIR_NAME, ProgressionModel  # noqa: E402
from onconova.safety import DEMO_LABEL  # noqa: E402

TEMPORAL_CUTOFF = date(2021, 1, 1)
LABEL = f"progression_within_{cohort.HORIZON_DAYS}d"


def matrix(rows):
    X = np.array([[np.nan if r[f] is None else float(r[f]) for f in FEATURES] for r in rows], dtype=float)
    return X


def evaluate(model, rows, threshold, seed):
    labelled = [r for r in rows if r[LABEL] is not None]
    X, y = matrix(labelled), np.array([r[LABEL] for r in labelled])
    p = model.pipeline.predict_proba(X)[:, 1]
    report = binary_report(y, p, threshold, seed=seed)
    risk_all = model.pipeline.predict_proba(matrix(rows))[:, 1]
    t, e = np.array([r["time_days"] for r in rows]), np.array([r["event"] for r in rows])
    c = concordance_index(t, e, risk_all)
    lo, hi = bootstrap_ci(lambda a, b, cc: concordance_index(a, b, cc), (t, e, risk_all), n=300, seed=seed)
    report["c_index"] = {"value": None if c is None else round(c, 4), "ci95": [None if lo is None else round(lo, 4),
                         None if hi is None else round(hi, 4)], "metric": "Harrell's C (time to simulated event)",
                         "n": len(rows)}
    report["excluded_censored_before_horizon"] = len(rows) - len(labelled)
    return report


def missingness(rows):
    return {f: round(float(np.mean([r[f] is None for r in rows])), 4) for f in FEATURES}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-patients", type=int, default=900)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    settings = get_settings()

    rows = cohort.generate(args.n_patients, seed=args.seed)
    data_dir = settings.data_dir / "synthetic" / "progression-cohort-v1"
    data_dir.mkdir(parents=True, exist_ok=True)
    with (data_dir / "cohort.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    dev = [r for r in rows if r["site"] != "site_C" and date.fromisoformat(r["enrollment_date"]) < TEMPORAL_CUTOFF]
    temporal = [r for r in rows if r["site"] != "site_C" and date.fromisoformat(r["enrollment_date"]) >= TEMPORAL_CUTOFF]
    external = [r for r in rows if r["site"] == "site_C"]
    split_of = patient_level_split([r["patient_id"] for r in dev], seed=args.seed)
    for r in dev:
        r["split"] = split_of[r["patient_id"]]
    for r in temporal:
        r["split"] = "temporal_test"
    for r in external:
        r["split"] = "external_test"
    counts = check_no_leakage(dev + temporal + external)

    train = [r for r in dev if r["split"] == "train" and r[LABEL] is not None]
    val = [r for r in dev if r["split"] == "val" and r[LABEL] is not None]
    internal = [r for r in dev if r["split"] == "test"]

    card = {
        "id": "progression-demo",
        "version": "0.1.0",
        "label": DEMO_LABEL,
        "kind": "logistic regression + 50-model bootstrap ensemble",
        "outcome_definition": f"Simulated progression event within {cohort.HORIZON_DAYS} days after the latest scan (synthetic label).",
        "horizon_days": cohort.HORIZON_DAYS,
        "features": FEATURES,
        "trained_on": f"{len(train)} synthetic patients (sites A/B, enrolled before {TEMPORAL_CUTOFF}).",
        "exclusions": "Patients censored before the horizon without an event have no binary label and are excluded "
                      "from binary training/metrics (they are kept for the C-index).",
        "missing_data": "Median imputation with missingness indicators, fitted on the training split only.",
        "limitations": [
            "Trained and evaluated on a simulated cohort; the outcome was generated from a known formula.",
            "Performance here shows the pipeline recovers a planted signal, not clinical validity.",
            "Does not predict where disease will spread; there are no metastasis-site labels.",
            "Feature contributions are associations inside this model, not causes.",
        ],
        "onconova_version": __version__,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_seed": args.seed,
    }
    Xtr, ytr = matrix(train), np.array([r[LABEL] for r in train])
    model = ProgressionModel.fit(Xtr, ytr, seed=args.seed, card=card)
    yval = np.array([r[LABEL] for r in val])
    threshold = youden_threshold(yval, model.pipeline.predict_proba(matrix(val))[:, 1])
    model.card["decision_threshold_for_metrics"] = round(threshold, 4)
    model.save(settings.models_dir / MODEL_DIR_NAME)

    evaluations = {
        "internal_test": evaluate(model, internal, threshold, args.seed),
        "temporal_validation": evaluate(model, temporal, threshold, args.seed),
        "external_validation_site_C": evaluate(model, external, threshold, args.seed),
    }
    subgroups = {}
    pooled = internal + temporal
    for key in ("sex", "site"):
        for value in sorted({r[key] for r in pooled}):
            subgroups[f"{key}={value}"] = evaluate(model, [r for r in pooled if r[key] == value], threshold, args.seed)

    report = {
        "report_id": f"progression-demo-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "task": "future_progression_risk",
        "model": {"id": card["id"], "version": card["version"],
                  "artifact_sha256": json.loads((settings.models_dir / MODEL_DIR_NAME / "model_card.json").read_text())["artifact_sha256"]},
        "dataset": {"id": "progression-cohort-v1", "label": DEMO_LABEL, "n_patients": len(rows), "seed": args.seed,
                    "split_counts": counts, "temporal_cutoff": TEMPORAL_CUTOFF.isoformat(),
                    "missingness_by_split": {"train": missingness(train), "temporal": missingness(temporal),
                                             "external": missingness(external)}},
        "protocol": "Patient-level splits; threshold chosen on validation (Youden); metrics on held-out internal, "
                    "temporal and external (unseen site) sets; 95% CIs by 1000-sample patient bootstrap (300 for C-index).",
        "results": evaluations,
        "subgroups": subgroups,
        "feature_importance": model.importance(),
        "caveat": "Synthetic cohort with a simulated outcome. These numbers are not evidence of clinical performance.",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out = settings.evaluations_dir / f"{report['report_id']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    for name, r in evaluations.items():
        print(f"{name:28s} n={r['n']:4d} AUROC={r['auroc']['value']} {r['auroc']['ci95']} "
              f"Brier={r['brier']['value']} C={r['c_index']['value']}")
    print(f"Saved model to {settings.models_dir / MODEL_DIR_NAME} and report to {out}")


if __name__ == "__main__":
    main()
