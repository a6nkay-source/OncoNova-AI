"""Logistic progression-risk model with a bootstrap ensemble for uncertainty."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from onconova.progression.features import CLINICAL_FEATURES, FEATURES, missing_required, to_row
from onconova.safety import DEMO_MODEL_LABEL, EvidenceKind, unavailable

MODEL_DIR_NAME = "progression-demo"


def make_pipeline(seed: int = 0) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=0.5, max_iter=2000, random_state=seed)),
    ])


@dataclass
class ProgressionModel:
    pipeline: Pipeline
    ensemble: list[Pipeline]
    feature_ranges: dict[str, list[float]]
    training_predictions: np.ndarray
    card: dict[str, Any]

    @classmethod
    def fit(cls, X: np.ndarray, y: np.ndarray, *, seed: int = 0, n_bootstrap: int = 50,
            card: dict[str, Any] | None = None) -> ProgressionModel:
        pipeline = make_pipeline(seed).fit(X, y)
        rng = np.random.default_rng(seed)
        ensemble = []
        while len(ensemble) < n_bootstrap:
            idx = rng.integers(0, len(y), len(y))
            if len(np.unique(y[idx])) < 2:
                continue
            ensemble.append(make_pipeline(seed).fit(X[idx], y[idx]))
        ranges = {}
        for j, name in enumerate(FEATURES):
            col = X[:, j][~np.isnan(X[:, j])]
            ranges[name] = [float(np.quantile(col, 0.01)), float(np.quantile(col, 0.99))]
        return cls(pipeline, ensemble, ranges, pipeline.predict_proba(X)[:, 1], dict(card or {}))

    def transformed_names(self) -> list[str]:
        return list(self.pipeline[:-1].get_feature_names_out(FEATURES))

    def importance(self) -> list[dict[str, Any]]:
        """Standardised coefficients: association within this model, not causation."""
        coef = self.pipeline[-1].coef_[0]
        out = [{"feature": n, "standardized_log_odds": round(float(c), 4)} for n, c in zip(self.transformed_names(), coef)]
        return sorted(out, key=lambda d: abs(d["standardized_log_odds"]), reverse=True)

    def predict(self, features: dict[str, Any]) -> dict[str, Any]:
        missing = missing_required(features)
        if missing:
            return unavailable(
                "Required imaging features are missing, so no progression estimate was produced.",
                missing=missing,
                how_to_fix="Provide at least two co-registered scans with a lesion segmentation at baseline.",
            )
        row = to_row(features)
        p = float(self.pipeline.predict_proba(row)[0, 1])
        ens = np.array([m.predict_proba(row)[0, 1] for m in self.ensemble])
        lo, hi = np.quantile(ens, [0.025, 0.975])
        z = self.pipeline[:-1].transform(row)[0]
        coef = self.pipeline[-1].coef_[0]
        contributions = sorted(
            ({"feature": n, "contribution_log_odds": round(float(c * v), 4)} for n, c, v in zip(self.transformed_names(), coef, z)),
            key=lambda d: abs(d["contribution_log_odds"]), reverse=True)
        warnings = []
        for name, (a, b) in self.feature_ranges.items():
            v = features.get(name)
            if v is not None and not (a <= float(v) <= b):
                warnings.append(f"{name}={float(v):.3g} is outside the training 1st-99th percentile range [{a:.3g}, {b:.3g}].")
        imputed = [k for k in CLINICAL_FEATURES if features.get(k) is None]
        width = float(hi - lo)
        return {
            "status": "ok",
            "kind": EvidenceKind.MODEL_OUTPUT.value,
            "label": DEMO_MODEL_LABEL,
            "outcome": self.card.get("outcome_definition"),
            "horizon_days": self.card.get("horizon_days"),
            "probability": round(p, 4),
            "interval95": [round(float(lo), 4), round(float(hi), 4)],
            "interval_method": f"2.5-97.5th percentile over {len(self.ensemble)} bootstrap refits (model uncertainty only)",
            "uncertain": width > 0.25 or bool(warnings),
            "training_percentile": round(float((self.training_predictions < p).mean() * 100), 1),
            "contributions": contributions,
            "imputed_features": imputed,
            "out_of_distribution_warnings": warnings,
            "inputs": {k: features.get(k) for k in FEATURES},
            "requires_expert_review": True,
            "limitations": self.card.get("limitations", []),
        }

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "model.joblib"
        joblib.dump({"pipeline": self.pipeline, "ensemble": self.ensemble, "feature_ranges": self.feature_ranges,
                     "training_predictions": self.training_predictions}, path)
        card = dict(self.card)
        card["artifact_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        card["feature_importance"] = self.importance()
        (directory / "model_card.json").write_text(json.dumps(card, indent=2))

    @classmethod
    def load(cls, directory: Path) -> ProgressionModel | None:
        path = directory / "model.joblib"
        card_path = directory / "model_card.json"
        if not path.exists() or not card_path.exists():
            return None
        card = json.loads(card_path.read_text())
        if card.get("artifact_sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
            return None
        blob = joblib.load(path)
        return cls(blob["pipeline"], blob["ensemble"], blob["feature_ranges"], blob["training_predictions"], card)
