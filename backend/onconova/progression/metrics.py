"""Discrimination, calibration and survival metrics with bootstrap intervals."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def sensitivity_specificity(y: np.ndarray, p: np.ndarray, threshold: float) -> tuple[float, float]:
    pred = p >= threshold
    tp, fn = np.sum(pred & (y == 1)), np.sum(~pred & (y == 1))
    tn, fp = np.sum(~pred & (y == 0)), np.sum(pred & (y == 0))
    return float(tp / max(tp + fn, 1)), float(tn / max(tn + fp, 1))


def youden_threshold(y: np.ndarray, p: np.ndarray) -> float:
    candidates = np.unique(np.quantile(p, np.linspace(0.02, 0.98, 97)))
    scores = [sum(sensitivity_specificity(y, p, t)) - 1 for t in candidates]
    return float(candidates[int(np.argmax(scores))])


def calibration_curve(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> dict[str, Any]:
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    bins = []
    ece = 0.0
    for b in range(len(edges) - 1):
        m = idx == b
        if not m.any():
            continue
        mean_p, frac = float(p[m].mean()), float(y[m].mean())
        bins.append({"mean_predicted": round(mean_p, 4), "observed_fraction": round(frac, 4), "n": int(m.sum())})
        ece += m.mean() * abs(mean_p - frac)
    # Calibration slope/intercept from logistic recalibration on the logit of p.
    logit = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    slope, intercept = _logistic_recalibration(logit, y)
    return {"bins": bins, "expected_calibration_error": round(float(ece), 4),
            "calibration_slope": slope, "calibration_intercept": intercept, "binning": "quantile"}


def _logistic_recalibration(logit: np.ndarray, y: np.ndarray) -> tuple[float | None, float | None]:
    from sklearn.linear_model import LogisticRegression

    if len(np.unique(y)) < 2:
        return None, None
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit[:, None], y)
    return round(float(lr.coef_[0, 0]), 4), round(float(lr.intercept_[0]), 4)


def concordance_index(time: np.ndarray, event: np.ndarray, risk: np.ndarray) -> float | None:
    """Harrell's C: among comparable pairs, the fraction where higher risk fails earlier."""
    order = np.argsort(time)
    time, event, risk = time[order], event[order], risk[order]
    concordant = permissible = 0.0
    for i in range(len(time)):
        if not event[i]:
            continue
        later = time > time[i]
        n = later.sum()
        if n == 0:
            continue
        permissible += n
        concordant += np.sum(risk[i] > risk[later]) + 0.5 * np.sum(risk[i] == risk[later])
    return None if permissible == 0 else float(concordant / permissible)


def bootstrap_ci(fn: Callable[..., float | None], arrays: tuple[np.ndarray, ...], *, n: int = 1000,
                 seed: int = 0, alpha: float = 0.05) -> tuple[float | None, float | None]:
    rng = np.random.default_rng(seed)
    size = len(arrays[0])
    stats = []
    for _ in range(n):
        idx = rng.integers(0, size, size)
        try:
            v = fn(*(a[idx] for a in arrays))
        except ValueError:
            continue
        if v is not None and np.isfinite(v):
            stats.append(v)
    if len(stats) < n * 0.5:
        return None, None
    return float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))


def binary_report(y: np.ndarray, p: np.ndarray, threshold: float, *, n_boot: int = 1000, seed: int = 0) -> dict[str, Any]:
    y = y.astype(int)
    if len(np.unique(y)) < 2:
        return {"n": int(len(y)), "status": "unavailable", "reason": "Only one outcome class present."}
    sens, spec = sensitivity_specificity(y, p, threshold)

    def with_ci(name: str, fn: Callable[..., float | None]) -> dict[str, Any]:
        lo, hi = bootstrap_ci(fn, (y, p), n=n_boot, seed=seed)
        value = fn(y, p)
        return {"value": None if value is None else round(float(value), 4),
                "ci95": [None if lo is None else round(lo, 4), None if hi is None else round(hi, 4)], "metric": name}

    return {
        "n": int(len(y)),
        "n_events": int(y.sum()),
        "prevalence": round(float(y.mean()), 4),
        "auroc": with_ci("AUROC", lambda a, b: roc_auc_score(a, b)),
        "auprc": with_ci("AUPRC", lambda a, b: average_precision_score(a, b)),
        "brier": with_ci("Brier", lambda a, b: brier_score_loss(a, b)),
        "threshold": round(threshold, 4),
        "threshold_source": "Youden index on the validation split",
        "sensitivity": with_ci("Sensitivity", lambda a, b: sensitivity_specificity(a, b, threshold)[0]),
        "specificity": with_ci("Specificity", lambda a, b: sensitivity_specificity(a, b, threshold)[1]),
        "calibration": calibration_curve(y, p),
    }
