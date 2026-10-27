"""Synthetic progression cohort. DEMO DATA: the outcome is simulated, not observed.

Feature distributions roughly mimic what the imaging pipeline measures on the
phantoms. The outcome follows a known generative model (an exponential
time-to-event with censoring), so the training workflow, metrics and calibration
can be checked end to end. Any performance measured on this cohort only shows
that the code recovers a signal that was deliberately planted.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from onconova.safety import DEMO_LABEL

HORIZON_DAYS = 180
SITES = ("site_A", "site_B", "site_C")


def generate(n_patients: int = 900, seed: int = 7) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    start = date(2015, 1, 1)
    for i in range(n_patients):
        site = SITES[int(rng.choice(3, p=[0.45, 0.35, 0.20]))]
        vol = float(np.exp(rng.normal(2.6, 0.7)))
        growth = float(rng.normal(0.05, 0.35))
        new = int(rng.poisson(0.25 + 0.6 * max(growth, 0)))
        node = int(rng.random() < 0.3)
        sph = float(np.clip(rng.normal(0.78, 0.07), 0.45, 0.98))
        age = float(np.clip(rng.normal(66, 9), 35, 90))
        sex = "F" if rng.random() < 0.48 else "M"
        site_shift = {"site_A": 0.0, "site_B": 0.15, "site_C": -0.2}[site]
        lin = (-2.9 + 0.55 * (np.log(vol) - 2.6) / 0.7 + 1.9 * growth + 0.7 * new + 0.5 * node
               - 2.0 * (sph - 0.78) + 0.015 * (age - 66) + site_shift)
        hazard = np.exp(lin) / 30.0  # per day
        t_event = float(rng.exponential(1.0 / hazard))
        t_censor = float(rng.uniform(120, 720))
        observed_time = min(t_event, t_censor)
        event = int(t_event <= t_censor)
        # Binary label at the horizon is undefined if censored before the horizon without an event.
        label = 1 if (event and observed_time <= HORIZON_DAYS) else (0 if observed_time > HORIZON_DAYS else None)
        rows.append({
            "patient_id": f"SYNP-{i:05d}",
            "label": DEMO_LABEL,
            "site": site,
            "sex": sex,
            "enrollment_date": (start + timedelta(days=int(rng.uniform(0, 8 * 365)))).isoformat(),
            "baseline_total_volume_cc": vol,
            "log_volume_ratio_per_100d": growth,
            "new_lesion_count": new,
            "node_region_lesion_observed": node,
            "primary_sphericity": sph if rng.random() > 0.05 else None,
            "age_years": age if rng.random() > 0.15 else None,
            "time_days": observed_time,
            "event": event,
            f"progression_within_{HORIZON_DAYS}d": label,
        })
    return rows
