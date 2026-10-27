"""Patient-level and temporal splitting with leakage checks.

All scans from one patient always land in the same split. Never split serial
imaging at the image level: follow-up scans of a training patient in the test
set leak identity, anatomy and scanner information into the evaluation.
"""

from __future__ import annotations

import hashlib
import hmac
from collections import defaultdict
from collections.abc import Iterable, Sequence
from datetime import date
from typing import Any

import numpy as np

SPLITS = ("train", "val", "test")


class LeakageError(ValueError):
    pass


def pseudonymous_id(raw_id: str, salt: str, prefix: str = "P") -> str:
    """Stable identifier that does not expose the source ID."""
    return f"{prefix}-{hmac.new(salt.encode(), raw_id.encode(), hashlib.sha256).hexdigest()[:10]}"


def patient_level_split(patient_ids: Iterable[str], fractions: Sequence[float] = (0.7, 0.15, 0.15),
                        seed: int = 42) -> dict[str, str]:
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("split fractions must sum to 1")
    unique = sorted(set(patient_ids))
    order = np.random.default_rng(seed).permutation(len(unique))
    n = len(unique)
    n_train = int(round(fractions[0] * n))
    n_val = int(round(fractions[1] * n))
    assignment = {}
    for rank, idx in enumerate(order):
        split = "train" if rank < n_train else "val" if rank < n_train + n_val else "test"
        assignment[unique[idx]] = split
    return assignment


def temporal_split(records: Sequence[dict[str, Any]], *, date_key: str, cutoff: date,
                   patient_key: str = "patient_id") -> dict[str, str]:
    """Patients enrolled before ``cutoff`` train; later patients test. Uses each patient's first date."""
    first: dict[str, date] = {}
    for r in records:
        d = r[date_key] if isinstance(r[date_key], date) else date.fromisoformat(str(r[date_key]))
        pid = r[patient_key]
        first[pid] = min(first.get(pid, d), d)
    return {pid: ("train" if d < cutoff else "test") for pid, d in first.items()}


def check_no_leakage(records: Iterable[dict[str, Any]], *, patient_key: str = "patient_id",
                     split_key: str = "split") -> dict[str, int]:
    seen: dict[str, set[str]] = defaultdict(set)
    counts: dict[str, int] = defaultdict(int)
    for r in records:
        seen[r[patient_key]].add(r[split_key])
        counts[r[split_key]] += 1
    leaked = sorted(pid for pid, splits in seen.items() if len(splits) > 1)
    if leaked:
        raise LeakageError(f"{len(leaked)} patient(s) appear in more than one split: {leaked[:5]}")
    return dict(counts)
