"""Provenance records attached to every stored result."""

from __future__ import annotations

import platform
from datetime import datetime, timezone
from typing import Any

from onconova import __version__


def provenance(*, case: dict[str, Any], model: dict[str, Any] | None = None, **extra: Any) -> dict[str, Any]:
    return {
        "onconova_version": __version__,
        "python": platform.python_version(),
        "computed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "case_id": case["case_id"],
        "data_label": case.get("label"),
        "demo_data": bool(case.get("demo")),
        "data_provenance": case.get("provenance", {}),
        "model": None if model is None else {
            "id": model.get("id"), "version": model.get("version"), "kind": model.get("kind"),
            "checkpoint_sha256": model.get("checkpoint_sha256") or model.get("artifact_sha256"),
            "training_manifest": model.get("training_manifest"),
        },
        **extra,
    }
