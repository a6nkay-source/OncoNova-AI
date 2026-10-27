"""Health, system status, model evaluations, dataset catalogue, audit log and settings."""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml
from fastapi import APIRouter, Depends, Query, Request

from onconova import __version__
from onconova.api.deps import audit, get_settings, get_store, require
from onconova.config import Settings
from onconova.errors import NotFoundError
from onconova.imaging.deid import load_profile
from onconova.safety import DEMO_LABEL, RESEARCH_DISCLAIMER, SIMULATION_LABEL
from onconova.security import Principal
from onconova.segmentation.registry import segmenters
from onconova.store import CaseStore

router = APIRouter(prefix="/api", tags=["system"])
REPORT_ID = re.compile(r"^[A-Za-z0-9_.-]{1,120}$")


@router.get("/health")
def health() -> dict:
    """Liveness probe. No authentication; returns no data."""
    return {"status": "ok", "version": __version__}


@router.get("/system")
def system(request: Request, settings: Settings = Depends(get_settings), store: CaseStore = Depends(get_store),
           principal: Principal = Depends(require("viewer"))) -> dict:
    """Version, safety statements, auth mode, model status and data counts."""
    cases = store.list_cases()
    return {
        "name": "OncoNova AI",
        "version": __version__,
        "disclaimer": RESEARCH_DISCLAIMER,
        "demo_label": DEMO_LABEL,
        "simulation_label": SIMULATION_LABEL,
        "device_control": "none - this software has no device interfaces and cannot send commands to medical equipment",
        "prescription": "never - the software does not determine or recommend radiation prescriptions",
        "external_transmission": "none - images and results stay on this server",
        "auth_mode": settings.auth_mode,
        "principal": {"user": principal.user, "role": principal.role},
        "limits": {"max_upload_mb": settings.max_upload_mb, "max_zip_members": settings.max_zip_members},
        "models": [{"id": m.card.id, "status": m.card.status, "reason": m.card.unavailable_reason}
                   for m in segmenters(settings.models_dir).values()],
        "cases": {"total": len(cases), "demo": sum(1 for c in cases if c.get("demo"))},
    }


def _reports(settings: Settings) -> list[Path]:
    return sorted(settings.evaluations_dir.glob("*.json"), reverse=True)


@router.get("/evaluations")
def evaluations(settings: Settings = Depends(get_settings), _: Principal = Depends(require("viewer"))) -> dict:
    """Measured evaluation reports written by the evaluation scripts. Empty if none have been run."""
    items = []
    for path in _reports(settings):
        data = json.loads(path.read_text())
        items.append({"report_id": data.get("report_id", path.stem), "task": data.get("task"),
                      "model": data.get("model"), "dataset": {k: data.get("dataset", {}).get(k) for k in ("id", "label")},
                      "created_utc": data.get("created_utc")})
    return {"reports": items, "note": "Every number comes from a stored evaluation run; none are entered by hand."}


@router.get("/evaluations/{report_id}")
def evaluation(report_id: str, settings: Settings = Depends(get_settings), _: Principal = Depends(require("viewer"))) -> dict:
    """One full evaluation report."""
    if not REPORT_ID.match(report_id):
        raise NotFoundError("report_not_found", "No such report.")
    path = settings.evaluations_dir / f"{report_id}.json"
    if not path.exists():
        raise NotFoundError("report_not_found", "No such report.")
    return json.loads(path.read_text())


@router.get("/datasets")
def datasets(settings: Settings = Depends(get_settings), _: Principal = Depends(require("viewer"))) -> dict:
    """Public dataset catalogue (verified fields, with 'not stated' where unverified) and local manifests."""
    catalogue_path = settings.configs_dir / "datasets.yaml"
    catalogue = yaml.safe_load(catalogue_path.read_text()) if catalogue_path.exists() else {"datasets": []}
    local = []
    synthetic_root = settings.data_dir / "synthetic"
    for manifest in sorted(synthetic_root.glob("*/manifest.json")):
        m = json.loads(manifest.read_text())
        local.append({k: m.get(k) for k in ("manifest_id", "label", "description", "created_utc", "generator", "n_patients",
                                            "split_unit", "dataset_seed", "split_seed", "patient_counts_by_split",
                                            "scan_counts_by_split", "label_definitions")}
                     | {"quality_issues": sum(1 for r in m["records"] for q in r.get("quality", []) if q.startswith("error"))})
    cohort = synthetic_root / "progression-cohort-v1" / "cohort.csv"
    if cohort.exists():
        local.append({"manifest_id": "progression-cohort-v1", "label": DEMO_LABEL,
                      "description": "Synthetic tabular progression cohort with simulated outcomes.",
                      "n_patients": sum(1 for _ in cohort.open()) - 1, "split_unit": "patient"})
    return {"catalogue": catalogue.get("datasets", []), "catalogue_checked": catalogue.get("checked"),
            "local_manifests": local}


@router.get("/audit")
def audit_log(request: Request, n: int = Query(200, ge=1, le=2000), principal: Principal = Depends(require("admin"))) -> dict:
    """Most recent audit events (admin only). Contains identifiers and actions, never image data."""
    audit(request, principal, "view_audit_log", "audit")
    return {"events": request.app.state.audit.tail(n)}


@router.get("/settings/deid-profile")
def deid_profile(settings: Settings = Depends(get_settings), _: Principal = Depends(require("researcher"))) -> dict:
    """The active DICOM de-identification profile (configs/deid_profile.yaml or the built-in default)."""
    path = settings.configs_dir / "deid_profile.yaml"
    return {"source": str(path.name) if path.exists() else "built-in default", "profile": load_profile(path)}
