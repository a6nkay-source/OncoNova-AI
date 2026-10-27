"""Cases, image volumes, overlays, metadata, annotations and uploads."""

from __future__ import annotations

import threading
from typing import Any, Literal

import numpy as np
from fastapi import APIRouter, Depends, Query, Request, Response

from onconova.api.binary import volume_response
from onconova.api.deps import audit, get_settings, get_store, require
from onconova.api.streaming import body_to_tempfile
from onconova.config import Settings
from onconova.errors import NotFoundError, OncoNovaError, UploadRejected
from onconova.imaging.registration import register_rigid, resample_to
from onconova.security import Principal
from onconova.segmentation.measure import mesh
from onconova.services import lesion_map
from onconova.store import CaseStore
from onconova.uploads import ingest

router = APIRouter(prefix="/api", tags=["cases"])
_registration_cache: dict[tuple[str, str, str], Any] = {}
_reg_lock = threading.Lock()


def _case_summary(case: dict[str, Any]) -> dict[str, Any]:
    return {k: case.get(k) for k in ("case_id", "title", "demo", "label", "modality", "cancer_type", "body_region")} | {
        "timepoints": [{k: tp.get(k) for k in ("tp_id", "label", "days_from_baseline", "modality")} for tp in case["timepoints"]],
        "has_reference": bool(case.get("reference", {}).get("available")),
        "has_structures": bool(case.get("structures", {}).get("available")),
    }


@router.get("/cases")
def list_cases(store: CaseStore = Depends(get_store), _: Principal = Depends(require("viewer"))) -> list[dict]:
    """List all cases (demo and uploaded) with their timepoints."""
    return [_case_summary(c) for c in store.list_cases()]


@router.get("/cases/{case_id}")
def get_case(case_id: str, store: CaseStore = Depends(get_store), _: Principal = Depends(require("viewer"))) -> dict:
    """Full case record: timepoints with geometry, reference/structure availability, provenance."""
    case = store.get_case(case_id)
    case.pop("metadata", None)  # technical DICOM metadata is served by the role-gated endpoint
    return case


@router.delete("/cases/{case_id}")
def delete_case(case_id: str, request: Request, store: CaseStore = Depends(get_store),
                principal: Principal = Depends(require("researcher"))) -> dict:
    """Permanently delete an uploaded case and all derived results. Demo cases cannot be deleted."""
    store.delete_case(case_id)
    audit(request, principal, "delete_case", case_id)
    return {"deleted": case_id}


@router.get("/cases/{case_id}/timepoints/{tp_id}/image")
def get_image(case_id: str, tp_id: str, store: CaseStore = Depends(get_store),
              _: Principal = Depends(require("viewer"))) -> Response:
    """Image voxels as int16 with geometry headers (see onconova.api.binary)."""
    vol = store.image(case_id, tp_id)
    case = store.get_case(case_id)
    return volume_response(vol, vol.array, kind="image", label=case.get("label"))


MapKind = Literal["reference", "structures", "annotation", "prediction", "uncertainty", "saliency", "dose"]


@router.get("/cases/{case_id}/timepoints/{tp_id}/maps/{kind}")
def get_map(case_id: str, tp_id: str, kind: MapKind, model_id: str | None = Query(None, pattern=r"^[a-z0-9-]{1,40}$"),
            key: str | None = Query(None, pattern=r"^dose_[a-z0-9_-]{1,60}$"), store: CaseStore = Depends(get_store),
            _: Principal = Depends(require("viewer"))) -> Response:
    """Overlay volumes on the image grid: label maps as uint8, continuous maps quantised to uint8.

    ``prediction``/``uncertainty``/``saliency`` need ``model_id``; ``dose`` needs ``key`` (from a dose result).
    """
    vol = store.image(case_id, tp_id)
    names = {"reference": "reference_lesions", "structures": "structures", "annotation": "annotation"}
    if kind in names:
        arr = store.label(case_id, tp_id, names[kind])
        if arr is None:
            raise NotFoundError("map_not_found", f"No {kind} map for this timepoint.")
        return volume_response(vol, arr, kind="label")
    if kind == "dose":
        if not key:
            raise OncoNovaError("key_required", "Pass key=<dose_map> from a dose result.", status_code=400)
        arr = store.float_map(case_id, tp_id, key)
        if arr is None:
            raise NotFoundError("map_not_found", "No such dose map; run the simulation first.")
        return volume_response(vol, arr, kind="map", vmax=1.25, label="SIMULATED - NON-CLINICAL")
    if not model_id:
        raise OncoNovaError("model_required", "Pass model_id for model outputs.", status_code=400)
    if kind == "prediction":
        arr = store.label(case_id, tp_id, f"pred_{model_id}")
        if arr is None:
            raise NotFoundError("map_not_found", "No prediction yet; run segmentation first.")
        return volume_response(vol, arr, kind="label", label="MODEL OUTPUT")
    prefix = "unc" if kind == "uncertainty" else "sal"
    arr = store.float_map(case_id, tp_id, f"{prefix}_{model_id}")
    if arr is None:
        raise NotFoundError("map_not_found", f"No {kind} map for {model_id} at this timepoint.")
    return volume_response(vol, arr, kind="map", vmax=1.0, label="MODEL OUTPUT")


@router.get("/cases/{case_id}/timepoints/{tp_id}/metadata")
def get_metadata(case_id: str, tp_id: str, request: Request, store: CaseStore = Depends(get_store),
                 principal: Principal = Depends(require("researcher"))) -> dict:
    """Geometry and de-identified technical metadata. Identifying fields are reported by name only."""
    case = store.get_case(case_id)
    tp = store.timepoint(case_id, tp_id)
    vol = store.image(case_id, tp_id)
    audit(request, principal, "view_metadata", f"{case_id}/{tp_id}")
    meta = case.get("metadata", {})
    return {
        "case_id": case_id, "tp_id": tp_id, "modality": vol.modality, "label": case.get("label"),
        "geometry": vol.geometry(),
        "intensity_range": [float(vol.array.min()), float(vol.array.max())],
        "source_format": vol.metadata.get("source_format"),
        "original_direction": meta.get("original_direction") or vol.metadata.get("original_direction"),
        "dicom": meta.get("dicom"),
        "deidentification": case.get("deidentification"),
        "warnings": case.get("warnings", []) + vol.metadata.get("warnings", []),
        "ground_truth": tp.get("ground_truth"),
        "provenance": case.get("provenance"),
    }


@router.put("/cases/{case_id}/timepoints/{tp_id}/annotation")
async def put_annotation(case_id: str, tp_id: str, request: Request, store: CaseStore = Depends(get_store),
                         principal: Principal = Depends(require("researcher"))) -> dict:
    """Save a manual annotation: raw uint8 body, C order, image shape, values 0-3 (lesion classes)."""
    vol = store.image(case_id, tp_id)
    body = await request.body()
    expected = int(np.prod(vol.shape))
    if len(body) != expected:
        raise UploadRejected("bad_annotation_size", f"Expected {expected} bytes for shape {list(vol.shape)}, got {len(body)}.")
    arr = np.frombuffer(body, dtype=np.uint8).reshape(vol.shape)
    if arr.max() > 3:
        raise UploadRejected("bad_annotation_values", "Annotation values must be 0-3.")
    store.save_map(case_id, tp_id, "annotation", arr.copy())
    case = store.get_case(case_id)
    case.setdefault("annotations", {})[tp_id] = {"by": principal.user, "voxels": int((arr > 0).sum())}
    store.write_case(case)
    audit(request, principal, "save_annotation", f"{case_id}/{tp_id}", voxels=int((arr > 0).sum()))
    return {"saved": True, "voxels": int((arr > 0).sum()), "volume_cc": round(float((arr > 0).sum()) * vol.voxel_volume_mm3 / 1000, 4)}


@router.delete("/cases/{case_id}/timepoints/{tp_id}/annotation")
def delete_annotation(case_id: str, tp_id: str, request: Request, store: CaseStore = Depends(get_store),
                      principal: Principal = Depends(require("researcher"))) -> dict:
    """Delete the manual annotation for this timepoint."""
    path = store.label_path(case_id, tp_id, "annotation")
    path.unlink(missing_ok=True)
    audit(request, principal, "delete_annotation", f"{case_id}/{tp_id}")
    return {"deleted": True}


@router.get("/cases/{case_id}/timepoints/{tp_id}/mesh")
def get_mesh(case_id: str, tp_id: str, source: str = Query("reference"), label: int = Query(1, ge=1, le=10),
             store: CaseStore = Depends(get_store), _: Principal = Depends(require("viewer"))) -> dict:
    """Surface mesh (LPS mm) for 3-D display. ``source``: reference, annotation, pred:<model> or structures."""
    vol = store.image(case_id, tp_id)
    arr = store.label(case_id, tp_id, "structures") if source == "structures" else lesion_map(store, case_id, tp_id, source)
    if arr is None:
        raise NotFoundError("map_not_found", f"No {source} map for this timepoint.")
    result = mesh(arr == label, vol)
    return result or {"vertices": [], "faces": [], "n_vertices": 0, "n_faces": 0, "empty": True}


@router.get("/cases/{case_id}/timepoints/{tp_id}/registered")
def get_registered(case_id: str, tp_id: str, kind: Literal["image", "lesions"] = "image", source: str = "reference",
                   store: CaseStore = Depends(get_store), _: Principal = Depends(require("viewer"))) -> Response:
    """This timepoint resampled into the baseline frame after rigid registration (for side-by-side review)."""
    case = store.get_case(case_id)
    base_id = case["timepoints"][0]["tp_id"]
    base, moving = store.image(case_id, base_id), store.image(case_id, tp_id)
    key = (case_id, tp_id, base_id)
    with _reg_lock:
        reg = _registration_cache.get(key)
    if reg is None and tp_id != base_id:
        reg = register_rigid(base, moving)
        with _reg_lock:
            _registration_cache[key] = reg
    transform = reg.transform if reg else None
    if kind == "image":
        arr = resample_to(base, moving, transform)
        return volume_response(base, arr, kind="image")
    lesions = lesion_map(store, case_id, tp_id, source)
    if lesions is None:
        raise NotFoundError("map_not_found", f"No {source} lesions for this timepoint.")
    arr = resample_to(base, moving.with_array(lesions.astype(np.uint8)), transform, is_label=True)
    return volume_response(base, arr, kind="label")


@router.post("/uploads")
async def upload(request: Request, filename: str = Query(..., max_length=200),
                 modality: Literal["CT", "MR", "PT"] | None = None, settings: Settings = Depends(get_settings),
                 store: CaseStore = Depends(get_store), principal: Principal = Depends(require("researcher"))) -> dict:
    """Upload one volume as the raw request body: .nii / .nii.gz (pass ``modality``) or a .zip of one DICOM series.

    The filename is used only to detect the format and is never stored. DICOM is de-identified before
    parsing; only the converted volume and de-identified metadata are kept.
    """
    tmp = await body_to_tempfile(request, settings.tmp_dir, settings.max_upload_mb * 1024 * 1024)
    try:
        case = ingest(tmp, filename, modality=modality, settings=settings, store=store, uploaded_by=principal.user)
    except OncoNovaError as exc:
        audit(request, principal, "upload_rejected", "uploads", code=exc.code)
        raise
    finally:
        tmp.unlink(missing_ok=True)
    audit(request, principal, "upload", case["case_id"], modality=case["modality"])
    return _case_summary(case) | {"warnings": case.get("warnings", []), "deidentification": case.get("deidentification")}
