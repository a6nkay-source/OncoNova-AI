"""Segmentation, comparison, longitudinal change, progression risk, dose illustration and summaries."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Request
from fastapi.concurrency import run_in_threadpool

from onconova import services
from onconova.api.deps import audit, get_settings, get_store, require
from onconova.api.schemas import ComparisonRequest, DoseRequest, SegmentationRequest, SourceRequest
from onconova.api.streaming import body_to_tempfile
from onconova.config import Settings
from onconova.dose import dvh as dvh_mod
from onconova.dose.rtdose import import_dose
from onconova.dose.synthetic import SCENARIOS
from onconova.errors import UploadRejected
from onconova.provenance import provenance
from onconova.safety import EvidenceKind, unavailable
from onconova.security import Principal
from onconova.segmentation.registry import segmenters
from onconova.store import CaseStore

router = APIRouter(prefix="/api", tags=["analysis"])


@router.get("/models")
def list_models(settings: Settings = Depends(get_settings), _: Principal = Depends(require("viewer"))) -> dict:
    """Segmentation and progression models with status, model card and provenance."""
    seg = [m.card.to_dict() for m in segmenters(settings.models_dir).values()]
    prog = services.progression_model(settings)
    progression = prog.card | {"status": "available"} if prog else {
        "id": "progression-demo", "status": "unavailable", "unavailable_reason": "Not trained. Run `make train-progression`."}
    return {"segmentation": seg, "progression": progression}


@router.post("/cases/{case_id}/timepoints/{tp_id}/segmentation")
async def segment(case_id: str, tp_id: str, body: SegmentationRequest, request: Request,
                  settings: Settings = Depends(get_settings), store: CaseStore = Depends(get_store),
                  principal: Principal = Depends(require("researcher"))) -> dict:
    """Run a segmentation model on one scan. Returns lesion list, volumes, uncertainty and warnings."""
    result = await run_in_threadpool(services.run_segmentation, settings, store, case_id, tp_id, body.model_id, body.mc_samples)
    audit(request, principal, "run_segmentation", f"{case_id}/{tp_id}", model_id=body.model_id)
    return result


@router.get("/cases/{case_id}/timepoints/{tp_id}/segmentation")
def get_segmentation(case_id: str, tp_id: str, model_id: str = Query(..., pattern=r"^[a-z0-9-]{1,40}$"),
                     store: CaseStore = Depends(get_store), _: Principal = Depends(require("viewer"))) -> dict:
    """Most recent stored segmentation summary, or an explicit unavailable state."""
    store.timepoint(case_id, tp_id)
    return store.result(case_id, f"seg_{tp_id.lower()}_{model_id}") or unavailable(
        f"{model_id} has not been run on {tp_id}.", how_to_fix="Run segmentation.")


@router.post("/cases/{case_id}/timepoints/{tp_id}/comparison")
async def compare(case_id: str, tp_id: str, body: ComparisonRequest, store: CaseStore = Depends(get_store),
                  _: Principal = Depends(require("researcher"))) -> dict:
    """Agreement between a model prediction and the reference or manual annotation (Dice, IoU, HD, HD95, ...)."""
    return await run_in_threadpool(services.compare_to_reference, store, case_id, tp_id, body.model_id, body.reference)


@router.post("/cases/{case_id}/longitudinal")
async def longitudinal(case_id: str, body: SourceRequest, request: Request, store: CaseStore = Depends(get_store),
                       principal: Principal = Depends(require("researcher"))) -> dict:
    """Register follow-ups to baseline and quantify observed lesion changes (no prediction)."""
    result = await run_in_threadpool(services.run_longitudinal, store, case_id, body.source)
    audit(request, principal, "run_longitudinal", case_id, source=body.source)
    return result


@router.get("/cases/{case_id}/longitudinal")
def get_longitudinal(case_id: str, source: str = "reference", store: CaseStore = Depends(get_store),
                     _: Principal = Depends(require("viewer"))) -> dict:
    """Stored longitudinal result for a lesion source, or an unavailable state."""
    return store.result(case_id, f"long_{services.source_key(source)}") or unavailable(
        "No longitudinal comparison has been computed for this source.", how_to_fix="Run the comparison.")


@router.post("/cases/{case_id}/progression")
async def progression(case_id: str, body: SourceRequest, request: Request, settings: Settings = Depends(get_settings),
                      store: CaseStore = Depends(get_store), principal: Principal = Depends(require("researcher"))) -> dict:
    """Estimate the (simulated) future-progression outcome, with interval, or return why it is unavailable."""
    result = await run_in_threadpool(services.run_progression, settings, store, case_id, body.source)
    audit(request, principal, "run_progression", case_id, source=body.source, status=result.get("status"))
    return result


@router.get("/progression/model")
def progression_model(settings: Settings = Depends(get_settings), _: Principal = Depends(require("viewer"))) -> dict:
    """Progression model card, including its outcome definition, horizon, features and limitations."""
    model = services.progression_model(settings)
    if model is None:
        return unavailable("Progression model not trained.", how_to_fix="Run `make train-progression`.")
    return model.card | {"status": "available"}


@router.get("/dose/scenarios")
def dose_scenarios(_: Principal = Depends(require("viewer"))) -> dict:
    """Hypothetical beam-arrangement scenarios for the geometric illustration."""
    return {"scenarios": [s.to_dict() for s in SCENARIOS.values()], "kind": EvidenceKind.SIMULATED.value}


@router.post("/cases/{case_id}/dose")
async def dose(case_id: str, body: DoseRequest, request: Request, store: CaseStore = Depends(get_store),
               principal: Principal = Depends(require("researcher"))) -> dict:
    """Compute the SIMULATED, NON-CLINICAL geometric dose illustration and its DVHs (relative units)."""
    result = await run_in_threadpool(services.run_dose, store, case_id, body.scenario_id, body.target_source)
    audit(request, principal, "run_dose_illustration", case_id, scenario=body.scenario_id)
    return result


@router.get("/cases/{case_id}/dose")
def get_dose(case_id: str, scenario_id: str = Query(..., pattern=r"^S[0-9]{1,2}$"), target_source: str = "reference",
             store: CaseStore = Depends(get_store), _: Principal = Depends(require("viewer"))) -> dict:
    """Stored dose illustration for a scenario, or an unavailable state."""
    return store.result(case_id, f"dose_{scenario_id.lower()}_{services.source_key(target_source)}") or unavailable(
        "This scenario has not been computed yet.", how_to_fix="Run the simulation.")


@router.post("/cases/{case_id}/dose/import")
async def dose_import(case_id: str, request: Request, filename: str = Query(..., max_length=200),
                      target_source: str = "reference", settings: Settings = Depends(get_settings),
                      store: CaseStore = Depends(get_store), principal: Principal = Depends(require("researcher"))) -> dict:
    """Import a research dose grid (DICOM RTDOSE or NIfTI) as the raw body; shown as stored, never verified."""
    name = filename.lower()
    if not name.endswith((".dcm", ".nii", ".nii.gz")):
        raise UploadRejected("unsupported_extension", "Upload an RTDOSE .dcm file or a NIfTI dose grid.")
    case = store.get_case(case_id)
    tp_id = case["timepoints"][0]["tp_id"]
    suffix = ".nii.gz" if name.endswith(".nii.gz") else ".nii" if name.endswith(".nii") else ".dcm"
    tmp = await body_to_tempfile(request, settings.tmp_dir, settings.max_upload_mb * 1024 * 1024)
    typed = tmp.parent / (tmp.name + suffix)
    try:
        tmp.rename(typed)
        volume = store.image(case_id, tp_id)
        dose_arr, info = await run_in_threadpool(import_dose, typed, volume)
    finally:
        tmp.unlink(missing_ok=True)
        typed.unlink(missing_ok=True)
    store.save_map(case_id, tp_id, "dose_import", dose_arr)
    structs = {}
    lesions = services.lesion_map(store, case_id, tp_id, target_source)
    if lesions is not None and (lesions == 1).any():
        structs["target_contour"] = lesions == 1
    structures = store.label(case_id, tp_id, "structures")
    if structures is not None:
        structs.update({n: structures == k for k, n in services.STRUCTURE_NAMES.items()})
        structs["body"] = structures > 0
    vmax = float(dose_arr.max()) or 1.0
    report = dvh_mod.dvh_report(dose_arr, structs, "target_contour", volume.voxel_volume_mm3 / 1000, max_level=vmax)
    result = {"status": "ok", "kind": "imported", "label": info["label"], "import": info, "dose_map": "dose_import",
              "dose_max": vmax, "dvh": report, "case_id": case_id, "tp_id": tp_id,
              "warnings": [] if structs else ["No contours available; DVH cannot be computed."],
              "provenance": provenance(case=case, engine="imported file")}
    store.save_result(case_id, "dose_import", result)
    audit(request, principal, "import_dose", case_id)
    return result


@router.get("/cases/{case_id}/summary")
def summary(case_id: str, settings: Settings = Depends(get_settings), store: CaseStore = Depends(get_store),
            _: Principal = Depends(require("viewer"))) -> dict:
    """Research summary, sections A-G, each item tagged measured / model output / uncertain / expert review."""
    return services.research_summary(settings, store, case_id)


@router.get("/cases/{case_id}/results")
def list_results(case_id: str, store: CaseStore = Depends(get_store), _: Principal = Depends(require("viewer"))) -> dict:
    """Names and timestamps of every stored result for this case."""
    return {name: {"saved_utc": r.get("saved_utc"), "kind": r.get("kind")} for name, r in store.results(case_id).items()}
