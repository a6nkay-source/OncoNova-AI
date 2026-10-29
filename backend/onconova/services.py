"""Orchestration between the store, the models and the API.

The API layer calls these functions and never touches model internals directly.
Each function stores its result with provenance so the dashboard and research
summary only show what was actually computed.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from onconova.config import Settings
from onconova.dose import dvh as dvh_mod
from onconova.dose.synthetic import SCENARIOS, illustrate
from onconova.errors import ModelInputError, NotFoundError, OncoNovaError
from onconova.longitudinal.compare import TimepointInput, analyze
from onconova.progression import features as prog_features
from onconova.progression.model import MODEL_DIR_NAME, ProgressionModel
from onconova.provenance import provenance
from onconova.safety import (SIMULATION_LABEL, EvidenceKind, finding, unavailable)
from onconova.segmentation import uncertainty as unc
from onconova.segmentation.base import LESION_CLASS_NAMES
from onconova.segmentation.measure import components, describe_component, volume_cc
from onconova.segmentation.metrics import evaluate_pair
from onconova.segmentation.registry import get_segmenter, segmenters
from onconova.store import CaseStore

STRUCTURE_NAMES = {1: "lung_left", 2: "lung_right", 3: "heart", 4: "esophagus", 5: "spinal_cord"}


def source_key(source: str) -> str:
    if source == "reference":
        return "ref"
    if source == "annotation":
        return "ann"
    if source.startswith("pred:"):
        return "pred-" + source[5:]
    raise OncoNovaError("invalid_source", "source must be 'reference', 'annotation' or 'pred:<model_id>'.")


def lesion_map(store: CaseStore, case_id: str, tp_id: str, source: str) -> np.ndarray | None:
    if source == "reference":
        return store.label(case_id, tp_id, "reference_lesions")
    if source == "annotation":
        return store.label(case_id, tp_id, "annotation")
    if source.startswith("pred:"):
        return store.label(case_id, tp_id, f"pred_{source[5:]}")
    source_key(source)
    return None


# ----- segmentation -----------------------------------------------------------

def run_segmentation(settings: Settings, store: CaseStore, case_id: str, tp_id: str, model_id: str,
                     mc_samples: int = 8) -> dict[str, Any]:
    case = store.get_case(case_id)
    volume = store.image(case_id, tp_id)
    model = get_segmenter(settings.models_dir, model_id)
    if model.card.status != "available":
        raise ModelInputError("model_unavailable", model.card.unavailable_reason or "Model unavailable.")
    t0 = time.time()
    result = model.predict(volume, mc_samples=mc_samples)
    runtime = time.time() - t0

    store.save_map(case_id, tp_id, f"pred_{model_id}", result.labels)
    store.save_map(case_id, tp_id, f"unc_{model_id}", result.uncertainty.astype(np.float32))
    if result.saliency is not None:
        store.save_map(case_id, tp_id, f"sal_{model_id}", result.saliency)

    comps, n = components(result.labels > 0)
    lesions = []
    for i in range(1, n + 1):
        mask = comps == i
        desc = describe_component(volume, mask, result.labels)
        desc["class"] = LESION_CLASS_NAMES.get(desc.get("class_id", 0), "lesion")
        desc["mean_foreground_probability"] = round(float(result.foreground_probability[mask].mean()), 4)
        desc["mean_uncertainty"] = round(float(result.uncertainty[mask].mean()), 4)
        lesions.append(desc)
    lesions.sort(key=lambda d: d["volume_cc"], reverse=True)
    regions = unc.unreliable_regions(result.uncertainty, volume, result.labels)

    findings = []
    for les in lesions:
        kind = EvidenceKind.UNCERTAIN if les["mean_uncertainty"] > 0.35 else EvidenceKind.MODEL_OUTPUT
        findings.append(finding(kind, f"Model segmented a {les['class']} region of {les['volume_cc']:.2f} cc "
                                      f"(mean probability {les['mean_foreground_probability']:.2f}).", value=les["volume_cc"], unit="cc"))
    for w in result.warnings:
        findings.append(finding(EvidenceKind.EXPERT_REVIEW if "outside" in w or "not assessed" in w else EvidenceKind.UNCERTAIN, w))

    summary = {
        "case_id": case_id, "tp_id": tp_id, "model_id": model_id,
        "kind": EvidenceKind.MODEL_OUTPUT.value, "label": model.card.label,
        "model": model.card.to_dict(),
        "volumes_cc": {name: round(volume_cc(result.labels == k, volume.spacing), 4) for k, name in LESION_CLASS_NAMES.items()},
        "volume_kind": EvidenceKind.MEASURED_FROM_MODEL.value,
        "lesions": lesions,
        "uncertainty": unc.summarize(result.uncertainty, result.labels),
        "uncertainty_method": model.card.uncertainty_method,
        "unreliable_regions": regions,
        "assessed_fraction_of_body": round(float(result.assessed_region.mean()), 4),
        "has_saliency": result.saliency is not None,
        "warnings": result.warnings,
        "findings": findings,
        "runtime_s": round(runtime, 2),
        "provenance": provenance(case=case, model=model.card.to_dict(), mc_samples=mc_samples),
    }
    store.save_result(case_id, f"seg_{tp_id.lower()}_{model_id}", summary)
    return summary


def compare_to_reference(store: CaseStore, case_id: str, tp_id: str, model_id: str, reference: str) -> dict[str, Any]:
    volume = store.image(case_id, tp_id)
    pred = store.label(case_id, tp_id, f"pred_{model_id}")
    if pred is None:
        raise NotFoundError("no_prediction", f"Run {model_id} on {tp_id} first.")
    ref = lesion_map(store, case_id, tp_id, reference)
    if ref is None:
        return unavailable(f"No {reference} mask exists for this timepoint.", missing=[reference],
                           how_to_fix="Draw and save a manual annotation in the Segmentation view.")
    body = store.label(case_id, tp_id, "structures")
    roi = (body > 0) if body is not None else (volume.array > -500 if volume.modality == "CT" else None)
    per_class = {}
    for k, name in LESION_CLASS_NAMES.items():
        if (ref == k).any() or (pred == k).any():
            per_class[name] = evaluate_pair(pred == k, ref == k, volume.spacing_zyx, roi)
    case = store.get_case(case_id)
    ref_kind = case.get("reference", {}).get("kind") if reference == "reference" else "manual annotation"
    out = {
        "case_id": case_id, "tp_id": tp_id, "model_id": model_id, "reference": reference,
        "reference_kind": ref_kind,
        "any_lesion": evaluate_pair(pred > 0, ref > 0, volume.spacing_zyx, roi),
        "per_class": per_class,
        "kind": EvidenceKind.MEASURED.value,
        "note": "Agreement for a single scan. Population performance is in Model Evaluation (patient-separated test set).",
    }
    store.save_result(case_id, f"cmp_{tp_id.lower()}_{model_id}_{source_key(reference)}", out)
    return out


# ----- longitudinal -----------------------------------------------------------

def run_longitudinal(store: CaseStore, case_id: str, source: str) -> dict[str, Any]:
    case = store.get_case(case_id)
    if len(case["timepoints"]) < 2:
        return unavailable("Longitudinal comparison needs at least two scans of the same patient.",
                           missing=["second timepoint"], how_to_fix="Import a follow-up scan for this case.")
    inputs, missing = [], []
    for tp in case["timepoints"]:
        lesions = lesion_map(store, case_id, tp["tp_id"], source)
        if lesions is None:
            missing.append(tp["tp_id"])
            continue
        inputs.append(TimepointInput(tp["tp_id"], int(tp["days_from_baseline"]), store.image(case_id, tp["tp_id"]), lesions))
    if missing:
        return unavailable(f"No {source} segmentation for timepoint(s) {', '.join(missing)}.", missing=missing,
                           how_to_fix="Run the segmentation model on every timepoint, or choose another source.")
    result = analyze(inputs, source=source)
    result.update({"status": "ok", "case_id": case_id, "provenance": provenance(case=case, source=source)})
    store.save_result(case_id, f"long_{source_key(source)}", result)
    return result


# ----- progression ------------------------------------------------------------

def progression_model(settings: Settings) -> ProgressionModel | None:
    return ProgressionModel.load(settings.models_dir / MODEL_DIR_NAME)


def run_progression(settings: Settings, store: CaseStore, case_id: str, source: str) -> dict[str, Any]:
    case = store.get_case(case_id)
    model = progression_model(settings)
    if model is None:
        return unavailable("The progression model has not been trained.", missing=["model artifact"],
                           how_to_fix="Run `make train-progression`.")
    longitudinal = store.result(case_id, f"long_{source_key(source)}") or run_longitudinal(store, case_id, source)
    if longitudinal.get("status") != "ok":
        out = unavailable("Progression risk needs a longitudinal comparison, which is unavailable for this case.",
                          missing=longitudinal.get("missing_inputs", []), how_to_fix=longitudinal.get("how_to_fix"))
        out["upstream_reason"] = longitudinal.get("reason")
        return out
    feats = prog_features.from_longitudinal(longitudinal, case.get("clinical"))
    prediction = model.predict(feats)
    prediction.update({"case_id": case_id, "source": source,
                       "provenance": provenance(case=case, model=model.card, source=source)})
    if prediction.get("status") == "ok":
        store.save_result(case_id, f"prog_{source_key(source)}", prediction)
    return prediction


# ----- dose -------------------------------------------------------------------

def run_dose(store: CaseStore, case_id: str, scenario_id: str, target_source: str) -> dict[str, Any]:
    case = store.get_case(case_id)
    if scenario_id not in SCENARIOS:
        raise NotFoundError("unknown_scenario", f"No scenario {scenario_id!r}.", details={"available": sorted(SCENARIOS)})
    scenario = SCENARIOS[scenario_id]
    tp_id = case["timepoints"][0]["tp_id"]
    volume = store.image(case_id, tp_id)
    lesions = lesion_map(store, case_id, tp_id, target_source)
    if lesions is None or not (lesions == 1).any():
        return unavailable("No primary-lesion contour to use as the illustrative target.", missing=["target contour"],
                           how_to_fix="Choose a source with a primary lesion (reference, annotation or a model run).")
    structures = store.label(case_id, tp_id, "structures")
    warnings = []
    if structures is None:
        warnings.append("No organ-at-risk contours: DVHs are shown for the target and body only.")
        body = volume.array > -500 if volume.modality == "CT" else volume.array > np.percentile(volume.array, 20)
        oars: dict[str, np.ndarray] = {}
    else:
        body = structures > 0
        oars = {name: structures == k for k, name in STRUCTURE_NAMES.items()}
    beam_slice = int(round(float(np.argwhere(lesions == 1)[:, 0].mean())))
    out = illustrate(volume, lesions == 1, body, scenario, beam_slice=beam_slice)
    key = f"dose_{scenario_id.lower()}_{source_key(target_source)}"
    store.save_map(case_id, tp_id, key, out["dose"])
    store.save_array(case_id, f"{key}_beams", out["beam_slices"])
    structs = {"illustrative_target": out["ptv"], **{k: v & ~out["ptv"] for k, v in oars.items()}, "body": body}
    report = dvh_mod.dvh_report(out["dose"], structs, "illustrative_target", volume.voxel_volume_mm3 / 1000)
    result = {
        "status": "ok", "kind": EvidenceKind.SIMULATED.value, "label": SIMULATION_LABEL,
        "case_id": case_id, "tp_id": tp_id, "scenario": scenario.to_dict(), "target_source": target_source,
        "dose_units": out["units"], "dose_map": key, "dvh": report,
        "beams": {"angles_deg": [round(float(a), 3) for a in out["angles_deg"]], "slice_index": beam_slice,
                  "angle_convention": "IEC 61217 gantry angle, head-first supine: 0 = anterior, 90 = patient left",
                  "note": "Each beam's share of the illustration on the axial slice through the target, for replay."},
        "warnings": out["warnings"] + warnings,
        "not_a_prescription": True,
        "provenance": provenance(case=case, engine="onconova.dose.synthetic (geometric illustration)"),
    }
    store.save_result(case_id, f"dose_{scenario_id.lower()}_{source_key(target_source)}", result)
    return result


# ----- research summary -------------------------------------------------------

def research_summary(settings: Settings, store: CaseStore, case_id: str) -> dict[str, Any]:
    case = store.get_case(case_id)
    results = store.results(case_id)
    seg = {k: v for k, v in results.items() if k.startswith("seg_")}
    long_ = {k: v for k, v in results.items() if k.startswith("long_")}
    prog = {k: v for k, v in results.items() if k.startswith("prog_")}
    dose = {k: v for k, v in results.items() if k.startswith("dose_")}

    a = [finding(EvidenceKind.MEASURED, f"{len(case['timepoints'])} {case['modality']} scan(s); "
                 f"voxel spacing {case['timepoints'][0]['geometry']['spacing_xyz_mm']} mm.")]
    if case.get("demo"):
        a.append(finding(EvidenceKind.SIMULATED, "All images are synthetic phantoms (DEMO DATA)."))
    for w in case.get("warnings", []):
        a.append(finding(EvidenceKind.UNCERTAIN, w))

    b, e, g = [], [], []
    for s in seg.values():
        vols = ", ".join(f"{k} {v:.2f} cc" for k, v in s["volumes_cc"].items() if v > 0) or "no lesion voxels"
        b.append(finding(EvidenceKind.MODEL_OUTPUT, f"{s['tp_id']} / {s['model_id']}: {vols}."))
        for w in s.get("warnings", []):
            e.append(finding(EvidenceKind.UNCERTAIN, f"{s['tp_id']} / {s['model_id']}: {w}"))
        if s.get("unreliable_regions"):
            g.append(finding(EvidenceKind.EXPERT_REVIEW, f"{s['tp_id']} / {s['model_id']}: "
                             f"{len(s['unreliable_regions'])} high-uncertainty region(s) in the segmentation."))
        e += [finding(EvidenceKind.UNCERTAIN, f"{s['model_id']} limitation: {lim}") for lim in s["model"]["limitations"][:2]]

    c = []
    for lres in long_.values():
        for f in lres["findings"]:
            (g if f["kind"] == EvidenceKind.EXPERT_REVIEW.value else c).append(f)

    d = []
    for p in prog.values():
        d.append(finding(EvidenceKind.MODEL_OUTPUT,
                         f"Estimated probability of the simulated outcome ({p['outcome']}) is {p['probability']:.2f} "
                         f"(95% interval {p['interval95'][0]:.2f}-{p['interval95'][1]:.2f}; source {p['source']}).",
                         value=p["probability"]))
        if p.get("uncertain"):
            e.append(finding(EvidenceKind.UNCERTAIN, "Progression estimate is uncertain (wide interval or out-of-range inputs)."))
        for w in p.get("out_of_distribution_warnings", []):
            e.append(finding(EvidenceKind.UNCERTAIN, w))
        g.append(finding(EvidenceKind.EXPERT_REVIEW, "Any progression estimate needs interpretation by a qualified clinician."))
    if not prog:
        d.append(finding(EvidenceKind.UNAVAILABLE, "No progression estimate has been computed for this case."))

    f_items = []
    for r in dose.values():
        t = r["dvh"]["stats"].get("illustrative_target", {})
        f_items.append(finding(EvidenceKind.SIMULATED,
                               f"{r['scenario']['name']}: target D95 {t.get('D95', 0) * 100:.0f}% of reference, "
                               f"body V5 {r['dvh']['stats']['body']['V5'] * 100:.0f}% (relative units, illustration only)."))
    e.append(finding(EvidenceKind.UNCERTAIN, "Models in this prototype were trained or tuned on synthetic data only."))
    return {
        "case_id": case_id, "label": case.get("label"), "demo": case.get("demo", False),
        "sections": {
            "A_imaging_observations": a,
            "B_segmentation_results": b or [finding(EvidenceKind.UNAVAILABLE, "No segmentation has been run.")],
            "C_changes_between_timepoints": c or [finding(EvidenceKind.UNAVAILABLE, "No longitudinal comparison available.")],
            "D_progression_model_output": d,
            "E_uncertainty_and_limitations": e,
            "F_research_radiation_simulation": f_items or [finding(EvidenceKind.UNAVAILABLE, "No simulation has been run.")],
            "G_items_requiring_expert_review": g or [finding(EvidenceKind.EXPERT_REVIEW, "All outputs require expert review before any use.")],
        },
        "disclaimer": "Research summary. Not a clinical report, diagnosis or treatment recommendation.",
        "provenance": provenance(case=case, models=sorted(segmenters(settings.models_dir))),
    }
