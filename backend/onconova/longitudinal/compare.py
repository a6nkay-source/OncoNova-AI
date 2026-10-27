"""Compare serial scans of one patient and quantify observed lesion changes.

This module answers "what changed between the scans we have?". It does not
predict the future (see onconova.progression) and it does not interpret new
findings: a new region in a follow-up segmentation is reported as a new finding
that needs expert review, never as metastasis.

Volumes are measured on each scan's native grid. Lesions are matched across time
in the baseline frame after rigid registration of each follow-up to baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from onconova.imaging.registration import register_rigid, resample_to
from onconova.imaging.volume import Volume
from onconova.safety import EvidenceKind, finding
from onconova.segmentation.base import LESION_CLASS_NAMES
from onconova.segmentation.measure import components, describe_component, volume_cc

MATCH_DISTANCE_MM = 15.0


@dataclass
class TimepointInput:
    tp_id: str
    days: int
    volume: Volume
    lesions: np.ndarray  # class map on the scan's own grid


def _measurement_kind(source: str) -> str:
    return (EvidenceKind.MEASURED if source == "reference" else EvidenceKind.MEASURED_FROM_MODEL).value


def _lesion_list(volume: Volume, lesions: np.ndarray) -> list[dict[str, Any]]:
    comps, n = components(lesions > 0)
    out = []
    for i in range(1, n + 1):
        mask = comps == i
        desc = describe_component(volume, mask, lesions)
        desc["class"] = LESION_CLASS_NAMES.get(desc.get("class_id", 0), "lesion")
        desc["_mask"] = mask
        out.append(desc)
    return out


def _match(base: list[dict], follow: list[dict], follow_in_base: np.ndarray, base_volume: Volume) -> dict[int, int]:
    """Greedy matching of follow-up lesions to baseline lesions by overlap, then centroid distance."""
    pairs = []
    for j, f in enumerate(follow):
        comps, _ = components(follow_in_base == (j + 1))
        f_mask = comps > 0
        f_centroid = base_volume.index_to_physical(np.argwhere(f_mask).mean(0)) if f_mask.any() else None
        for i, b in enumerate(base):
            overlap = int(np.sum(b["_mask"] & f_mask))
            dist = float(np.linalg.norm(np.asarray(b["centroid_mm"]) - f_centroid)) if f_centroid is not None else np.inf
            if overlap > 0 or dist <= MATCH_DISTANCE_MM:
                pairs.append((-overlap, dist, i, j))
    pairs.sort()
    used_b, used_f, matches = set(), set(), {}
    for _, _, i, j in pairs:
        if i not in used_b and j not in used_f:
            matches[j] = i
            used_b.add(i)
            used_f.add(j)
    return matches


def analyze(timepoints: list[TimepointInput], *, source: str, register: bool = True) -> dict[str, Any]:
    if len(timepoints) < 2:
        raise ValueError("longitudinal analysis needs at least two timepoints")
    timepoints = sorted(timepoints, key=lambda t: t.days)
    base = timepoints[0]
    base_lesions = _lesion_list(base.volume, base.lesions)
    kind = _measurement_kind(source)

    tracks: list[dict[str, Any]] = []
    for i, les in enumerate(base_lesions):
        tracks.append({"track_id": f"L{i + 1}", "class": les["class"], "first_seen": base.tp_id,
                       "measurements": {base.tp_id: {k: v for k, v in les.items() if k != "_mask"}},
                       "baseline_centroid_mm": les["centroid_mm"]})
    base_track_of = {i: i for i in range(len(base_lesions))}

    tp_summaries = [{
        "tp_id": base.tp_id, "days_from_baseline": base.days,
        "total_lesion_volume_cc": round(volume_cc(base.lesions > 0, base.volume.spacing), 3),
        "lesion_count": len(base_lesions), "registration": None,
    }]
    for tp in timepoints[1:]:
        reg = register_rigid(base.volume, tp.volume) if register else None
        follow = _lesion_list(tp.volume, tp.lesions)
        # Component index map of the follow-up, moved into the baseline frame.
        follow_index = np.zeros(tp.lesions.shape, dtype=np.uint8)
        for j, f in enumerate(follow):
            follow_index[f["_mask"]] = j + 1
        moved = follow_index if reg is None and tp.volume.same_geometry(base.volume) else resample_to(
            base.volume, tp.volume.with_array(follow_index), reg.transform if reg else None, is_label=True)
        matches = _match(base_lesions, follow, moved, base.volume)
        for j, f in enumerate(follow):
            entry = {k: v for k, v in f.items() if k != "_mask"}
            mask_in_base = moved == (j + 1)
            entry["centroid_in_baseline_frame_mm"] = (
                [round(float(v), 2) for v in base.volume.index_to_physical(np.argwhere(mask_in_base).mean(0))]
                if mask_in_base.any() else None)
            if j in matches:
                track = tracks[base_track_of[matches[j]]]
            else:
                track = next((t for t in tracks if t["first_seen"] != base.tp_id and t.get("_last_centroid") is not None
                              and entry["centroid_in_baseline_frame_mm"] is not None
                              and np.linalg.norm(np.subtract(t["_last_centroid"], entry["centroid_in_baseline_frame_mm"])) <= MATCH_DISTANCE_MM), None)
                if track is None:
                    track = {"track_id": f"L{len(tracks) + 1}", "class": f["class"], "first_seen": tp.tp_id,
                             "measurements": {}, "baseline_centroid_mm": None}
                    tracks.append(track)
            track["measurements"][tp.tp_id] = entry
            track["_last_centroid"] = entry["centroid_in_baseline_frame_mm"]
        tp_summaries.append({
            "tp_id": tp.tp_id, "days_from_baseline": tp.days,
            "total_lesion_volume_cc": round(volume_cc(tp.lesions > 0, tp.volume.spacing), 3),
            "lesion_count": len(follow),
            "registration": reg.summary() if reg else None,
        })

    last = timepoints[-1].tp_id
    findings = []
    for t in tracks:
        t.pop("_last_centroid", None)
        series = [t["measurements"][tp.tp_id]["volume_cc"] if tp.tp_id in t["measurements"] else None for tp in timepoints]
        t["volume_series_cc"] = series
        first = next((v for v in series if v is not None), None)
        last_v = series[-1]
        if t["first_seen"] != base.tp_id:
            t["status"] = "new"
            findings.append(finding(EvidenceKind.EXPERT_REVIEW,
                                    f"New region {t['track_id']} ({t['class']}) first segmented at {t['first_seen']}, "
                                    f"{last_v or 0:.2f} cc. Needs expert review; a new segmented region is not by itself "
                                    "evidence of metastasis.", track_id=t["track_id"]))
        elif last_v is None:
            t["status"] = "not_segmented_at_last_scan"
            findings.append(finding(EvidenceKind.EXPERT_REVIEW,
                                    f"{t['track_id']} is not segmented at {last}; confirm resolution versus segmentation failure.",
                                    track_id=t["track_id"]))
        else:
            t["status"] = "persisting"
        if first and last_v is not None and t["status"] == "persisting":
            t["volume_change_pct"] = round(100 * (last_v - first) / first, 1)
            c0 = np.asarray(t["baseline_centroid_mm"])
            c1 = t["measurements"][last].get("centroid_in_baseline_frame_mm")
            t["centroid_shift_mm"] = round(float(np.linalg.norm(np.asarray(c1) - c0)), 2) if c1 else None
            findings.append(finding(EvidenceKind(kind),
                                    f"{t['track_id']} ({t['class']}) volume {first:.2f} -> {last_v:.2f} cc "
                                    f"({t['volume_change_pct']:+.1f}%) between {base.tp_id} and {last}.",
                                    track_id=t["track_id"], value=t["volume_change_pct"], unit="%"))

    total0, total1 = tp_summaries[0]["total_lesion_volume_cc"], tp_summaries[-1]["total_lesion_volume_cc"]
    for s in tp_summaries[1:]:
        if s["registration"] and s["registration"]["qc"] != "ok":
            findings.append(finding(EvidenceKind.EXPERT_REVIEW,
                                    f"Registration of {s['tp_id']} to baseline moved >30 mm or rotated >10 deg; check alignment."))
    return {
        "source": source,
        "measurement_kind": kind,
        "baseline_tp": base.tp_id,
        "timepoints": tp_summaries,
        "tracks": tracks,
        "total_volume_change_pct": round(100 * (total1 - total0) / total0, 1) if total0 > 0 else None,
        "new_lesion_count": sum(1 for t in tracks if t["status"] == "new"),
        "findings": findings,
        "notes": [
            "Observed change analysis only. It does not estimate future progression.",
            "Volumes come from segmentation masks; if the masks are model output, so are the volumes.",
            "Max axial diameter is a mask measurement, not a RECIST assessment.",
        ],
    }
