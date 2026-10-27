"""Validated ingestion of uploaded NIfTI volumes and zipped DICOM series.

Safeguards: extension allow-list, magic-byte checks, size limits (enforced while
streaming in the API layer), ZIP member/size/ratio limits, path-traversal and
symlink rejection, DICOM de-identification before anything is parsed further,
private temp directories that are always removed, and no use of the uploaded
filename (it may contain a name) anywhere in stored data.
"""

from __future__ import annotations

import gzip
import secrets
import shutil
import stat
import struct
import tempfile
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import pydicom

from onconova.config import Settings
from onconova.errors import UploadRejected
from onconova.imaging.deid import deidentify, load_profile
from onconova.imaging.io import PHI_KEYWORDS, load_dicom_series, load_nifti, save_nifti
from onconova.store import CaseStore

ALLOWED_MODALITIES = {"CT", "MR", "PT"}
MAX_COMPRESSION_RATIO = 200


def kind_from_name(filename: str) -> str:
    name = filename.lower()
    if name.endswith(".nii.gz"):
        return "nifti_gz"
    if name.endswith(".nii"):
        return "nifti"
    if name.endswith(".zip"):
        return "dicom_zip"
    raise UploadRejected("unsupported_extension", "Upload a .nii, .nii.gz or a .zip of one DICOM series.")


def check_magic(path: Path, kind: str) -> None:
    with path.open("rb") as fh:
        head = fh.read(4)
    if kind == "nifti_gz":
        if head[:2] != b"\x1f\x8b":
            raise UploadRejected("bad_magic", "File is not gzip-compressed.")
        with gzip.open(path, "rb") as fh:
            head = fh.read(4)
    if kind in ("nifti", "nifti_gz"):
        if len(head) < 4 or (struct.unpack("<i", head)[0] not in (348, 540) and struct.unpack(">i", head)[0] not in (348, 540)):
            raise UploadRejected("bad_magic", "File does not have a NIfTI-1 or NIfTI-2 header.")
    elif kind == "dicom_zip" and head[:4] != b"PK\x03\x04":
        raise UploadRejected("bad_magic", "File is not a ZIP archive.")


def safe_extract(zip_path: Path, dest: Path, settings: Settings) -> list[Path]:
    try:
        zf = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile as exc:
        raise UploadRejected("bad_zip", "The ZIP archive is corrupt.") from exc
    with zf:
        members = [m for m in zf.infolist() if not m.is_dir()]
        if len(members) > settings.max_zip_members:
            raise UploadRejected("too_many_files", f"Archive has {len(members)} files; the limit is {settings.max_zip_members}.")
        total = sum(m.file_size for m in members)
        if total > settings.max_uncompressed_mb * 1024 * 1024:
            raise UploadRejected("archive_too_large", "Archive expands beyond the uncompressed size limit.", status_code=413)
        compressed = sum(m.compress_size for m in members) or 1
        if total / compressed > MAX_COMPRESSION_RATIO:
            raise UploadRejected("suspicious_archive", "Archive compression ratio is implausibly high.")
        out = []
        for i, m in enumerate(members):
            p = PurePosixPath(m.filename)
            if p.is_absolute() or ".." in p.parts or "\\" in m.filename:
                raise UploadRejected("unsafe_path", "Archive contains an unsafe path.")
            if stat.S_ISLNK(m.external_attr >> 16):
                raise UploadRejected("unsafe_path", "Archive contains a symbolic link.")
            target = dest / f"{i:06d}.dcm"  # never reuse member names
            with zf.open(m) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            out.append(target)
        return out


def _deidentify_series(files: list[Path], out_dir: Path, settings: Settings) -> dict[str, Any]:
    profile = load_profile(settings.configs_dir / "deid_profile.yaml")
    salt = settings.salt()
    phi_seen: Counter[str] = Counter()
    report: dict[str, Any] = {"files_read": 0, "files_skipped_not_dicom": 0, "warnings": []}
    actions: dict[str, set[str]] = {"removed": set(), "blanked": set(), "pseudonymized": set(), "uids_remapped": set()}
    for i, path in enumerate(files):
        try:
            ds = pydicom.dcmread(str(path), force=False)
        except Exception:  # not DICOM; ignore quietly but count it
            report["files_skipped_not_dicom"] += 1
            continue
        report["files_read"] += 1
        for keyword in PHI_KEYWORDS:
            if keyword in ds and str(ds.get(keyword, "")).strip():
                phi_seen[keyword] += 1
        r = deidentify(ds, salt, profile)
        for key in actions:
            actions[key].update(getattr(r, key))
        for w in r.warnings:
            if w not in report["warnings"]:
                report["warnings"].append(w)
        ds.save_as(str(out_dir / f"{i:06d}.dcm"), enforce_file_format=False)
    report.update({k: sorted(v) for k, v in actions.items()})
    report["phi_keywords_present_in_source"] = sorted(phi_seen)
    report["private_tags_removed"] = bool(profile.get("remove_private_tags", True))
    if report["files_read"] == 0:
        raise UploadRejected("no_dicom", "The archive contains no readable DICOM files.")
    return report


def ingest(upload: Path, filename: str, *, modality: str | None, settings: Settings, store: CaseStore,
           uploaded_by: str) -> dict[str, Any]:
    kind = kind_from_name(filename)
    check_magic(upload, kind)
    work = Path(tempfile.mkdtemp(prefix="ingest-", dir=settings.tmp_dir))
    try:
        deid_report = None
        if kind.startswith("nifti"):
            if modality not in ALLOWED_MODALITIES:
                raise UploadRejected("modality_required", "NIfTI has no modality field; choose CT, MR or PT.")
            suffix = ".nii.gz" if kind == "nifti_gz" else ".nii"
            local = work / f"upload{suffix}"
            shutil.copyfile(upload, local)
            volume = load_nifti(local, modality=modality)
        else:
            raw, clean = work / "raw", work / "clean"
            raw.mkdir()
            clean.mkdir()
            files = safe_extract(upload, raw, settings)
            deid_report = _deidentify_series(files, clean, settings)
            shutil.rmtree(raw)
            volume = load_dicom_series(clean)
            if modality and modality != volume.modality:
                volume.metadata.setdefault("warnings", []).append(
                    f"Requested modality {modality} differs from DICOM Modality {volume.modality}; using DICOM value.")

        case_id = f"UP-{secrets.token_hex(4).upper()}"
        tp_id = "T0"
        save_nifti(volume.with_array(volume.array), store.case_dir(case_id) / tp_id / "image.nii.gz")
        case = {
            "case_id": case_id,
            "demo": False,
            "label": "UPLOADED RESEARCH DATA",
            "title": f"Uploaded {volume.modality} volume",
            "cancer_type": "not stated",
            "body_region": "not stated",
            "modality": volume.modality,
            "timepoints": [{"tp_id": tp_id, "label": "Scan 1", "days_from_baseline": 0, "modality": volume.modality,
                            "geometry": volume.geometry()}],
            "reference": {"available": False, "kind": None, "note": "Add a manual annotation to enable comparison."},
            "structures": {"available": False},
            "clinical": {"age_years": None},
            "metadata": {k: v for k, v in volume.metadata.items() if k != "warnings"},
            "warnings": volume.metadata.get("warnings", []),
            "deidentification": deid_report,
            "provenance": {"source": "upload", "source_format": volume.metadata.get("source_format"),
                           "uploaded_by": uploaded_by, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")},
        }
        store.write_case(case)
        return case
    finally:
        shutil.rmtree(work, ignore_errors=True)
