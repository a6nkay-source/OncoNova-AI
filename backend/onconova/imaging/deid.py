"""Configurable DICOM de-identification.

This is a pragmatic subset of the DICOM PS3.15 Basic Application Level
Confidentiality Profile, driven by configs/deid_profile.yaml. It is a safeguard
for a research prototype, not a certified de-identification tool: review the
output before sharing data and check pixel data for burned-in text.
"""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pydicom
import yaml

DEFAULT_PROFILE: dict[str, Any] = {
    "remove": [
        "PatientBirthDate", "PatientAddress", "PatientTelephoneNumbers", "OtherPatientIDs",
        "OtherPatientNames", "PatientMotherBirthName", "MedicalRecordLocator", "ReferringPhysicianName",
        "PerformingPhysicianName", "OperatorsName", "InstitutionAddress", "StationName",
        "DeviceSerialNumber", "PatientComments", "AdditionalPatientHistory",
    ],
    "blank": ["InstitutionName", "StudyTime", "SeriesTime", "AcquisitionTime", "ContentTime"],
    "pseudonymize": ["PatientID", "PatientName", "AccessionNumber", "StudyID"],
    "remap_uids": ["StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID", "FrameOfReferenceUID"],
    "remove_dates": ["StudyDate", "SeriesDate", "AcquisitionDate", "ContentDate"],
    "remove_private_tags": True,
}


@dataclass
class DeidReport:
    removed: list[str] = field(default_factory=list)
    blanked: list[str] = field(default_factory=list)
    pseudonymized: list[str] = field(default_factory=list)
    uids_remapped: list[str] = field(default_factory=list)
    private_tags_removed: bool = False
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def load_profile(path: Path | None) -> dict[str, Any]:
    if path and path.exists():
        loaded = yaml.safe_load(path.read_text()) or {}
        return {**DEFAULT_PROFILE, **loaded}
    return dict(DEFAULT_PROFILE)


def pseudonym(value: str, salt: str, prefix: str = "ONV") -> str:
    digest = hmac.new(salt.encode(), value.encode(), hashlib.sha256).hexdigest()[:12].upper()
    return f"{prefix}-{digest}"


def remap_uid(uid: str, salt: str) -> str:
    # 2.25.<decimal of 128-bit hash> is the standard form for UUID-derived UIDs.
    digest = hmac.new(salt.encode(), uid.encode(), hashlib.sha256).digest()[:16]
    return f"2.25.{int.from_bytes(digest, 'big')}"


def deidentify(ds: pydicom.Dataset, salt: str, profile: dict[str, Any] | None = None) -> DeidReport:
    """De-identify ``ds`` in place and report what changed (keywords only, never values)."""
    profile = profile or DEFAULT_PROFILE
    report = DeidReport()
    for keyword in profile.get("remove", []) + profile.get("remove_dates", []):
        if keyword in ds:
            delattr(ds, keyword)
            report.removed.append(keyword)
    for keyword in profile.get("blank", []):
        if keyword in ds:
            setattr(ds, keyword, "")
            report.blanked.append(keyword)
    for keyword in profile.get("pseudonymize", []):
        if keyword in ds and str(ds.get(keyword, "")).strip():
            setattr(ds, keyword, pseudonym(str(ds.get(keyword)), salt))
            report.pseudonymized.append(keyword)
    for keyword in profile.get("remap_uids", []):
        if keyword in ds:
            setattr(ds, keyword, remap_uid(str(ds.get(keyword)), salt))
            report.uids_remapped.append(keyword)
    if getattr(ds, "file_meta", None) is not None and "MediaStorageSOPInstanceUID" in ds.file_meta:
        ds.file_meta.MediaStorageSOPInstanceUID = remap_uid(str(ds.file_meta.MediaStorageSOPInstanceUID), salt)
    if profile.get("remove_private_tags", True):
        ds.remove_private_tags()
        report.private_tags_removed = True
    ds.PatientIdentityRemoved = "YES"
    ds.DeidentificationMethod = "OncoNova research profile (subset of DICOM PS3.15 Basic Profile)"
    if str(ds.get("BurnedInAnnotation", "")).upper() == "YES":
        report.warnings.append("BurnedInAnnotation=YES: pixel data may contain identifying text.")
    return report
