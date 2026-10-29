"""Load and validate NIfTI volumes and DICOM image series.

Every loader reorients to LPS (identity direction for axis-aligned scans) without
resampling, so voxel spacing and physical coordinates are preserved exactly. The
original direction matrix is kept in ``metadata`` for provenance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pydicom
import SimpleITK as sitk

from onconova.errors import ImagingError
from onconova.imaging.volume import Volume

SUPPORTED_MODALITIES = {"CT", "MR", "PT"}
NIFTI_SUFFIXES = (".nii", ".nii.gz")
MAX_VOXELS = 512 * 512 * 1200
MIN_AXIS = 8
MAX_SPACING_MM = 20.0

# Plausible stored-value ranges. Outside them the scan is still loaded, with a warning.
CT_HU_RANGE = (-1100.0, 4000.0)

# Technical DICOM attributes that are safe to show researchers. Anything that can
# identify a person is excluded here and reported only by keyword in ``phi_present``.
TECHNICAL_TAGS = (
    "Modality", "Manufacturer", "ManufacturerModelName", "BodyPartExamined", "SliceThickness",
    "KVP", "ConvolutionKernel", "ContrastBolusAgent", "MagneticFieldStrength", "RepetitionTime",
    "EchoTime", "FlipAngle", "MRAcquisitionType", "Units", "CorrectedImage", "DecayCorrection",
    "Rows", "Columns", "PixelSpacing", "ImageOrientationPatient", "PatientPosition",
    "RescaleSlope", "RescaleIntercept", "BitsStored", "PhotometricInterpretation",
)
PHI_KEYWORDS = (
    "PatientName", "PatientID", "PatientBirthDate", "PatientAddress", "PatientTelephoneNumbers",
    "OtherPatientIDs", "OtherPatientNames", "PatientMotherBirthName", "MedicalRecordLocator",
    "ReferringPhysicianName", "PerformingPhysicianName", "OperatorsName", "InstitutionName",
    "InstitutionAddress", "StationName", "AccessionNumber", "StudyDate", "SeriesDate",
    "AcquisitionDate", "ContentDate", "StudyTime", "StudyID", "DeviceSerialNumber",
)


def has_nifti_suffix(path: Path | str) -> bool:
    name = str(path).lower()
    return name.endswith(NIFTI_SUFFIXES)


def validate_volume(volume: Volume) -> list[str]:
    """Raise ImagingError for unusable data; return warnings for unusual data."""
    arr = volume.array
    if arr.ndim != 3:
        raise ImagingError("unsupported_dimension", f"Expected a 3-D volume, got {arr.ndim}-D data.")
    if min(arr.shape) < MIN_AXIS:
        raise ImagingError(
            "volume_too_small",
            f"Volume shape {list(arr.shape)} has an axis shorter than {MIN_AXIS} voxels; "
            "a single slice or a 2-D image cannot be analysed in 3-D.",
        )
    if arr.size > MAX_VOXELS:
        raise ImagingError("volume_too_large", f"Volume has {arr.size:,} voxels; the limit is {MAX_VOXELS:,}.")
    spacing = np.asarray(volume.spacing, dtype=float)
    if not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ImagingError("invalid_spacing", f"Voxel spacing {spacing.tolist()} must be positive and finite.")
    if np.any(spacing > MAX_SPACING_MM):
        raise ImagingError("invalid_spacing", f"Voxel spacing {spacing.tolist()} mm exceeds {MAX_SPACING_MM} mm.")
    if not np.all(np.isfinite(arr)):
        raise ImagingError("non_finite_values", "The image contains NaN or infinite values.")

    warnings: list[str] = []
    if volume.is_oblique:
        warnings.append("Oblique acquisition: axes are not aligned with the patient; views show voxel axes.")
    if spacing.max() / spacing.min() > 4:
        warnings.append(f"Highly anisotropic voxels ({spacing.round(2).tolist()} mm); 3-D measurements are less precise.")
    if volume.modality == "CT":
        lo, hi = float(arr.min()), float(arr.max())
        if lo < CT_HU_RANGE[0] - 1000 or hi > CT_HU_RANGE[1] * 2:
            warnings.append(f"CT values span {lo:.0f} to {hi:.0f}; the data may not be in Hounsfield units.")
    return warnings


def _reorient(image: sitk.Image) -> sitk.Image:
    return sitk.DICOMOrient(image, "LPS")


def load_nifti(path: Path | str, *, modality: str = "UNKNOWN", is_label: bool = False) -> Volume:
    path = Path(path)
    if not path.exists():
        raise ImagingError("file_not_found", f"No file at {path.name}.", status_code=404)
    if not has_nifti_suffix(path):
        raise ImagingError("unsupported_format", "Only .nii and .nii.gz volumes are accepted as NIfTI.")
    try:
        image = sitk.ReadImage(str(path))
    except RuntimeError as exc:
        raise ImagingError("unreadable_image", "The file could not be parsed as a NIfTI volume.") from exc
    if image.GetDimension() != 3:
        raise ImagingError(
            "unsupported_dimension",
            f"{image.GetDimension()}-D NIfTI is not supported; split multi-sequence or 4-D data into 3-D volumes.",
        )
    if image.GetNumberOfComponentsPerPixel() != 1:
        raise ImagingError("unsupported_pixel_type", "Vector or RGB NIfTI volumes are not supported.")
    original_direction = list(image.GetDirection())
    image = _reorient(image)
    volume = Volume.from_sitk(
        image,
        modality=modality,
        metadata={"source_format": "NIfTI", "original_direction": original_direction, "reoriented_to": "LPS"},
        dtype=np.uint8 if is_label else np.float32,
    )
    volume.metadata["warnings"] = [] if is_label else validate_volume(volume)
    return volume


def save_nifti(volume: Volume, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(volume.to_sitk(), str(path), useCompression=True)


def load_like(path: Path | str, reference: Volume, *, is_label: bool) -> Volume:
    """Load a label or continuous map and require that it shares the reference image geometry."""
    overlay = load_nifti(path, is_label=is_label)
    if not overlay.same_geometry(reference):
        kind = "label map" if is_label else "map"
        raise ImagingError(
            "geometry_mismatch",
            f"The {kind} does not share the image's shape, spacing, origin and direction.",
            details={"image": reference.geometry(), "overlay": overlay.geometry()},
        )
    return overlay


def load_label_like(path: Path | str, reference: Volume) -> np.ndarray:
    return load_like(path, reference, is_label=True).array


def _slice_positions(headers: list[pydicom.Dataset]) -> np.ndarray:
    orientations = {tuple(round(float(v), 4) for v in h.ImageOrientationPatient) for h in headers}
    if len(orientations) > 1:
        raise ImagingError("inconsistent_orientation", "Slices in the series have different orientations.")
    row, col = np.asarray(headers[0].ImageOrientationPatient, dtype=float).reshape(2, 3)
    normal = np.cross(row, col)
    return np.sort(np.array([float(np.dot(normal, np.asarray(h.ImagePositionPatient, dtype=float))) for h in headers]))


def check_slice_spacing(headers: list[pydicom.Dataset]) -> float:
    """Return the slice spacing in mm; reject duplicate or unevenly spaced slices."""
    for h in headers:
        if "ImagePositionPatient" not in h or "ImageOrientationPatient" not in h:
            raise ImagingError("missing_geometry", "DICOM slices lack ImagePositionPatient/ImageOrientationPatient.")
    sizes = {(int(h.Rows), int(h.Columns)) for h in headers}
    if len(sizes) > 1:
        raise ImagingError("inconsistent_matrix", f"Slices have different matrix sizes: {sorted(sizes)}.")
    gaps = np.diff(_slice_positions(headers))
    if gaps.size == 0:
        raise ImagingError("volume_too_small", "A single DICOM slice cannot be analysed as a volume.")
    if np.any(gaps < 1e-3):
        raise ImagingError("duplicate_slices", "The series contains slices at the same position (duplicates or multiple phases).")
    if gaps.max() - gaps.min() > max(0.05, 0.01 * gaps.mean()):
        raise ImagingError(
            "nonuniform_slice_spacing",
            f"Slice spacing varies from {gaps.min():.2f} to {gaps.max():.2f} mm. Resample the series to "
            "uniform spacing before import.",
            details={"min_gap_mm": float(gaps.min()), "max_gap_mm": float(gaps.max())},
        )
    return float(gaps.mean())


def dicom_metadata(header: pydicom.Dataset) -> dict[str, Any]:
    technical: dict[str, Any] = {}
    for keyword in TECHNICAL_TAGS:
        if keyword in header:
            value = header.get(keyword)
            if isinstance(value, pydicom.multival.MultiValue):
                value = [float(v) if isinstance(v, (int, float, pydicom.valuerep.DSfloat)) else str(v) for v in value]
            elif isinstance(value, (pydicom.valuerep.DSfloat, pydicom.valuerep.IS)):
                value = float(value)
            else:
                value = str(value)
            technical[keyword] = value
    phi_present = [k for k in PHI_KEYWORDS if k in header and str(header.get(k, "")).strip()]
    return {
        "technical": technical,
        "phi_present": phi_present,
        "burned_in_annotation": str(header.get("BurnedInAnnotation", "not stated")),
    }


def load_dicom_series(directory: Path | str, series_uid: str | None = None) -> Volume:
    directory = Path(directory)
    reader = sitk.ImageSeriesReader()
    series_ids = list(reader.GetGDCMSeriesIDs(str(directory)) or [])
    if not series_ids:
        raise ImagingError("no_dicom_series", "No DICOM image series was found.")
    warnings: list[str] = []
    if series_uid is None:
        counts = {uid: len(reader.GetGDCMSeriesFileNames(str(directory), uid)) for uid in series_ids}
        series_uid = max(counts, key=counts.get)
        if len(series_ids) > 1:
            warnings.append(f"{len(series_ids)} series found; loaded the largest ({counts[series_uid]} slices).")
    elif series_uid not in series_ids:
        raise ImagingError("series_not_found", "The requested SeriesInstanceUID is not in the upload.")

    files = list(reader.GetGDCMSeriesFileNames(str(directory), series_uid))
    headers = [pydicom.dcmread(f, stop_before_pixels=True) for f in files]
    modality = str(headers[0].get("Modality", "UNKNOWN"))
    if modality not in SUPPORTED_MODALITIES:
        raise ImagingError(
            "unsupported_modality",
            f"Modality {modality!r} is not an image series this prototype can analyse "
            f"(supported: {', '.join(sorted(SUPPORTED_MODALITIES))}).",
        )
    slice_gap = check_slice_spacing(headers)
    reader.SetFileNames(files)
    try:
        image = reader.Execute()
    except RuntimeError as exc:
        raise ImagingError("unreadable_image", "The DICOM pixel data could not be decoded.") from exc
    original_direction = list(image.GetDirection())
    image = _reorient(image)
    meta = dicom_metadata(headers[0])
    if modality == "PT":
        warnings.append("PET values are shown as stored; SUV conversion is not implemented (relative uptake only).")
    volume = Volume.from_sitk(
        image,
        modality=modality,
        metadata={
            "source_format": "DICOM",
            "n_slices": len(files),
            "slice_gap_mm": round(slice_gap, 4),
            "original_direction": original_direction,
            "reoriented_to": "LPS",
            "dicom": meta,
        },
        dtype=np.float32,
    )
    volume.metadata["warnings"] = warnings + validate_volume(volume)
    if meta["phi_present"]:
        volume.metadata["warnings"].append(
            "Source headers contained identifying attributes; they were not stored (see de-identification report)."
        )
    if meta["burned_in_annotation"].upper() == "YES":
        volume.metadata["warnings"].append("BurnedInAnnotation=YES: pixel data may contain identifying text.")
    return volume
