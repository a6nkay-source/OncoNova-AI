"""Read-only import of research dose grids (DICOM RTDOSE or NIfTI).

Imported dose is resampled onto the image grid and displayed as stored. OncoNova
does not verify the dose, its units or the plan that produced it, and it never
writes or sends plan data anywhere.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk

from onconova.errors import ImagingError
from onconova.imaging.io import load_nifti
from onconova.imaging.registration import resample_to
from onconova.imaging.volume import Volume


def load_rtdose(path: Path) -> Volume:
    try:
        ds = pydicom.dcmread(str(path))
    except Exception as exc:  # pydicom raises several types for malformed files
        raise ImagingError("unreadable_dose", "The file could not be read as DICOM.") from exc
    if str(ds.get("Modality", "")) != "RTDOSE":
        raise ImagingError("not_rtdose", f"Expected Modality RTDOSE, found {ds.get('Modality', 'none')!r}.")
    for keyword in ("DoseGridScaling", "GridFrameOffsetVector", "ImagePositionPatient", "ImageOrientationPatient", "PixelSpacing"):
        if keyword not in ds:
            raise ImagingError("incomplete_rtdose", f"RTDOSE is missing {keyword}.")
    arr = ds.pixel_array.astype(np.float32) * float(ds.DoseGridScaling)
    if arr.ndim == 2:
        arr = arr[None]
    offsets = np.asarray(ds.GridFrameOffsetVector, dtype=float)
    if len(offsets) != arr.shape[0]:
        raise ImagingError("incomplete_rtdose", "GridFrameOffsetVector length does not match the number of frames.")
    gaps = np.diff(offsets)
    if len(gaps) and (np.any(gaps <= 0) or np.ptp(gaps) > 1e-3):
        raise ImagingError("nonuniform_dose_grid", "RTDOSE frames are not uniformly spaced in increasing order.")
    row, col = np.asarray(ds.ImageOrientationPatient, dtype=float).reshape(2, 3)
    normal = np.cross(row, col)
    origin = np.asarray(ds.ImagePositionPatient, dtype=float) + normal * offsets[0]
    image = sitk.GetImageFromArray(arr)
    dz = float(gaps[0]) if len(gaps) else float(ds.get("SliceThickness", 1.0) or 1.0)
    image.SetSpacing((float(ds.PixelSpacing[1]), float(ds.PixelSpacing[0]), dz))
    image.SetOrigin(tuple(origin))
    image.SetDirection(tuple(np.stack([row, col, normal], axis=1).ravel()))
    units = str(ds.get("DoseUnits", "not stated"))
    return Volume.from_sitk(image, modality="RTDOSE", metadata={
        "dose_units": units, "dose_type": str(ds.get("DoseType", "not stated")),
        "dose_summation_type": str(ds.get("DoseSummationType", "not stated")), "source_format": "DICOM RTDOSE",
    }, dtype=np.float32)


def import_dose(path: Path, reference: Volume) -> tuple[np.ndarray, dict]:
    """Load a dose file and resample it onto ``reference``. Returns (dose, info)."""
    name = path.name.lower()
    if name.endswith((".nii", ".nii.gz")):
        dose_vol = load_nifti(path)
        dose_vol.metadata["dose_units"] = "as stored (NIfTI has no dose-unit field)"
    else:
        dose_vol = load_rtdose(path)
    resampled = resample_to(reference, dose_vol, default_value=0.0).astype(np.float32)
    coverage = float(np.mean(resampled > 0))
    info = {
        "units": dose_vol.metadata.get("dose_units"),
        "source_format": dose_vol.metadata.get("source_format"),
        "dose_type": dose_vol.metadata.get("dose_type"),
        "summation_type": dose_vol.metadata.get("dose_summation_type"),
        "native_geometry": dose_vol.geometry(),
        "nonzero_fraction_of_image_grid": round(coverage, 4),
        "label": "IMPORTED RESEARCH DOSE - displayed as stored, not verified by OncoNova",
    }
    return resampled, info
