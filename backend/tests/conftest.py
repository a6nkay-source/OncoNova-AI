from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pydicom
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

from onconova.config import REPO_ROOT, Settings
from onconova.synthetic import cases as demo_cases
from onconova.synthetic.phantom import random_patient_spec, render_timepoint

PHI_NAME = "Doe^Jane"
PHI_MRN = "MRN-0042-XYZ"


def make_settings(tmp_path: Path, **overrides) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        artifacts_dir=tmp_path / "artifacts",
        configs_dir=REPO_ROOT / "configs",
        pseudonym_salt="test-salt",
        **overrides,
    )


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    s = make_settings(tmp_path)
    s.ensure_dirs()
    return s


@pytest.fixture(scope="session")
def phantom_case():
    """A three-timepoint growing phantom with a node-region lesion and a new nodule at T2."""
    spec = random_patient_spec("TEST-001", 101, n_timepoints=3, trajectory="growing", node=True, new_lesion=True)
    return spec, [render_timepoint(spec, tp) for tp in spec.timepoints]


@pytest.fixture
def small_demo(monkeypatch):
    """Limit demo materialisation to one multi-scan and one single-scan case for speed."""
    monkeypatch.setattr(demo_cases, "DEMO_CASES", [demo_cases.DEMO_CASES[0], demo_cases.DEMO_CASES[4]])


@pytest.fixture
def client(settings, small_demo):
    from fastapi.testclient import TestClient

    from onconova.api.app import create_app

    with TestClient(create_app(settings)) as c:
        yield c


def write_ct_series(directory: Path, *, n: int = 12, dz: float = 2.5, px: float = 0.7, size: int = 32,
                    positions: list[float] | None = None) -> Path:
    """Write a minimal CT DICOM series with identifying attributes (for de-identification tests)."""
    directory.mkdir(parents=True, exist_ok=True)
    study, series, frame = generate_uid(), generate_uid(), generate_uid()
    zs = positions if positions is not None else [i * dz for i in range(n)]
    for i, z in enumerate(zs):
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = CTImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        path = directory / f"slice_{i:03d}.dcm"
        ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SOPClassUID = CTImageStorage
        ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.StudyInstanceUID, ds.SeriesInstanceUID, ds.FrameOfReferenceUID = study, series, frame
        ds.Modality = "CT"
        ds.PatientName = PHI_NAME
        ds.PatientID = PHI_MRN
        ds.PatientBirthDate = "19600101"
        ds.InstitutionName = "General Hospital"
        ds.StudyDate = "20240102"
        ds.Manufacturer = "TestVendor"
        ds.add_new(0x00091010, "LO", "private value")
        ds.Rows = ds.Columns = size
        ds.PixelSpacing = [px, px]
        ds.SliceThickness = dz
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.ImagePositionPatient = [0.0, 0.0, float(z)]
        ds.InstanceNumber = i + 1
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1
        ds.RescaleIntercept = -1024
        ds.RescaleSlope = 1
        arr = np.full((size, size), 1024, dtype=np.int16)  # 0 HU
        arr[10:14, 10:14] = 1024 + 500
        ds.PixelData = arr.tobytes()
        ds.save_as(str(path), enforce_file_format=True)
    return directory


def zip_dir(directory: Path, out: Path) -> Path:
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(directory.glob("*.dcm")):
            zf.write(p, arcname=f"series/{p.name}")
    return out


def read_dicom(path: Path) -> pydicom.Dataset:
    return pydicom.dcmread(str(path))
