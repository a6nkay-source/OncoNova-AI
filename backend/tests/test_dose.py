"""Research dose visualisation: DVH maths, synthetic illustration labels, RTDOSE import."""

import numpy as np
import pydicom
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, RTDoseStorage, generate_uid

from onconova.dose.dvh import cumulative_dvh, dose_at_volume, dvh_report, volume_at_dose
from onconova.dose.rtdose import import_dose
from onconova.dose.synthetic import SCENARIOS, illustrate
from onconova.errors import ImagingError
from onconova.imaging.volume import Volume


def test_cumulative_dvh_is_monotone():
    rng = np.random.default_rng(0)
    dose = rng.uniform(0, 1.2, (20, 20, 20))
    mask = np.ones_like(dose, bool)
    bins = np.linspace(0, 1.3, 50)
    curve = cumulative_dvh(dose, mask, bins)
    assert curve[0] == 1.0 and curve[-1] == 0.0
    assert np.all(np.diff(curve) <= 1e-12)


def test_uniform_dose_dvh_is_a_step():
    dose = np.full((5, 5, 5), 0.6)
    mask = np.ones_like(dose, bool)
    curve = cumulative_dvh(dose, mask, np.array([0.0, 0.59, 0.6, 0.61]))
    assert curve.tolist() == [1.0, 1.0, 1.0, 0.0]
    assert dose_at_volume(dose, mask, 0.95) == pytest.approx(0.6)
    assert volume_at_dose(dose, mask, 0.6) == 1.0


def test_dvh_report_skips_empty_structures():
    dose = np.ones((4, 4, 4))
    rep = dvh_report(dose, {"t": np.ones((4, 4, 4), bool), "empty": np.zeros((4, 4, 4), bool)}, "t", 0.001)
    assert rep["skipped_empty"] == ["empty"] and "D95" in rep["stats"]["t"]


def test_synthetic_illustration_is_relative_and_labelled(phantom_case):
    _, renders = phantom_case
    image, lesions, structures = renders[0]
    body = structures.array > 0
    out = illustrate(image, lesions.array == 1, body, SCENARIOS["S2"])
    assert np.median(out["dose"][out["ptv"]]) == pytest.approx(1.0, abs=1e-3)
    assert out["dose"][~body].max() == 0.0
    assert "NON-CLINICAL" in out["warnings"][0] and "relative" in out["units"]


def test_more_beams_spread_more_low_dose(phantom_case):
    """The qualitative trade-off the simulator is meant to illustrate."""
    _, renders = phantom_case
    image, lesions, structures = renders[0]
    body = structures.array > 0
    v5 = {s: volume_at_dose(illustrate(image, lesions.array == 1, body, SCENARIOS[s])["dose"], body, 0.05) for s in ("S1", "S3")}
    assert v5["S3"] > v5["S1"]


def _rtdose(path, origin, spacing, shape, value=2.0, modality="RTDOSE"):
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = RTDoseStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID, ds.SOPInstanceUID = RTDoseStorage, meta.MediaStorageSOPInstanceUID
    ds.Modality = modality
    ds.Rows, ds.Columns = shape[1], shape[2]
    ds.NumberOfFrames = shape[0]
    ds.PixelSpacing = [spacing[1], spacing[0]]
    ds.ImagePositionPatient = list(origin)
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.GridFrameOffsetVector = [i * spacing[2] for i in range(shape[0])]
    ds.FrameIncrementPointer = (0x3004, 0x000C)
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = ds.BitsStored = 32
    ds.HighBit = 31
    ds.PixelRepresentation = 0
    ds.DoseGridScaling = 0.001
    ds.DoseUnits, ds.DoseType, ds.DoseSummationType = "GY", "PHYSICAL", "PLAN"
    ds.PixelData = np.full(shape, int(value / 0.001), dtype=np.uint32).tobytes()
    ds.save_as(str(path), enforce_file_format=True)
    return path


def test_rtdose_import_resamples_onto_image_grid(tmp_path):
    image = Volume(np.zeros((10, 20, 20), np.float32), (2.0, 2.0, 3.0), (0.0, 0.0, 0.0), modality="CT")
    path = _rtdose(tmp_path / "dose.dcm", origin=(-4.0, -4.0, -3.0), spacing=(4.0, 4.0, 3.0), shape=(14, 14, 14))
    dose, info = import_dose(path, image)
    assert dose.shape == image.shape
    assert dose.mean() == pytest.approx(2.0, rel=1e-3)
    assert info["units"] == "GY" and "not verified" in info["label"]


def test_non_rtdose_is_rejected(tmp_path):
    image = Volume(np.zeros((10, 10, 10), np.float32), (1, 1, 1), modality="CT")
    path = _rtdose(tmp_path / "x.dcm", (0, 0, 0), (1, 1, 1), (4, 4, 4), modality="CT")
    with pytest.raises(ImagingError) as e:
        import_dose(path, image)
    assert e.value.code == "not_rtdose"
    assert pydicom.dcmread(str(path)).Modality == "CT"
