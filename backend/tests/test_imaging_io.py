"""Image ingestion, orientation and geometry."""

import nibabel as nib
import numpy as np
import pytest

from onconova.errors import ImagingError
from onconova.imaging.io import load_dicom_series, load_nifti, save_nifti
from onconova.imaging.normalize import normalize_for_model, window
from onconova.imaging.volume import Volume

from conftest import PHI_MRN, write_ct_series


def test_nifti_roundtrip_preserves_geometry(tmp_path):
    arr = np.random.default_rng(0).normal(size=(10, 12, 14)).astype(np.float32)
    vol = Volume(arr, spacing=(0.8, 1.1, 3.0), origin=(-10.0, 20.0, 5.5), modality="CT")
    save_nifti(vol, tmp_path / "v.nii.gz")
    back = load_nifti(tmp_path / "v.nii.gz", modality="CT")
    assert back.shape == (10, 12, 14)
    np.testing.assert_allclose(back.spacing, (0.8, 1.1, 3.0), atol=1e-5)
    np.testing.assert_allclose(back.origin, (-10.0, 20.0, 5.5), atol=1e-4)
    np.testing.assert_allclose(back.array, arr, atol=1e-5)


def test_ras_nifti_is_reoriented_without_moving_anatomy(tmp_path):
    """A voxel keeps its physical (patient) position after reorientation to LPS."""
    data = np.zeros((20, 16, 12), dtype=np.float32)  # nibabel order: i, j, k
    i, j, k = 3, 11, 7
    data[i, j, k] = 100.0
    affine = np.diag([2.0, 1.5, 4.0, 1.0])  # RAS+ voxel-to-world
    affine[:3, 3] = [-15.0, 30.0, -8.0]
    nib.save(nib.Nifti1Image(data, affine), tmp_path / "ras.nii.gz")
    ras = affine @ np.array([i, j, k, 1.0])
    expected_lps = np.array([-ras[0], -ras[1], ras[2]])

    vol = load_nifti(tmp_path / "ras.nii.gz", modality="MR")
    idx = np.argwhere(vol.array == vol.array.max())[0]
    np.testing.assert_allclose(vol.index_to_physical(idx), expected_lps, atol=1e-3)
    np.testing.assert_allclose(sorted(vol.spacing), sorted([2.0, 1.5, 4.0]), atol=1e-5)
    np.testing.assert_allclose(vol.direction_matrix, np.eye(3), atol=1e-6)
    assert vol.metadata["reoriented_to"] == "LPS"


def test_rejects_4d_and_tiny_volumes(tmp_path):
    nib.save(nib.Nifti1Image(np.zeros((8, 8, 8, 4), np.float32), np.eye(4)), tmp_path / "4d.nii.gz")
    with pytest.raises(ImagingError) as e:
        load_nifti(tmp_path / "4d.nii.gz")
    assert e.value.code == "unsupported_dimension"
    nib.save(nib.Nifti1Image(np.zeros((64, 64, 2), np.float32), np.eye(4)), tmp_path / "thin.nii.gz")
    with pytest.raises(ImagingError) as e:
        load_nifti(tmp_path / "thin.nii.gz")
    assert e.value.code == "volume_too_small"


def test_rejects_non_nifti_content(tmp_path):
    bad = tmp_path / "fake.nii.gz"
    bad.write_bytes(b"not an image at all")
    with pytest.raises(ImagingError) as e:
        load_nifti(bad)
    assert e.value.code == "unreadable_image"
    with pytest.raises(ImagingError) as e:
        load_nifti(tmp_path / "missing.nii")
    assert e.value.code == "file_not_found"


def test_dicom_series_geometry_and_hu(tmp_path):
    write_ct_series(tmp_path / "ct", n=12, dz=2.5, px=0.7)
    vol = load_dicom_series(tmp_path / "ct")
    assert vol.modality == "CT"
    assert vol.shape == (12, 32, 32)
    np.testing.assert_allclose(vol.spacing, (0.7, 0.7, 2.5), atol=1e-4)
    assert vol.array.min() == pytest.approx(0.0)  # rescale intercept applied: stored 1024 -> 0 HU
    assert vol.array.max() == pytest.approx(500.0)
    meta = vol.metadata["dicom"]
    assert "PatientName" in meta["phi_present"] and "PatientID" in meta["phi_present"]
    assert PHI_MRN not in str(meta)  # identifying values are never copied into metadata
    assert meta["technical"]["Manufacturer"] == "TestVendor"


def test_dicom_series_rejects_duplicate_and_uneven_slices(tmp_path):
    write_ct_series(tmp_path / "dup", positions=[0, 2.5, 2.5, 5.0, 7.5, 10.0])
    with pytest.raises(ImagingError) as e:
        load_dicom_series(tmp_path / "dup")
    assert e.value.code == "duplicate_slices"
    write_ct_series(tmp_path / "gap", positions=[0, 2.5, 5.0, 10.0, 12.5, 15.0])
    with pytest.raises(ImagingError) as e:
        load_dicom_series(tmp_path / "gap")
    assert e.value.code == "nonuniform_slice_spacing"


def test_empty_directory_has_informative_error(tmp_path):
    with pytest.raises(ImagingError) as e:
        load_dicom_series(tmp_path)
    assert e.value.code == "no_dicom_series"


def test_modality_aware_normalisation():
    ct = Volume(np.array([[[-2000.0, -1000.0, 400.0, 3000.0]] * 8] * 8, dtype=np.float32), (1, 1, 1), modality="CT")
    x, how = normalize_for_model(ct)
    assert x.min() == 0.0 and x.max() == 1.0 and "HU" in how
    mr = Volume(np.random.default_rng(1).gamma(2.0, 100.0, (8, 8, 8)).astype(np.float32), (1, 1, 1), modality="MR")
    x, how = normalize_for_model(mr)
    assert "z-scored" in how and abs(float(x[x > x.min()].mean())) < 1.0
    assert window(np.array([-1350.0, -600.0, 150.0]), -600, 1500).tolist() == [0.0, 0.5, 1.0]
