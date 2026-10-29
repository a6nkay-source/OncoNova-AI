"""Coordinate correctness end to end: every voxel keeps its patient (LPS) position.

Each test writes markers with distinct values at known patient positions, loads the
file through the normal loader, finds each marker again and checks where the loaded
geometry says it is. Non-square pixels and matrices catch swapped axes; a different
marker position on every slice catches flips and slice-order mistakes.
"""

import itertools

import nibabel as nib
import numpy as np
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

from onconova.api.binary import EXPOSED_HEADERS
from onconova.imaging.io import load_dicom_series, load_nifti

ROWS, COLS = 20, 28
ROW_MM, COL_MM = 0.7, 0.9  # DICOM PixelSpacing: between rows, between columns


def write_series(directory, *, iop, positions, file_order=None, instance_numbers=None):
    """Write a CT series; slice i gets one marker pixel of value 100 + 10 i HU.

    Returns the expected LPS position of each marker, keyed by its HU value.
    """
    directory.mkdir(parents=True, exist_ok=True)
    row_dir, col_dir = np.asarray(iop[:3], float), np.asarray(iop[3:], float)
    study, series, frame = generate_uid(), generate_uid(), generate_uid()
    order = file_order if file_order is not None else range(len(positions))
    expected = {}
    for name_idx, i in enumerate(order):
        ipp = np.asarray(positions[i], float)
        r, c = 2 + (3 * i) % (ROWS - 4), 3 + (5 * i) % (COLS - 6)
        expected[100 + 10 * i] = ipp + c * COL_MM * row_dir + r * ROW_MM * col_dir

        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = CTImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        path = directory / f"f{name_idx:03d}.dcm"  # file names in a different order from the slices
        ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SOPClassUID, ds.SOPInstanceUID = CTImageStorage, meta.MediaStorageSOPInstanceUID
        ds.StudyInstanceUID, ds.SeriesInstanceUID, ds.FrameOfReferenceUID = study, series, frame
        ds.Modality = "CT"
        ds.Rows, ds.Columns = ROWS, COLS
        ds.PixelSpacing = [ROW_MM, COL_MM]
        ds.ImageOrientationPatient = [float(v) for v in iop]
        ds.ImagePositionPatient = [float(v) for v in ipp]
        ds.InstanceNumber = instance_numbers[i] if instance_numbers is not None else i + 1
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1
        ds.RescaleIntercept, ds.RescaleSlope = -1024, 1
        arr = np.full((ROWS, COLS), 1024, dtype=np.int16)  # 0 HU
        arr[r, c] = 1024 + 100 + 10 * i
        ds.PixelData = arr.tobytes()
        ds.save_as(str(path), enforce_file_format=True)
    return expected


def assert_markers_in_place(vol, expected):
    for value, lps in expected.items():
        hits = np.argwhere(np.isclose(vol.array, value))
        assert len(hits) == 1, f"marker {value} found {len(hits)} times"
        np.testing.assert_allclose(vol.index_to_physical(hits[0]), lps, atol=1e-3, err_msg=f"marker {value}")


def normal(iop):
    return np.cross(np.asarray(iop[:3], float), np.asarray(iop[3:], float))


AXIAL = (1, 0, 0, 0, 1, 0)
c15, s15 = np.cos(np.radians(15)), np.sin(np.radians(15))
ORIENTATIONS = {
    "axial": AXIAL,
    "axial, rows run right-to-left": (-1, 0, 0, 0, 1, 0),
    "axial, columns run posterior-to-anterior": (1, 0, 0, 0, -1, 0),
    "coronal": (1, 0, 0, 0, 0, -1),
    "sagittal": (0, 1, 0, 0, 0, -1),
    "oblique 15 degrees about z": (c15, s15, 0, -s15, c15, 0),
}


@pytest.mark.parametrize("name", list(ORIENTATIONS))
@pytest.mark.parametrize("step", [2.5, -2.5], ids=["slices along +normal", "slices along -normal"])
def test_dicom_markers_keep_their_patient_position(tmp_path, name, step):
    iop = ORIENTATIONS[name]
    n = 10
    origin = np.array([-40.0, 25.0, 110.0])
    positions = [origin + i * step * normal(iop) for i in range(n)]
    expected = write_series(tmp_path / "s", iop=iop, positions=positions)
    vol = load_dicom_series(tmp_path / "s")
    assert_markers_in_place(vol, expected)
    assert sorted(np.round(vol.spacing, 4)) == sorted([ROW_MM, COL_MM, 2.5])
    assert vol.metadata["slice_gap_mm"] == pytest.approx(2.5)
    assert vol.is_oblique == name.startswith("oblique")
    if not vol.is_oblique:
        np.testing.assert_allclose(vol.direction_matrix, np.eye(3), atol=1e-6)


@pytest.mark.parametrize(
    "file_order, instance_numbers",
    [
        ([7, 2, 9, 0, 4, 1, 8, 3, 6, 5], [7, 2, 9, 0, 4, 1, 8, 3, 6, 5]),  # shuffled files and instance numbers
        (list(range(9, -1, -1)), list(range(10, 0, -1))),  # reversed: instance 1 is the top slice
        (list(range(10)), [5, 3, 9, 1, 2, 10, 4, 8, 6, 7]),  # instance numbers disagree with position
    ],
    ids=["shuffled", "reversed", "instance numbers misleading"],
)
def test_dicom_slices_are_ordered_by_position_not_file_or_instance(tmp_path, file_order, instance_numbers):
    positions = [np.array([0.0, 0.0, 2.5 * i]) for i in range(10)]
    expected = write_series(tmp_path / "s", iop=AXIAL, positions=positions, file_order=file_order,
                            instance_numbers=instance_numbers)
    vol = load_dicom_series(tmp_path / "s")
    assert_markers_in_place(vol, expected)
    # LPS: array z increases towards superior, so marker values climb with z.
    zs = [int(np.argwhere(np.isclose(vol.array, v))[0][0]) for v in sorted(expected)]
    assert zs == list(range(10))


SIGNED_PERMUTATIONS = [
    (perm, signs) for perm in itertools.permutations(range(3)) for signs in itertools.product((1, -1), repeat=3)
]


@pytest.mark.parametrize("perm, signs", SIGNED_PERMUTATIONS, ids=lambda p: str(p))
def test_nifti_any_axis_order_and_flip_keeps_patient_position(tmp_path, perm, signs):
    """All 48 axis-aligned voxel orders load to identity direction with markers unmoved."""
    spacing = np.array([0.8, 1.3, 2.5])
    axes = np.zeros((3, 3))
    for i, (p, s) in enumerate(zip(perm, signs)):
        axes[p, i] = s * spacing[i]  # voxel axis i runs along world axis p
    affine = np.eye(4)
    affine[:3, :3] = axes
    affine[:3, 3] = [12.0, -30.0, 47.0]
    data = np.zeros((10, 12, 14), np.float32)
    markers = {100.0: (1, 2, 3), 200.0: (8, 3, 11), 300.0: (4, 10, 1)}
    expected = {}
    for value, ijk in markers.items():
        data[ijk] = value
        ras = affine @ np.array([*ijk, 1.0])
        expected[value] = np.array([-ras[0], -ras[1], ras[2]])
    nib.save(nib.Nifti1Image(data, affine), tmp_path / "v.nii.gz")

    vol = load_nifti(tmp_path / "v.nii.gz", modality="MR")
    assert_markers_in_place(vol, expected)
    np.testing.assert_allclose(vol.direction_matrix, np.eye(3), atol=1e-6)
    # Spacing follows the axes to their new order: x, y, z are L, P, S.
    world_spacing = np.zeros(3)
    for i, p in enumerate(perm):
        world_spacing[p] = spacing[i]
    np.testing.assert_allclose(vol.spacing, world_spacing, atol=1e-5)


def test_oblique_nifti_keeps_patient_position_and_warns(tmp_path):
    t = np.radians(20)
    rot = np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]])
    affine = np.eye(4)
    affine[:3, :3] = rot @ np.diag([1.0, 1.2, 2.0])
    affine[:3, 3] = [-5.0, 8.0, 3.0]
    data = np.zeros((10, 12, 14), np.float32)
    data[2, 9, 5] = 100.0
    nib.save(nib.Nifti1Image(data, affine), tmp_path / "o.nii.gz")
    ras = affine @ np.array([2, 9, 5, 1.0])

    vol = load_nifti(tmp_path / "o.nii.gz", modality="MR")
    assert_markers_in_place(vol, {100.0: np.array([-ras[0], -ras[1], ras[2]])})
    assert vol.is_oblique
    assert any("Oblique" in w for w in vol.metadata["warnings"])


GEOMETRY_HEADERS = ("x-shape", "x-spacing", "x-origin", "x-direction")


def test_overlays_carry_their_image_frame_and_geometry(client):
    img = client.get("/api/cases/DEMO-001/timepoints/T0/image")
    ref = client.get("/api/cases/DEMO-001/timepoints/T0/maps/reference")
    other = client.get("/api/cases/DEMO-001/timepoints/T1/maps/reference")
    assert img.status_code == ref.status_code == other.status_code == 200
    assert img.headers["x-frame"] == ref.headers["x-frame"] == "DEMO-001/T0"
    assert other.headers["x-frame"] == "DEMO-001/T1"  # same shape, different scan: the client must not mix them
    for h in GEOMETRY_HEADERS:
        assert ref.headers[h] == img.headers[h]
    assert "X-Frame" in EXPOSED_HEADERS  # readable by a cross-origin client too
    # Registered follow-ups are resampled into the baseline grid, so they carry the baseline frame.
    reg = client.get("/api/cases/DEMO-001/timepoints/T1/registered?kind=lesions")
    assert reg.headers["x-frame"] == "DEMO-001/T0"
    for h in GEOMETRY_HEADERS:
        assert reg.headers[h] == img.headers[h]


def test_a_map_on_a_different_grid_is_refused_not_drawn(client, settings):
    from onconova.imaging.io import load_nifti, save_nifti

    tp_dir = settings.cases_dir / "DEMO-001" / "T0"
    image = load_nifti(tp_dir / "image.nii.gz")
    shifted = image.with_array(np.zeros(image.shape, np.float32), origin=tuple(o + 5.0 for o in image.origin))
    save_nifti(shifted, tp_dir / "unc_demo-classical.nii.gz")
    r = client.get("/api/cases/DEMO-001/timepoints/T0/maps/uncertainty?model_id=demo-classical")
    assert r.status_code == 422 and r.json()["error"]["code"] == "geometry_mismatch"
