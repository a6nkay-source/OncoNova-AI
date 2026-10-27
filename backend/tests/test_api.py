"""API validation, error envelopes, uploads, annotations, access control and end-to-end flow."""

import gzip
import io
import zipfile

import nibabel as nib
import numpy as np
import pytest
import yaml
from fastapi.testclient import TestClient

from onconova.api.app import create_app
from onconova.security import hash_key

from conftest import PHI_MRN, PHI_NAME, make_settings, write_ct_series, zip_dir


def nifti_bytes(shape=(16, 16, 16)) -> bytes:
    img = nib.Nifti1Image(np.random.default_rng(0).normal(0, 50, shape).astype(np.float32), np.diag([1.5, 1.5, 2.0, 1.0]))
    buf = io.BytesIO()
    buf.write(img.to_bytes())
    return gzip.compress(buf.getvalue())


def test_health_and_cases(client):
    assert client.get("/api/health").json()["status"] == "ok"
    cases = client.get("/api/cases").json()
    assert {c["case_id"] for c in cases} == {"DEMO-001", "DEMO-005"}
    assert all(c["demo"] and c["label"] == "DEMO DATA" for c in cases)


def test_image_endpoint_returns_geometry_headers(client):
    r = client.get("/api/cases/DEMO-001/timepoints/T0/image")
    shape = [int(v) for v in r.headers["x-shape"].split(",")]
    assert r.status_code == 200 and r.headers["x-dtype"] == "int16"
    assert len(r.content) == int(np.prod(shape)) * 2
    assert [float(v) for v in r.headers["x-spacing"].split(",")] == [2.2, 2.2, 2.5]
    assert r.headers["x-data-label"] == "DEMO DATA"


def test_error_envelope_is_safe(client):
    r = client.get("/api/cases/NOPE")
    body = r.json()
    assert r.status_code == 404 and body["error"]["code"] == "case_not_found" and body["request_id"]
    assert "Traceback" not in r.text
    r = client.get("/api/cases/bad%2E%2Eid/timepoints/T0/image")
    assert r.status_code in (400, 404)


def test_request_validation(client):
    r = client.post("/api/cases/DEMO-001/longitudinal", json={"source": "../../etc"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_request"
    r = client.post("/api/cases/DEMO-001/timepoints/T0/segmentation", json={"model_id": "Robert'); DROP"})
    assert r.status_code == 422


def test_full_pipeline(client):
    for tp in ("T0", "T1", "T2"):
        r = client.post(f"/api/cases/DEMO-001/timepoints/{tp}/segmentation", json={"model_id": "demo-classical"})
        assert r.status_code == 200, r.text
        assert r.json()["kind"] == "model_output" and r.json()["volume_kind"] == "measured_from_model_output"
    cmp_ = client.post("/api/cases/DEMO-001/timepoints/T0/comparison", json={"model_id": "demo-classical"}).json()
    assert cmp_["any_lesion"]["dice"] > 0.7 and cmp_["reference_kind"] == "synthetic generator ground truth"
    long_ = client.post("/api/cases/DEMO-001/longitudinal", json={"source": "pred:demo-classical"}).json()
    assert long_["status"] == "ok" and len(long_["timepoints"]) == 3
    dose = client.post("/api/cases/DEMO-001/dose", json={"scenario_id": "S1", "target_source": "reference"}).json()
    assert dose["kind"] == "simulated" and dose["not_a_prescription"] and "NON-CLINICAL" in dose["label"]
    m = client.get(f"/api/cases/DEMO-001/timepoints/T0/maps/dose?key={dose['dose_map']}")
    assert m.status_code == 200 and m.headers["x-data-label"] == "SIMULATED - NON-CLINICAL"
    summary = client.get("/api/cases/DEMO-001/summary").json()
    assert set(summary["sections"]) == {
        "A_imaging_observations", "B_segmentation_results", "C_changes_between_timepoints",
        "D_progression_model_output", "E_uncertainty_and_limitations", "F_research_radiation_simulation",
        "G_items_requiring_expert_review"}
    mesh = client.get("/api/cases/DEMO-001/timepoints/T0/mesh?source=reference&label=1").json()
    assert mesh["n_faces"] > 0


def test_unavailable_states(client):
    r = client.post("/api/cases/DEMO-005/longitudinal", json={"source": "reference"}).json()
    assert r["status"] == "unavailable" and "second timepoint" in r["missing_inputs"]
    r = client.post("/api/cases/DEMO-005/progression", json={"source": "reference"}).json()
    assert r["status"] == "unavailable" and "probability" not in r
    r = client.get("/api/cases/DEMO-001/timepoints/T0/segmentation?model_id=demo-unet").json()
    assert r["status"] == "unavailable"
    r = client.post("/api/cases/DEMO-001/timepoints/T0/segmentation", json={"model_id": "demo-unet"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "model_unavailable"  # no checkpoint in the test artifacts dir


def test_evaluations_empty_without_runs(client):
    assert client.get("/api/evaluations").json()["reports"] == []


def test_annotation_roundtrip_and_comparison(client):
    shape = [int(v) for v in client.get("/api/cases/DEMO-001/timepoints/T0/image").headers["x-shape"].split(",")]
    bad = client.put("/api/cases/DEMO-001/timepoints/T0/annotation", content=b"\0" * 10)
    assert bad.status_code == 400 and bad.json()["error"]["code"] == "bad_annotation_size"
    ann = np.zeros(shape, np.uint8)
    ann[30:40, 50:60, 50:60] = 1
    ok = client.put("/api/cases/DEMO-001/timepoints/T0/annotation", content=ann.tobytes())
    assert ok.status_code == 200 and ok.json()["voxels"] == 1000
    client.post("/api/cases/DEMO-001/timepoints/T0/segmentation", json={"model_id": "demo-classical"})
    cmp_ = client.post("/api/cases/DEMO-001/timepoints/T0/comparison", json={"model_id": "demo-classical", "reference": "annotation"})
    assert cmp_.json()["reference_kind"] == "manual annotation"


def test_upload_validation(client):
    r = client.post("/api/uploads?filename=scan.exe", content=b"MZ....")
    assert r.status_code == 400 and r.json()["error"]["code"] == "unsupported_extension"
    r = client.post("/api/uploads?filename=scan.nii.gz&modality=CT", content=b"plain text, not gzip")
    assert r.status_code == 400 and r.json()["error"]["code"] == "bad_magic"
    r = client.post("/api/uploads?filename=scan.nii.gz", content=nifti_bytes())
    assert r.status_code == 400 and r.json()["error"]["code"] == "modality_required"
    r = client.post("/api/uploads?filename=empty.zip", content=b"")
    assert r.status_code == 400


def test_nifti_upload_and_delete(client):
    r = client.post("/api/uploads?filename=Jane_Doe_scan.nii.gz&modality=MR", content=nifti_bytes())
    assert r.status_code == 200, r.text
    case = client.get(f"/api/cases/{r.json()['case_id']}").json()
    assert case["label"] == "UPLOADED RESEARCH DATA" and "Jane" not in str(case)  # filename never stored
    assert client.delete(f"/api/cases/{case['case_id']}").status_code == 200
    assert client.get(f"/api/cases/{case['case_id']}").status_code == 404
    assert client.delete("/api/cases/DEMO-001").status_code == 409


def test_dicom_upload_is_deidentified(client, tmp_path, settings):
    archive = zip_dir(write_ct_series(tmp_path / "ct"), tmp_path / "ct.zip")
    r = client.post("/api/uploads?filename=ct.zip", content=archive.read_bytes())
    assert r.status_code == 200, r.text
    deid = r.json()["deidentification"]
    assert {"PatientName", "PatientID"} <= set(deid["phi_keywords_present_in_source"])
    assert "PatientBirthDate" in deid["removed"] and deid["private_tags_removed"]
    case_id = r.json()["case_id"]
    meta = client.get(f"/api/cases/{case_id}/timepoints/T0/metadata").json()
    assert meta["modality"] == "CT" and meta["geometry"]["spacing_xyz_mm"] == [0.7, 0.7, 2.5]
    stored = "".join(p.read_text(errors="ignore") for p in settings.data_dir.rglob("*.json"))
    stored += settings.audit_log.read_text()
    assert PHI_NAME not in stored and PHI_MRN not in stored and "General Hospital" not in stored
    assert not list(settings.tmp_dir.iterdir())  # temporary files removed


def test_zip_path_traversal_rejected(client):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("../../evil.dcm", b"x" * 200)
    r = client.post("/api/uploads?filename=evil.zip", content=buf.getvalue())
    assert r.status_code == 400 and r.json()["error"]["code"] == "unsafe_path"


def test_upload_size_limit(tmp_path, small_demo):
    s = make_settings(tmp_path, max_upload_mb=1, generate_demo_cases=False)
    with TestClient(create_app(s)) as c:
        r = c.post("/api/uploads?filename=big.nii&modality=CT", content=b"\0" * (2 * 1024 * 1024))
        assert r.status_code == 413 and r.json()["error"]["code"] == "file_too_large"


@pytest.fixture
def keyed_client(tmp_path, small_demo):
    keys = tmp_path / "keys.yaml"
    keys.write_text(yaml.safe_dump({"keys": [
        {"sha256": hash_key("viewer-key"), "user": "vic", "role": "viewer"},
        {"sha256": hash_key("research-key"), "user": "rita", "role": "researcher"},
    ]}))
    s = make_settings(tmp_path, auth_mode="api_key", api_keys_file=keys)
    with TestClient(create_app(s)) as c:
        yield c, s


def test_access_control(keyed_client):
    c, s = keyed_client
    assert c.get("/api/cases").status_code == 401
    assert c.get("/api/cases", headers={"X-API-Key": "wrong"}).status_code == 401
    assert c.get("/api/cases", headers={"X-API-Key": "viewer-key"}).status_code == 200
    r = c.get("/api/cases/DEMO-001/timepoints/T0/metadata", headers={"X-API-Key": "viewer-key"})
    assert r.status_code == 403
    assert c.get("/api/cases/DEMO-001/timepoints/T0/metadata", headers={"X-API-Key": "research-key"}).status_code == 200
    assert c.get("/api/audit", headers={"X-API-Key": "research-key"}).status_code == 403
    log = s.audit_log.read_text()
    assert '"access_denied"' in log and '"view_metadata"' in log and "research-key" not in log
