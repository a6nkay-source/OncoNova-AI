"""Safety guarantees: no device interfaces, no outbound network clients, de-identification."""

import ast
import sys
import types
from pathlib import Path

import pydicom
import pytest

import onconova
from onconova.imaging.deid import deidentify, pseudonym
from onconova.safety import FORBIDDEN_MODULES, DeviceInterfaceDetected, assert_no_device_interfaces

from conftest import PHI_MRN, PHI_NAME, write_ct_series

PACKAGE = Path(onconova.__file__).parent
NETWORK_CLIENTS = {"socket", "requests", "httpx", "urllib.request", "urllib3", "http.client", "ftplib", "smtplib",
                   "aiohttp", "websocket", "websockets", "paramiko"}


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


@pytest.mark.parametrize("path", sorted(PACKAGE.rglob("*.py")), ids=lambda p: str(p.relative_to(PACKAGE)))
def test_no_device_or_network_client_imports(path):
    for name in imported_modules(path):
        root = name.split(".")[0]
        assert root not in FORBIDDEN_MODULES, f"{path.name} imports device module {name}"
        assert name not in NETWORK_CLIENTS and root not in {"requests", "httpx", "aiohttp", "paramiko"}, (
            f"{path.name} imports network client {name}; medical data must not leave the server")
        if root in {"transformers", "huggingface_hub"}:  # can download from the hub unless forced offline
            src = path.read_text()
            assert 'os.environ["HF_HUB_OFFLINE"] = "1"' in src and "local_files_only=True" in src, (
                f"{path.name} uses {root} without forcing offline, local-only loading")


def test_runtime_guard_refuses_device_modules(monkeypatch):
    assert_no_device_interfaces()
    monkeypatch.setitem(sys.modules, "serial", types.ModuleType("serial"))
    with pytest.raises(DeviceInterfaceDetected):
        assert_no_device_interfaces()


def test_no_treatment_delivery_routes():
    from onconova.api.app import app

    paths = " ".join(getattr(r, "path", "") for r in app.routes).lower()
    for word in ("deliver", "treat", "beam-on", "beam_on", "execute", "export-plan", "send", "linac", "machine", "prescri"):
        assert word not in paths, f"route containing {word!r} found"


def test_deidentify_removes_and_pseudonymises(tmp_path):
    write_ct_series(tmp_path, n=1)
    ds = pydicom.dcmread(str(next(tmp_path.glob("*.dcm"))))
    original_uid = ds.SOPInstanceUID
    report = deidentify(ds, "salt")
    assert "PatientBirthDate" not in ds and "StudyDate" not in ds
    assert str(ds.PatientName) != PHI_NAME and str(ds.PatientID) == pseudonym(PHI_MRN, "salt")
    assert ds.InstitutionName == "" and ds.PatientIdentityRemoved == "YES"
    assert ds.SOPInstanceUID != original_uid and ds.SOPInstanceUID.startswith("2.25.")
    assert not any(elem.tag.is_private for elem in ds)
    assert report.private_tags_removed and PHI_MRN not in str(report.to_dict())


def test_pseudonyms_are_stable_and_salted():
    assert pseudonym("A", "s1") == pseudonym("A", "s1") != pseudonym("A", "s2")


def test_gitignore_excludes_medical_data_and_secrets():
    ignore = (PACKAGE.parents[1] / ".gitignore").read_text()
    for pattern in ("data/", "*.nii", "*.nii.gz", "*.dcm", ".env", "*.pt", "artifacts/models/"):
        assert pattern in ignore
