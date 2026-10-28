"""Filesystem case store.

Layout::

    data/cases/<case_id>/case.json
    data/cases/<case_id>/<tp>/image.nii.gz
    data/cases/<case_id>/<tp>/reference_lesions.nii.gz   (demo ground truth or imported reference)
    data/cases/<case_id>/<tp>/structures.nii.gz
    data/cases/<case_id>/<tp>/annotation.nii.gz          (manual annotation)
    data/cases/<case_id>/<tp>/pred_<model>.nii.gz, unc_<model>.nii.gz, sal_<model>.nii.gz
    data/cases/<case_id>/results/<name>.json

Only converted volumes and de-identified metadata are stored; uploaded DICOM files
are deleted after conversion.
"""

from __future__ import annotations

import json
import re
import shutil
import threading
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from onconova.errors import ConflictError, NotFoundError, OncoNovaError
from onconova.imaging.io import load_label_like, load_nifti, save_nifti
from onconova.imaging.volume import Volume

ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
LABEL_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")


def _check_id(value: str, what: str) -> str:
    if not ID_PATTERN.match(value):
        raise OncoNovaError("invalid_identifier", f"Invalid {what} identifier.", status_code=400)
    return value


class CaseStore:
    def __init__(self, root: Path, cache_size: int = 24) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._cache: OrderedDict[tuple[str, float], Volume] = OrderedDict()
        self._cache_size = cache_size
        self._lock = threading.Lock()

    # ----- cases -------------------------------------------------------
    def case_dir(self, case_id: str) -> Path:
        return self.root / _check_id(case_id, "case")

    def exists(self, case_id: str) -> bool:
        return (self.case_dir(case_id) / "case.json").exists()

    def list_cases(self) -> list[dict[str, Any]]:
        cases = []
        for path in sorted(self.root.glob("*/case.json")):
            cases.append(json.loads(path.read_text()))
        return cases

    def get_case(self, case_id: str) -> dict[str, Any]:
        path = self.case_dir(case_id) / "case.json"
        if not path.exists():
            raise NotFoundError("case_not_found", f"No case {case_id!r}.")
        return json.loads(path.read_text())

    def write_case(self, case: dict[str, Any]) -> None:
        directory = self.case_dir(case["case_id"])
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "case.json").write_text(json.dumps(case, indent=2))

    def delete_case(self, case_id: str) -> None:
        case = self.get_case(case_id)
        if case.get("demo"):
            raise ConflictError("demo_case", "Demo cases are regenerated automatically and cannot be deleted.")
        shutil.rmtree(self.case_dir(case_id))
        with self._lock:
            for key in [k for k in self._cache if k[0].startswith(str(self.case_dir(case_id)))]:
                del self._cache[key]

    def timepoint(self, case_id: str, tp_id: str) -> dict[str, Any]:
        case = self.get_case(case_id)
        for tp in case["timepoints"]:
            if tp["tp_id"] == tp_id:
                return tp
        raise NotFoundError("timepoint_not_found", f"Case {case_id!r} has no timepoint {tp_id!r}.")

    def tp_dir(self, case_id: str, tp_id: str) -> Path:
        self.timepoint(case_id, tp_id)
        return self.case_dir(case_id) / _check_id(tp_id, "timepoint")

    # ----- volumes -----------------------------------------------------
    def _cached(self, path: Path, loader) -> Volume:
        key = (str(path), path.stat().st_mtime)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        vol = loader()
        with self._lock:
            self._cache[key] = vol
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return vol

    def image(self, case_id: str, tp_id: str) -> Volume:
        tp = self.timepoint(case_id, tp_id)
        path = self.tp_dir(case_id, tp_id) / "image.nii.gz"
        if not path.exists():
            raise NotFoundError("image_not_found", "Image volume is missing for this timepoint.")
        vol = self._cached(path, lambda: load_nifti(path, modality=tp.get("modality", "UNKNOWN")))
        vol.metadata.update({"demo": self.get_case(case_id).get("demo", False)})
        return vol

    def label_path(self, case_id: str, tp_id: str, name: str) -> Path:
        if not LABEL_NAME_PATTERN.match(name):
            raise OncoNovaError("invalid_label_name", "Invalid label name.", status_code=400)
        return self.tp_dir(case_id, tp_id) / f"{name}.nii.gz"

    def label(self, case_id: str, tp_id: str, name: str) -> np.ndarray | None:
        path = self.label_path(case_id, tp_id, name)
        if not path.exists():
            return None
        ref = self.image(case_id, tp_id)
        return self._cached(path, lambda: ref.with_array(load_label_like(path, ref))).array

    def float_map(self, case_id: str, tp_id: str, name: str) -> np.ndarray | None:
        path = self.label_path(case_id, tp_id, name)
        if not path.exists():
            return None
        return self._cached(path, lambda: load_nifti(path)).array

    def save_map(self, case_id: str, tp_id: str, name: str, array: np.ndarray) -> Path:
        ref = self.image(case_id, tp_id)
        path = self.label_path(case_id, tp_id, name)
        save_nifti(ref.with_array(array), path)
        return path

    # ----- results -----------------------------------------------------
    def results_dir(self, case_id: str) -> Path:
        path = self.case_dir(case_id) / "results"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_result(self, case_id: str, name: str, data: dict[str, Any]) -> None:
        if not LABEL_NAME_PATTERN.match(name):
            raise OncoNovaError("invalid_result_name", "Invalid result name.", status_code=400)
        data = {**data, "saved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        (self.results_dir(case_id) / f"{name}.json").write_text(json.dumps(data, indent=2, default=_json_default))

    def result(self, case_id: str, name: str) -> dict[str, Any] | None:
        if not LABEL_NAME_PATTERN.match(name):
            return None
        path = self.results_dir(case_id) / f"{name}.json"
        return json.loads(path.read_text()) if path.exists() else None

    def save_array(self, case_id: str, name: str, array: np.ndarray) -> None:
        """Small derived arrays that are not on the image grid (e.g. per-beam dose slices)."""
        if not LABEL_NAME_PATTERN.match(name):
            raise OncoNovaError("invalid_result_name", "Invalid result name.", status_code=400)
        np.save(self.results_dir(case_id) / f"{name}.npy", array, allow_pickle=False)

    def array(self, case_id: str, name: str) -> np.ndarray | None:
        if not LABEL_NAME_PATTERN.match(name):
            return None
        path = self.results_dir(case_id) / f"{name}.npy"
        return np.load(path, allow_pickle=False) if path.exists() else None

    def results(self, case_id: str, prefix: str = "") -> dict[str, dict[str, Any]]:
        out = {}
        for path in sorted(self.results_dir(case_id).glob(f"{prefix}*.json")):
            out[path.stem] = json.loads(path.read_text())
        return out


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serialisable: {type(value)}")
