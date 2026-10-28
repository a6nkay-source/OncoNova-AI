# Safety, intended use and limits

## Intended use

Research and education: building, testing and explaining imaging-AI pipelines for oncology with honest uncertainty, provenance and evaluation. Users are researchers and students working with synthetic, public or properly authorised de-identified data.

## Not intended for

Diagnosis, prognosis, staging, treatment selection, radiation prescription or planning, or any decision about a real person. The software is not a medical device and has no regulatory clearance.

## Hard boundaries in the code

| Boundary | How it is enforced |
|---|---|
| No control of, or communication with, a LINAC or other medical equipment | No device, fieldbus or DICOM-network library is a dependency. `safety.assert_no_device_interfaces()` refuses to start if one is imported. `tests/test_safety_privacy.py` statically checks every module's imports and that no route path suggests delivery, execution or plan export. |
| No prescriptions | The dose module works only in relative units (1.0 = median in an illustrative target). There is no input or output for a prescribed dose, fractionation or monitor units. Every dose result has `not_a_prescription: true` and the SIMULATED · NON-CLINICAL label. |
| No unsupported predictions | Missing required inputs return `{"status": "unavailable", …}` with the reason and what is missing. Tests check that no probability is returned in that case. |
| No silent model substitution | Model cards carry the checkpoint SHA-256. A mismatched checkpoint makes the model unavailable. |
| No invented performance | The Models & Data page and `docs/EVALUATION_REPORT.md` read stored evaluation artifacts only. With none stored, they say so. |
| No outbound transmission of data | The package imports no network clients (tested). The frontend bundles its fonts and calls only its own API. |
| No metastasis claims | Longitudinal analysis reports new segmented regions as *requires expert review*, with text stating that a new region is not by itself evidence of metastasis. The progression model has no spread-destination labels and says so. |

## Labels the user always sees

- **DEMO DATA**: every synthetic case, in the top bar, case records, binary responses (`X-Data-Label`) and exports.
- **DEMO MODEL**: every model trained or tuned on phantoms.
- **SIMULATED · NON-CLINICAL**: the LINAC Simulator banner, the "Simulation · dummy patient" label in the 3-D room, each dose result and the dose-map and per-beam replay headers. The simulator only replays a precomputed illustration in the browser; the gantry, beam and patient exist nowhere else.
- **Evidence badges**: measured, measured from AI mask, AI prediction, uncertain, expert review, simulated, unavailable.
- A permanent banner: research prototype, not for clinical use, no connection to medical equipment.

## Privacy and security controls

- Uploads: size limit enforced while streaming; extension allow-list; NIfTI/gzip/ZIP magic bytes; ZIP member-count, uncompressed-size and compression-ratio limits; path-traversal and symlink rejection; archive members renamed on extraction.
- DICOM is de-identified (configurable profile, `configs/deid_profile.yaml`) **before** it is parsed into a volume. Only the converted volume and technical metadata are stored. Identifying attributes are reported by name only. Uploaded filenames are never stored.
- Temporary files go to a private `0700` directory and are always removed.
- Access control: localhost-only dev mode, or hashed API keys with viewer/researcher/admin roles. Metadata views, uploads, analyses, deletions and denied requests are audit-logged with user, role and request ID, never with image content or identifiers.
- Errors return a stable code, a safe message and a request ID. Stack traces are logged server-side only.
- Deletion: uploaded cases and all derived results can be deleted from the UI or API.

## Known gaps

The de-identification profile is a practical subset of DICOM PS3.15. It does not inspect pixel data for burned-in text (it only warns when `BurnedInAnnotation=YES`), and it removes dates rather than shifting them. Treat it as a safeguard, not a certification. Dev mode trusts any local process. Use `api_key` mode behind TLS for anything shared.
