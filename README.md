# OncoNova AI

**AI-assisted cancer imaging research workstation: tumour segmentation, longitudinal change, progression-risk modelling and research-only radiation-dose visualisation.**

> **Research and educational prototype. Not a medical device.** OncoNova AI does not diagnose, does not prescribe or recommend radiation doses, has no interface to any linear accelerator or other medical equipment, and produces nothing clinically actionable. Every model here was trained or tuned on **synthetic data**. Clinical use would need clinical validation, regulatory authorisation and qualified medical supervision. See [docs/SAFETY.md](docs/SAFETY.md).

---

## The three pages

| Page | What you see |
|---|---|
| **CT Scan Analysis** | Pick a case and the pipeline runs by itself: the AI draws lesions on every scan (with a "where the AI is unsure" view and the true outline for synthetic cases), the change between scans is measured, and a separate model estimates the chance of progression over 180 days with a 95% interval, limitations and an explicit unavailable state when inputs are missing. Includes a 3-D view, CT import and the A-G research report. |
| **LINAC Simulator** | A 3-D treatment room with a breathing dummy patient built around the case's anatomy. A virtual gantry replays a precomputed geometric dose illustration beam by beam, with optional breath-hold gating, a simulated patient monitor and a dose slice that builds up as each beam is replayed. DVH and summary numbers appear at the end. Everything is labelled SIMULATED · NON-CLINICAL and there is no device interface. |
| **Models & Data** | Measured evaluation results (from stored reports only), the verified public-dataset catalogue and local cases, and the safety, de-identification, access and audit settings. |

## What it does

| Module | What is implemented | Evidence type shown in the UI |
|---|---|---|
| **A. Medical imaging** | NIfTI and DICOM-series ingestion (CT, MR, PET) with validation, reorientation to LPS without resampling, modality-aware normalisation, rigid registration, de-identified metadata inspection, three-plane viewer with window presets and custom W/L, zoom and pan, keyboard and slider slice navigation, mask outlines traced from the raw mask, overlays drawn only when their frame and geometry match the scan, and crosshair positions in patient millimetres (manual annotation is available through the API) | Measured |
| **B. Tumour segmentation** | Classical lung-lesion baseline and a MONAI 3-D U-Net (trained on phantoms), per-lesion volumes in cc, Monte Carlo dropout uncertainty, unreliable-region detection, gradient saliency, out-of-distribution warnings, Dice / IoU / Hausdorff / HD95 / sensitivity / specificity against a reference, 3-D surfaces (saliency is computed by the API, not shown in the UI) | AI prediction · Measured from AI mask |
| **C. Progression research** | Three separate tasks: observed lesions (B), **observed change** between scans (registration, lesion tracking, volume change, new regions flagged for review) and **future risk** (logistic model + bootstrap ensemble, fixed 180-day horizon, explicit unavailable state) | Measured · AI prediction · Expert review |
| **D. LINAC research simulator** | Geometric beam illustration in relative units (labelled SIMULATED · NON-CLINICAL), four hypothetical beam/margin scenarios, per-beam shares for a live replay in a 3-D room with a breathing dummy patient and gating, DVHs and plan-quality indices. Read-only RTDOSE/NIfTI dose import is in the API. No device interface. | Simulated · Imported |
| **E. Integrated dashboard** | Automatic pipeline on the CT page, image + 3-D view, cards for each module, research summary sections A-G with every line tagged by evidence type, JSON/Markdown export, provenance and model versions | All of the above |

All synthetic cases are marked **DEMO DATA** in the API, the UI and every export.

## Measured results (synthetic data only)

From `docs/EVALUATION_REPORT.md`, generated from stored evaluation artifacts (`make report`):

| Model | Test set | Headline (95% CI) | Known failure |
|---|---|---|---|
| Classical heuristic | 9 held-out synthetic patients / 18 scans | Primary-lesion Dice 0.910 (0.894-0.925) | Node-region lesions never detected (by design) |
| 3-D U-Net (MONAI) | same | Primary-lesion Dice 0.940 (0.930-0.949); 0.11 false-positive components/scan | Node-region lesions missed; "other" lesions labelled primary |
| Progression (logistic) | simulated cohort, external site | AUROC 0.769 (0.698-0.833), Brier 0.197 | Predicts a *simulated* outcome; shows only that the code recovers planted signal |

These numbers describe phantoms and a simulated cohort. **They are not evidence of performance on patients.**

## Quick start

Requirements: Python 3.11+ (3.13 tested), Node 20+ (26 tested). macOS, Linux or WSL.

```bash
make setup            # venv + backend + frontend dependencies
make setup-dl         # optional: PyTorch + MONAI for the 3-D U-Net
make train-progression
make train-seg        # ~20 min on an Apple-silicon GPU; generates the synthetic dataset first
make evaluate && make report
```

Run the two servers in separate terminals:

```bash
make api              # http://127.0.0.1:8010  (OpenAPI docs at /docs)
```

```bash
make web              # http://127.0.0.1:5174  (proxies /api to :8010)
```

Demo cases are generated automatically when the API first starts. Open **CT Scan Analysis** and the pipeline runs on the selected case; then try **LINAC Simulator** and press **Start simulation**.

Without `make train-seg`, the U-Net shows as *unavailable*, and without `make train-progression` the risk panel shows an explicit unavailable state. Nothing pretends to run.

## Tests

```bash
make test             # backend pytest + frontend vitest + TypeScript type-check
```

Backend tests cover ingestion and orientation (markers keep their patient position through all 48 axis orders and flips of a NIfTI, oblique NIfTI and DICOM, and DICOM series that are shuffled, reversed, feet-first or coronal/sagittal), overlays refused unless they share the image's grid, physical volumes and diameters (diameter bias on digitised ellipses bounded per spacing), metrics, segmentation output and uncertainty, registration and longitudinal change, patient-level splitting and leakage, progression metrics and missing data, DVH maths and RTDOSE import, API validation and error envelopes, uploads (size limits, magic bytes, zip-slip), de-identification, access control, audit logging, and static checks that the package imports no device or network-client libraries. Frontend tests cover error, unavailable and evidence states, the API client, viewer geometry, mask outlines (area within half a voxel of the voxel count, every vertex on a mask voxel edge), the screen transform (round trips, zoom about the cursor, pan), overlay alignment through the real renderer (scan pixel, mask outline, crosshair and click land on the same screen pixel in every plane after resizing, zooming and panning), overlay geometry and frame checks, never drawing on the previous scan while the next loads, keyboard, slider and drag input, and the LINAC replay clock (every beam replayed once, beam off while the gantry moves, gating holds the beam outside the exhale window). Backend dose tests check that per-beam shares sum to the illustration and that gantry angles follow the IEC convention.

## Repository layout

```
backend/onconova/
  imaging/        volume container, NIfTI/DICOM I/O + validation, normalisation, registration, de-identification
  segmentation/   classical baseline, MONAI U-Net, metrics, measurements, uncertainty, model registry
  longitudinal/   registration-based lesion tracking and observed change
  progression/    features, simulated cohort, logistic + bootstrap model, metrics with CIs
  dose/           geometric illustration, DVH maths, RTDOSE/NIfTI import (read-only)
  data/           patient-level splits, synthetic and MSD manifests
  synthetic/      thoracic CT phantom generator and the DEMO DATA cases
  api/            FastAPI app, routes, binary volume responses, auth dependencies
  services.py     orchestration between store, models and API
  safety.py       labels, evidence kinds, device-interface guard
backend/scripts/  train / evaluate / report / import scripts
backend/tests/    pytest suite
frontend/src/     React + TypeScript + Tailwind UI (viewer, 3-D room, replay clock, charts, three pages)
configs/          training config, de-identification profile, public dataset catalogue
docs/             architecture, API, datasets, deployment, safety, evaluation report, roadmap
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md): modules, data flow, design decisions
- [API reference](docs/API.md): every endpoint, generated from the OpenAPI schema
- [Dataset guide](docs/DATASETS.md): verified public datasets, ground-truth caveats, recommended starter and download steps
- [Model evaluation report](docs/EVALUATION_REPORT.md): intended use, data, protocol, measured results, limitations
- [Deployment guide](docs/DEPLOYMENT.md): local, Docker, access control, hardening
- [Safety and intended use](docs/SAFETY.md)
- [Roadmap](docs/ROADMAP.md)

## Status by stage

| Stage | Status | Not done yet |
|---|---|---|
| 1. Architecture and UI | Done | - |
| 2. Ingestion and visualisation | Done | DICOM SEG/RTSTRUCT import, series picker UI, PET SUV conversion |
| 3. Segmentation | Done on synthetic data | Training on real data (MSD importer is ready), nnU-Net baseline |
| 4. Longitudinal comparison | Done (rigid) | Deformable registration, image-based registration QC |
| 5. Progression pipeline | Framework done on a simulated cohort | Real longitudinal labels (e.g. LUMIERE), survival models |
| 6. Dose visualisation | Done (illustration + import) | RTSTRUCT contours for imported plans, a validated external research dose engine |
| 7. Evaluation, uncertainty, provenance | Done | Calibration of segmentation uncertainty against real error |
| 8. Tests, docs, security, deployment | Tests and docs done | Docker images written but **not built or tested here** (Docker was not available) |

## Licence and data

Code: choose a licence before publishing. No medical images, credentials or model weights are committed (see `.gitignore`). Public datasets keep their own licences (several are non-commercial); see [docs/DATASETS.md](docs/DATASETS.md).
