# Architecture

```
 Browser (React + TS, Vite)                 FastAPI (Python)                         Storage (local disk)
 ─────────────────────────                 ────────────────                         ────────────────────
 Pages ──► api/client.ts ──HTTP /api──►  routes_*  ──► services.py ──► models       data/cases/<id>/…  (NIfTI + JSON)
 Viewer ◄── binary volumes (int16/uint8)  deps.py (auth, audit)          │          data/audit/audit.jsonl
 Three.js ◄── surface meshes (JSON)        binary.py                     ▼          artifacts/models/…  (weights + cards)
                                            app.py (errors, CORS,    imaging/ segmentation/ longitudinal/ artifacts/evaluations/*.json
                                            logging, safety guard)   progression/ dose/ data/ synthetic/
```

## Principles

1. **The UI never touches model internals.** Pages call typed client functions; routes call `services.py`; services call models and the store. Models can be swapped without UI changes.
2. **Geometry travels with voxels.** `imaging.volume.Volume` holds the array (z, y, x) plus spacing, origin and direction in ITK/LPS conventions. Loaders reorient to LPS with `DICOMOrient` (a permutation and flip, no resampling), so physical coordinates are preserved exactly; DICOM slices are stacked by position along the slice normal, never by file name or InstanceNumber. Every stored label or continuous map is loaded against its image and refused (`geometry_mismatch`) unless shape, spacing, origin and direction agree. Binary responses carry that geometry plus `X-Frame` (`case/timepoint`, the grid the voxels live on; registered follow-ups carry the baseline's frame), and the viewer draws an overlay only when frame and geometry match its image. Volumes are computed as voxel count × voxel volume; distances and diameters use the per-axis spacing in mm, never screen pixels.
3. **Evidence is typed.** Every statement returned to the UI is a `finding(kind, text)`, where `kind` is one of `measured`, `measured_from_model_output`, `model_output`, `uncertain`, `requires_expert_review`, `simulated` or `unavailable`. The UI renders each kind with its own badge.
4. **Missing inputs give an explicit unavailable state** (`safety.unavailable`), never an extrapolated answer.
5. **Results carry provenance.** Each stored result records software version, data label, generator or upload source, model id/version and checkpoint or artifact SHA-256. Checkpoints whose hash does not match their model card are refused.
6. **Nothing leaves the server.** The package imports no network clients (enforced by a test). There is no device interface (runtime guard plus static test).

## Modules

| Package | Responsibility |
|---|---|
| `imaging.io` | NIfTI and DICOM-series loading, slice-spacing and orientation checks, technical metadata (PHI listed by keyword only), validation with stable error codes |
| `imaging.normalize` | CT HU windows; MR foreground z-score; PET percentile scaling (no SUV) |
| `imaging.registration` | Rigid Euler-3-D registration (Mattes MI, 3-level pyramid) and resampling (nearest for labels) |
| `imaging.deid` | Profile-driven DICOM de-identification (remove, blank, HMAC pseudonyms, UID remap, private tags) |
| `segmentation.classical` | Lung-envelope + density + shape heuristic; heuristic entropy uncertainty inside the envelope |
| `segmentation.unet` | MONAI 3-D U-Net (single checkpoint or an ensemble listed in its card), resampled to its training spacing and run in overlapping tiles of its training patch (Gaussian blending), probabilities resampled back to the scan grid before the argmax; optional 8-mirror TTA; MC-dropout entropy, input-gradient saliency, small-component post-processing, checkpoint hash check |
| `segmentation.grid` | The model working grid: same patient region as the scan at the model's spacing; resampling there and back (class probabilities channel by channel) |
| `segmentation.medsam` | MedSAM (ViT-B) refining another model's detections slice by slice; boxes come only from that model, never from reference masks; loads local weights only, with the Hugging Face hub forced offline |
| `segmentation.metrics` / `measure` | Dice, IoU, Hausdorff, HD95, MSD, sensitivity, specificity (with ROI); cc volumes, max axial diameter, sphericity, marching-cubes meshes |
| `longitudinal.compare` | Registers follow-ups to baseline, tracks lesions across time, reports persisting / new / not-segmented regions, change and centroid shift |
| `progression` | Feature extraction from longitudinal output, simulated cohort, logistic + bootstrap ensemble, AUROC / AUPRC / Brier / C-index / calibration with bootstrap CIs |
| `dose` | Geometric illustration (relative units), DVH maths and indices, RTDOSE/NIfTI import resampled to the image grid |
| `data` | Patient-level and temporal splits, leakage checks, synthetic and MSD manifests (`scripts/import_msd.py` validates every scan and mask) |
| `store` | Filesystem case store with ID validation (no path traversal) and an LRU volume cache |
| `uploads` | Streaming size limit, extension allow-list, magic bytes, ZIP member/size/ratio limits, zip-slip and symlink rejection, de-identification before parsing, private temp dirs |
| `security` / `audit` | Hashed API keys with roles (viewer < researcher < admin), localhost-only dev mode, JSONL audit log |

## Real-CT evaluation (Stage 3)

- `scripts/evaluate_segmentation.py --dataset msd-task06-lung --split val` evaluates on any manifest, one model per process, recording per-scan inference time and the process's peak memory. An external dataset's `test` split is refused without `--unlock-test`, so models are chosen on `val`.
- `scripts/train_segmentation_msd.py` trains the same U-Net architecture on the MSD training patients. Scans are resampled to its grid and cached as memory-mapped arrays. Checkpoints are chosen on an inner split of the training patients, so the `val` patients stay unseen by both training and selection. It trains one model per seed and writes an ensemble card.
- `scripts/calibrate_segmentation.py` measures whether TTA or MC dropout improves voxel calibration (ECE and Brier, near the tumour and over the body) before any confidence overlay is shown.
- `scripts/build_segmentation_comparison.py` writes `docs/SEGMENTATION_COMPARISON.md` from the stored reports only, with paired per-patient differences against the best baseline.
- `notebooks/nnunet_msd_task06_colab.ipynb` trains nnU-Net v2 on exactly the same fit and inner-validation patients on a Colab GPU. The `val` and `test` patients are never copied into its dataset.
- New models appear in the app only when their card records `"adopted": true`, a decision taken from the comparison.

## Data flow for one case

1. **Ingest:** upload → validate → (DICOM) de-identify → convert → `image.nii.gz` + `case.json`.
2. **Segment:** `POST …/segmentation` → model → `pred_<model>.nii.gz`, `unc_<model>.nii.gz`, optional saliency → summary JSON with lesions, uncertainty, OOD warnings and findings.
3. **Observed change:** `POST /longitudinal` → registration per follow-up → tracks → findings (new regions go to expert review).
4. **Risk:** `POST /progression` → features from the longitudinal result → model, or unavailable.
5. **Simulation:** `POST /dose` → illustrative target (contour + margin) → relative dose → DVHs.
6. **Summary:** `GET /summary` assembles sections A-G from whatever has actually been computed.

## Frontend

Three pages: `pages/ScanAnalysis.tsx` (runs segmentation → longitudinal → progression automatically for the selected case and model, reusing stored results), `pages/LinacSimulator.tsx` and `pages/ModelsData.tsx` (evaluation, datasets and safety sections under `pages/models/`).

- `viewer/SliceView.tsx` draws one plane in layers. The windowed CT slice and any continuous maps (uncertainty, dose) are cached as native-resolution bitmaps, one canvas pixel per voxel, and scaled to the physical aspect ratio; the CT may be interpolated for display ("Smooth") or shown as stored ("Voxels"). Label masks are never resampled into the image: `viewer/contours.ts` traces their outlines (marching squares on the raw mask) and they are drawn as vector paths at screen resolution, fill and/or line, so the drawn boundary stays on the mask's own voxel edges at any zoom. Display is radiological. Zoom is anchored at the cursor (⌘/Ctrl + wheel, pinch, keys or buttons); pan with Shift-, right- or middle-drag; Alt-drag sets window/level. Slices change with the wheel (one per mouse notch, trackpads accumulate), the arrow, Page and Home/End keys, or the slider, which also shows the slice position in mm. A brush paints annotations on the axial plane.
- `viewer/transform.ts` is the single mapping between voxels, plane pixels and the screen (fit, zoom about a point, pan, `voxelToScreen` / `screenToVoxel`). Image, outlines, crosshair, markers, clicks and brush all use it. `test/alignment.test.tsx` renders a scan with one bright voxel and a mask on it through the real `SliceView` into a recording canvas and checks that pixel, outline, crosshair and click agree in every plane after resizing, zooming and panning.
- `viewer/geometry.ts` holds the client's patient geometry: voxel centre to LPS mm (with the direction matrix, so oblique scans are exact), anatomical labels for coordinates (R/L, A/P, I/S), and `geometryMismatch`, which says why an overlay must not be drawn on an image.
- `viewer/CaseViewer.tsx` loads the image and overlays (shared request cache; a volume is never handed back for a URL other than the one asked for, so while the next scan loads the last one stays on screen under a veil with no overlays), refuses overlays whose frame or geometry differ and says why in the overlay's row, shows the crosshair in patient mm, flags oblique scans, lays out the planes, and holds window presets plus custom W/L, the interpolation switch, a matrix and voxel-size readout, and per-overlay colour, style and opacity controls that double as the legend. These display choices are kept in `localStorage` as a convenience; the page works without it.
- `components/Volume3D.tsx` builds three.js meshes from server-side marching cubes, in LPS, viewed from anterior.
- `components/charts.tsx` provides dependency-free SVG line, interval, forest, strip and diverging-bar charts.
- `sim/treatment.ts` is the LINAC replay clock: a pure, unit-tested state machine that moves a virtual gantry through the illustration's beam angles (IEC 61217, head-first supine), breathes the dummy patient (cos⁴ waveform, 8 mm superior-inferior target motion) and holds the beam outside an exhale gating window. It only weights the per-beam slices the API already computed (`GET /api/cases/{id}/dose/beams`); it models no physics and has no output other than the screen.
- `components/TreatmentRoom.tsx` draws the room in the case's LPS frame: the anatomy meshes from the API inside a stylised dummy, a couch, room lasers and a C-arm gantry rotating about the isocentre (target centre). `components/SimPanels.tsx` holds the breathing monitor and the dose slice that builds up beam by beam.

## Why these choices

- **Filesystem store, not a database:** single-user research prototype, easy to inspect and delete. The store interface is small enough to swap later.
- **Server-side meshing, client-side slicing:** slicing a 128×128×72 volume in the browser is instant and makes overlay opacity and window changes free. Marching cubes stays in Python (scikit-image).
- **Classical baseline next to the U-Net:** a transparent reference shows what learning adds, and fails in known, documented ways.
- **Simulated progression cohort:** no public dataset combines these imaging features with a progression label, so a planted-signal cohort is the honest way to exercise splits, calibration and CIs end to end.
