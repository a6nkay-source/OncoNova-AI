# Changelog

Each entry says what changed and how it was checked. "Measured" means a test or a recorded run shows it; anything else is labelled as unverified.

## Unreleased

### Stage 3: segmentation on real CT (in progress)

Not finished: the candidate models are still training, so no candidate result exists yet and none is claimed here.

- **Real data.** MSD Task06 Lung was downloaded from the MONAI mirror and checked: 9,163,696,640 bytes, MD5 equal to MONAI's published value. It was extracted with a path-checking filter. The release has 63 labelled and 32 unlabelled CTs (the dataset catalogue said 64 labelled; corrected).
  - The patient-level manifest has 44 train, 9 val and 10 test patients, with none excluded.
  - Candidates are compared on val only. The test split is locked: the evaluation script refuses it without `--unlock-test`.
- **The U-Net now runs where it was trained.** Scans are resampled to the training spacing and segmented in overlapping tiles of the training patch, with Gaussian blending. Class probabilities are resampled back to the scan grid before the argmax.
  - On the 18 synthetic test scans: any-lesion Dice 0.920 → 0.933, primary-lesion Dice 0.940 → 0.954, 1.2 → 1.7 s per scan.
  - On the same scans resampled to 0.7 × 0.7 × 1.25 mm, as real CT is: 0.249 without resampling, 0.911 with it.
  - *Tests:* a lesion keeps its centre to within a quarter voxel through the model grid and back, including for a rotated scan; class probabilities still sum to 1; any scan spacing gives output on the scan's own grid.
- **Baselines on real CT (9 val patients).** These are the two models OncoNova had before this stage. Both are near zero, as expected for models built on synthetic phantoms.
  - Classical heuristic: Dice 0.024 (95% CI 0.010–0.042), 40 s per scan.
  - Synthetic-trained U-Net: Dice 0.029 (0.000–0.078), 23 s per scan.
- **Candidates set up.** None evaluated yet.
  - The same U-Net architecture trained on 37 MSD train patients, 3 seeds plus their ensemble. Checkpoints are chosen on 7 other train patients, so the val patients stay unseen.
  - MedSAM refining the ensemble's detections. Its weights were checked against the published SHA-256. It is prompted only by the detector, never by reference boxes, and loads offline from local files.
  - nnU-Net v2 via a Colab notebook on exactly the same patients.
  - A calibration check that decides whether mirror test-time augmentation or MC dropout may drive a confidence overlay.
- **Labels.** Segmentation results now carry the model's own label, so a real-CT model is not described as "trained on synthetic phantoms".
- **Safety.** The static check now also requires any module using `transformers` or `huggingface_hub` to force offline, local-only loading.

### Stage 2: coordinate correctness

The loaders were already correct. What was missing was proof, plus three ways an overlay could reach the screen in the wrong place.

- **DICOM and NIfTI orientation, spacing and slice order are now tested end to end** (`backend/tests/test_geometry.py`, 66 tests). Each test writes markers with distinct values at known patient positions, loads the file normally, finds each marker and checks its LPS position to 0.001 mm. Non-square pixels and matrices catch swapped axes, and a different marker position on every slice catches flips and slice-order mistakes. Cases covered:
  - all 48 axis orders and flips of a NIfTI;
  - an oblique NIfTI (position kept, oblique warning raised);
  - DICOM series that are axial, axial with either in-plane axis reversed, coronal, sagittal and oblique, each stacked both ways along the slice normal;
  - series whose file names and InstanceNumbers are shuffled, reversed or misleading.

  *All passed on the existing loader.* To show the tests can fail, I broke the loader three ways for one run each:
  - stacking files in name order: 8 failures;
  - skipping the LPS reorientation: 55 failures;
  - mirroring the voxels without changing the header: all 64 failures.
- **Overlays are checked against their image on both ends.**
  - Server: uncertainty, saliency and dose maps were loaded without comparing their grid to the image, and were served with the image's geometry headers. Every stored map is now loaded against its image and refused with `geometry_mismatch` if shape, spacing, origin or direction differ. Overlays are served with their own geometry and a new `X-Frame` header naming the scan they belong to. *Tested:* a map shifted by 5 mm returns 422 `geometry_mismatch`, and overlays carry the same frame and geometry as their image. A follow-up registered into the baseline carries the baseline's frame.
  - Client: the viewer accepted any overlay with the same voxel count, so a mask from another scan of the same size would have been drawn without warning. It now requires a matching frame and geometry, using the server's tolerances. Otherwise the overlay's row says "Not drawn" and why. The LINAC dose replay checks its target mask and per-beam slices the same way.
- **Switching scans can no longer pair one scan's overlays with another's image.** While a new scan loaded, the image hook kept returning the previous scan. Now it never returns a volume for a URL other than the one requested. The viewer keeps the last scan on screen under a "Loading scan" veil with no overlays, so zoom and pan survive the switch. *Regression test added; it fails on the old hook.*
- **One path from voxel to screen.** Crosshair, lesion focus and clicks now go through `voxelToScreen` and `screenToVoxel` in `viewer/transform.ts`.
  - *Unit test:* every voxel of an anisotropic, non-cubic volume, in all three planes, at three viewport sizes, zooms and pans, is drawn at the pixel a click there selects.
  - *Render-path test* (`frontend/src/test/alignment.test.tsx`): a scan with one bright voxel and a mask on it is drawn through the real `SliceView` into a recording canvas. In each plane, after two resizes, a zoom and a pan, the bright pixel, the mask outline's bounds, the crosshair and a click all land on the same screen pixel. I broke this path five ways to check the tests catch it, and each was caught:
    - markers half a voxel off: 4 failures;
    - outline half a voxel off: 3 failures;
    - coronal rows not flipped: 2 failures;
    - overlays accepted by voxel count: 1 failure;
    - the old loading hook: 1 failure.
  - *Bug it found:* the crosshair was snapped with `round(x) + 0.5`, putting it up to 1 screen pixel off the voxel centre, which is a third of a voxel at the fitted zoom. It now snaps to the pixel containing the centre, at most 0.5 px off.
- **Positions in patient millimetres.** Each view's slice readout and the crosshair footer show LPS millimetres from the image header, as anatomical directions ("R 75.9 · A 1.1 · S 13.8 mm"). The client conversion uses the direction matrix and matches the server's `index_to_physical`. Oblique scans show "oblique" instead of a slice position, plus a note that the orientation letters are approximate.
- **Diameter bias reduced, not removed.** The maximum axial diameter used the longest centre-to-centre distance plus one x-spacing, whatever the direction. On simulated, digitised ellipses 10–40 mm across, that overestimated round lesions by up to 0.9 voxel.
  - I compared four estimators (centres alone, plus one voxel, plus half the voxel's projected width, plus all of it). Adding half the voxel's width along the measured direction, in mm, was best: mean |bias| 0.23 voxel against 0.39, worst case 0.73 against 0.91.
  - *Test on a different random seed:* the mean bias stays under 0.8 voxel for every spacing, size and shape. By spacing it is −0.17 to +0.35 mm at 0.7 mm, −1.5 to +1.0 mm at 2.2 mm, and −0.5 to +0.7 mm at 1.6 × 0.8 mm pixels.
  - A binary mask cannot resolve a diameter more finely than about a voxel, and the remaining error depends on shape.
  - On the stored DEMO-001 masks, the new primary-lesion diameters are 0.7–0.9 mm smaller than the old ones. Results stored before this change keep the old value until the case is re-run.
- **Measurements use physical spacing.** A test shows that diameter and volume scale with the per-axis spacing in mm, not voxel counts, and that stretching rows and columns gives different results.

Test status at the end of the stage: 195 backend tests pass (127 before: 66 geometry, 2 measurement), and 40 frontend tests pass (32 before). The TypeScript check is clean and the production build succeeds. In the browser, on DEMO-001:
- the footer position matches the header arithmetic;
- switching scans raised no errors or "Not drawn" notices;
- at 381% zoom in Voxels mode, both outlines sit on the voxel grid over the lesion;
- the LINAC replay still draws its dose wash and target outline.

Not covered, carried forward:
- The app has no cropping or resampling of displayed volumes apart from registration. Registration's label resampling was already tested (`test_longitudinal.py`).
- Phone touch pinch is still not implemented.
- The "Loading scan" veil appears only when a scan isn't cached yet. In the browser the scans were already cached, so I could only check it in the unit test.

### Stage 1: viewer graphics and resolution (display only)

No data, model or measurement code changed in this stage.

- **Mask outlines follow the raw mask.** Before, masks were mixed into the same small bitmap as the CT and then enlarged with smoothing, so edges blurred and colours bled into each other. Now outlines are traced from the stored mask (`viewer/contours.ts`) and drawn as vector paths at screen resolution, as fill and line, line only, or fill only. The mask is never altered.
  *Verified by unit tests:* every outline vertex lies on an edge between an inside and an outside voxel; a simple region's outline encloses its voxel count minus exactly 0.5 voxel; a ring encloses its exact count; labels and diagonal-only neighbours stay separate.
- **Zoom and pan.**
  - Zoom: ⌘/Ctrl + wheel, trackpad pinch, `+`/`-` keys or per-view buttons. It is anchored at the cursor and limited to 50–1600%.
  - Pan: Shift-, right- or middle-drag.
  - Fit: double-click, `0`, or "Fit views" for all three planes.
  - Clicking a lesion chip centres a zoomed-in view on it.

  All drawing and hit-testing go through one transform (`viewer/transform.ts`).
  *Verified by unit tests* (round trips at any zoom and pan, the point under the cursor stays put, pan can't lose the image) and in the browser (DEMO-001: the chip centred the zoomed axial view on voxel x29 y63, and pinch zoom moved towards the cursor).
- **Slice navigation.**
  - Each view has a slider with the slice position in mm.
  - Keys: arrows, Page Up/Down (10% of the stack), Home and End.
  - The wheel moves one slice per mouse notch; trackpads accumulate 50 px per slice, so slices aren't skipped. Wheel events over the viewer no longer scroll the page.

  *Verified by unit tests* (keyboard and slider moves stay in range) and in the browser (one notch moved 1 slice; 100 px of small trackpad deltas moved 2).
- **Window/level.** Lung, mediastinal and bone presets plus Auto, as before, with editable W and L fields and Alt-drag. Windowing only changes the display.
  *Verified in the browser:* an Alt-drag from Lung gave W 1300 / L −540 and showed "custom".
- **Resolution shown, interpolation switchable.**
  - The viewer shows the matrix, voxel size and slice spacing.
  - Each view shows its zoom and the screen pixels per voxel.
  - "Smooth" interpolates the CT for display; "Voxels" shows each stored voxel as a square.
  - No enhancement or super-resolution is applied.
- **Overlay controls.** Each visible overlay has a colour per class, a style and an opacity. These replace the fixed legend.
- **Fixes found while testing.**
  - A pan drag applied only its first pointer move, because the delta was read after the drag origin had moved on. *Regression test added; it fails on the old code.*
  - Clicking a lesion chip a second time did nothing, because the same array object was passed again.
  - A pointer event without coordinates could report a NaN voxel position.
- **Performance.** Moving the crosshair or zooming no longer recomputes the slice bitmap; only changing the slice, window or overlay data does.

Test status at the end of the stage: 32 frontend tests pass (19 before), the TypeScript check is clean, and the production build succeeds. The backend is unchanged (127 tests).

Known limits, carried into Stage 2:
- The viewer still accepts an overlay whose voxel count matches the image, without comparing shape, spacing or origin.
- Touch pinch on phones isn't implemented.
- The demo scans are only 128×128×72 at 2.2 mm, so they look soft at any zoom.
