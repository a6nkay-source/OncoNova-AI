# Changelog

Each entry says what changed and how it was checked. "Measured" means a test or a recorded run shows it; anything else is labelled as unverified.

## Unreleased

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
