# Segmentation: baseline vs candidates on real CT

Generated 2026-10-29 06:23 UTC by `backend/scripts/build_segmentation_comparison.py` from the stored evaluation reports. Do not edit by hand.

OncoNova AI is an educational research prototype, not a clinically validated diagnostic system. These are research measurements on a public dataset, not evidence of clinical performance.

**Data:** `msd-task06-lung`, **val** split: 9 patients, one CT each. Lung (CC-BY-SA 4.0). The split is by patient: {'val': 9, 'train': 44, 'test': 10}. Candidates were trained on (part of) the train patients and chose their checkpoints on an inner split of them, so these validation patients were unseen by training and selection. The test patients stay locked for the final evaluation.

**Reference:** the released single-rater tumour masks (not pathology-confirmed). **Protocol:** Patient-separated val split from the manifest. Per-scan metrics against the released manual masks, averaged per patient; mean, median and 2000-sample patient bootstrap 95% CI across patients. Specificity is computed inside the body mask (largest region above -500 HU, holes filled). Any-lesion only: the masks have one tumour label. One deterministic pass (no MC dropout).

With 9 patients the intervals are wide; a difference whose interval spans 0 is not shown to be real.

## Results

| Model | Dice mean (95% CI) | Dice median | IoU mean | HD95 mm median | Tumours found | False regions / scan | s / scan | Peak RAM MB |
|---|---|---|---|---|---|---|---|---|
| `demo-classical` (baseline) | 0.024 (0.010-0.042) | 0.017 | 0.013 | 176.1 | 8/28 | 55.4 | 40.0 | 4645 |
| `demo-unet` (baseline) | 0.029 (0.000-0.078) | 0.000 | 0.016 | 199.8 | 2/28 | 13.6 | 23.3 | 7672 |
| `nnunet-msd` | not evaluated | | | | | | | |

Dice and IoU: overlap with the reference (1 is perfect). HD95: 95th-percentile surface distance (lower is better; undefined when a model finds nothing). Tumours found: reference tumour regions that the prediction overlaps at all. False regions: predicted regions touching no reference tumour. Timing is inference only, on this Mac (mps); peak RAM is the whole evaluation process.

## Paired difference in Dice against the best baseline (`demo-unet`)

| Model | Mean difference (95% CI) | Patients improved |
|---|---|---|
| `demo-classical` | -0.005 (-0.042 to +0.018) | 7/9 |

## Per patient (Dice)

| Patient | `demo-classical` | `demo-unet` |
|---|---|---|
| lung_001 | 0.017 | 0.000 |
| lung_022 | 0.052 | 0.043 |
| lung_031 | 0.028 | 0.000 |
| lung_036 | 0.001 | 0.000 |
| lung_053 | 0.032 | 0.000 |
| lung_061 | 0.001 | 0.000 |
| lung_066 | 0.000 | 0.000 |
| lung_070 | 0.011 | 0.000 |
| lung_073 | 0.078 | 0.220 |

## Calibration of the foreground probability

Not run yet (`scripts/calibrate_segmentation.py`).

## Models

- `demo-classical`: DEMO MODEL - trained or tuned on synthetic phantoms only; not validated on patients. Checkpoint `n/a`, report `segmentation-demo-classical-msd-task06-lung-val-20261029T061405Z`.
- `demo-unet`: DEMO MODEL - trained or tuned on synthetic phantoms only; not validated on patients. Checkpoint `b964d38dd42c8b7f`, report `segmentation-demo-unet-msd-task06-lung-val-20261029T062135Z`.
- `nnunet-msd`: nnU-Net v2 `3d_fullres`, 100 epochs, trained on the same 37 fit patients in Colab (notebooks/nnunet_msd_task06_colab.ipynb). Not evaluated until its checkpoint is in artifacts/models/nnunet-msd/.
