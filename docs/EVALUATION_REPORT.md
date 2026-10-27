# OncoNova AI model evaluation report

Generated 2026-10-27 07:04 UTC by `make report` from the stored evaluation artifacts in `artifacts/evaluations/`. No number in this file was typed by hand.

> **Read this first.** Every model here was trained and evaluated on **synthetic data** (DEMO DATA): CT phantoms with exact generator ground truth, and a tabular cohort whose outcome was simulated from a known formula. The results show that the pipelines run end to end and recover planted signal. They say nothing about performance on patients. None of these models is validated for any clinical use.

## 1. Intended use

Research and teaching: exercising an imaging-AI workflow (ingestion, segmentation, longitudinal change, risk modelling, dose visualisation) with honest uncertainty and provenance. Not for diagnosis, prognosis, treatment planning or any decision about a person.

## 2. Supported cancer types and modalities

| Component | Cancer type | Modality | Status |
|---|---|---|---|
| Image ingestion and viewer | any | CT, MRI, PET (NIfTI or DICOM series) | Implemented; PET shown as stored values (no SUV) |
| Classical lesion heuristic | lung lesions in thoracic CT | CT (HU) | Tuned on phantoms only |
| 3-D U-Net (MONAI) | synthetic thoracic lesions | CT | Trained on phantoms only |
| Progression model | none (simulated cohort) | imaging-derived features | Simulated outcome only |
| Dose visualisation | n/a | any image grid | Geometric illustration (synthetic) or imported research dose |

## 3. Training data

- Segmentation: manifest `synthetic-seg-v1`, 60 synthetic patients x 2 scans, generator `thorax-phantom-0.1`, patient-level split 42/9/9 (seed 42). Labels: primary lesion, node-region lesion, other lesion (generator definitions, not pathology).
- Progression: `progression-cohort-v1`, 900 synthetic patients over 3 synthetic sites with simulated 180-day progression and time-to-event with censoring. Development split from sites A/B before 2021-01-01; temporal validation on later A/B enrolment; external validation on site C.

## 4. Exclusion criteria

- Progression binary metrics exclude patients censored before the 180-day horizon without an event (no defined label); they are kept for the C-index. Counts are in the table below.
- Segmentation class-wise metrics only use scans where the class exists in the reference.
- Inputs the models refuse: non-CT input for the CT models (error), missing longitudinal features for the progression model (explicit unavailable state).

## 5. Evaluation protocol

- Patient-level splits everywhere; leakage is checked by `check_no_leakage` and by tests.
- Segmentation: per-scan metrics against generator ground truth, averaged per patient, then mean/median with a 2000-sample patient bootstrap 95% CI. HD95 and Hausdorff in millimetres using voxel spacing. Specificity inside the body mask.
- Progression: decision threshold chosen on validation (Youden). AUROC, AUPRC, Brier, sensitivity/specificity with 1000-sample bootstrap CIs, Harrell's C (300 samples), quantile-bin calibration with ECE and logistic calibration slope/intercept, sex and site subgroups.

## 6. Measured results

### 6.1 Segmentation (held-out synthetic test patients)

### demo-classical (v0.1.0, classical-heuristic)

Report `segmentation-demo-classical-20261027T070417Z` · checkpoint `n/a` · 9 test patients / 18 scans from `synthetic-seg-v1`.

| Target | Scans | Dice mean (95% CI) | Dice median | IoU mean | HD95 mm mean / median | Sensitivity | Specificity (body) |
|---|---|---|---|---|---|---|---|
| Any lesion | 18 | 0.879 (0.827-0.916) | 0.890 | 0.791 | 19.5 / 2.2 | 0.791 | 1.0000 |
| Primary lesion | 18 | 0.910 (0.894-0.925) | 0.919 | 0.837 | 2.2 / 2.2 | 0.837 | 1.0000 |
| Node-region lesion | 4 | 0.000 (0.000-0.000) | 0.000 | 0.000 | n/a / n/a | 0.000 | 1.0000 |
| Other lesion | 3 | 0.588 (0.431-0.738) | 0.596 | 0.428 | 2.6 / 2.8 | 0.428 | 1.0000 |

Lesion-level detection: 21/25 reference lesions overlapped (sensitivity 0.840); 0.00 false-positive components per scan.

Voxel class confusion over all test scans (rows: reference, columns: prediction):

| Reference \ Predicted | background | primary_tumor | lymph_node_region_lesion | other_lesion |
|---|---|---|---|---|
| background | 21224360 (100%) | 0 (0%) | 0 (0%) | 0 (0%) |
| primary_tumor | 1263 (14%) | 7459 (86%) | 0 (0%) | 0 (0%) |
| lymph_node_region_lesion | 375 (100%) | 0 (0%) | 0 (0%) | 0 (0%) |
| other_lesion | 109 (53%) | 0 (0%) | 0 (0%) | 98 (47%) |

### demo-unet (v0.1.0, deep-learning)

Report `segmentation-demo-unet-20261027T070429Z` · checkpoint `b964d38dd42c8b7f` · 9 test patients / 18 scans from `synthetic-seg-v1`.

| Target | Scans | Dice mean (95% CI) | Dice median | IoU mean | HD95 mm mean / median | Sensitivity | Specificity (body) |
|---|---|---|---|---|---|---|---|
| Any lesion | 18 | 0.920 (0.876-0.947) | 0.941 | 0.857 | 19.3 / 2.2 | 0.946 | 0.9999 |
| Primary lesion | 18 | 0.940 (0.930-0.949) | 0.941 | 0.887 | 22.7 / 2.2 | 0.999 | 0.9999 |
| Node-region lesion | 4 | 0.000 (0.000-0.000) | 0.000 | 0.000 | n/a / n/a | 0.000 | 1.0000 |
| Other lesion | 3 | 0.000 (0.000-0.000) | 0.000 | 0.000 | n/a / n/a | 0.000 | 1.0000 |

Lesion-level detection: 21/25 reference lesions overlapped (sensitivity 0.840); 0.11 false-positive components per scan.

Voxel class confusion over all test scans (rows: reference, columns: prediction):

| Reference \ Predicted | background | primary_tumor | lymph_node_region_lesion | other_lesion |
|---|---|---|---|---|
| background | 21223437 (100%) | 922 (0%) | 1 (0%) | 0 (0%) |
| primary_tumor | 8 (0%) | 8714 (100%) | 0 (0%) | 0 (0%) |
| lymph_node_region_lesion | 375 (100%) | 0 (0%) | 0 (0%) | 0 (0%) |
| other_lesion | 51 (25%) | 155 (75%) | 1 (0%) | 0 (0%) |

### 6.2 Future progression risk (simulated cohort)

Report `progression-demo-20261027T062128Z` · artifact `0db41d16c04ce96d` · cohort `progression-cohort-v1` (n=900, DEMO DATA).

| Split | n (events) | AUROC (95% CI) | AUPRC (95% CI) | Brier (95% CI) | C-index (95% CI) | Sens / Spec @ threshold | ECE | Cal. slope |
|---|---|---|---|---|---|---|---|---|
| internal test | 76 (30) | 0.838 (0.740-0.918) | 0.777 (0.630-0.900) | 0.169 (0.126-0.217) | 0.759 (0.695-0.823) | 0.77 / 0.83 @ 0.518 | 0.167 | 0.94 |
| temporal validation | 163 (95) | 0.793 (0.721-0.856) | 0.859 (0.792-0.914) | 0.193 (0.165-0.224) | 0.705 (0.656-0.751) | 0.56 / 0.87 @ 0.518 | 0.106 | 1.09 |
| external validation site C | 178 (75) | 0.769 (0.698-0.833) | 0.707 (0.599-0.805) | 0.197 (0.167-0.231) | 0.739 (0.688-0.779) | 0.65 / 0.76 @ 0.518 | 0.061 | 0.79 |

**Subgroups** (pooled internal + temporal test sets):

| Subgroup | n | AUROC (95% CI) | Brier |
|---|---|---|---|
| sex=F | 107 | 0.797 (0.699-0.877) | 0.178 |
| sex=M | 132 | 0.801 (0.720-0.875) | 0.191 |
| site=site_A | 130 | 0.810 (0.731-0.875) | 0.177 |
| site=site_B | 109 | 0.781 (0.685-0.869) | 0.195 |

**Standardised coefficients** (association inside the model, not causation): `baseline_total_volume_cc` +0.80, `log_volume_ratio_per_100d` +0.78, `new_lesion_count` +0.67, `node_region_lesion_observed` +0.34, `missingindicator_age_years` -0.18, `missingindicator_primary_sphericity` +0.16, `age_years` +0.10, `primary_sphericity` -0.05

## 7. Known limitations

- **Synthetic only.** Phantoms are analytic shapes with simple noise; real CT has vessels, motion, artefacts, contrast phases, scanner variation and pathology the phantoms do not have.
- **Classical heuristic** never finds lesions outside the lung envelope (Dice 0 on node-region lesions by design) and misses lesions touching the pleura or mediastinum.
- **U-Net** (demo configuration, class-balanced crop sampling) gets class Dice 0 for node-region and 'other' lesions in this run. The confusion table in 6.1 shows where those voxels went: predicted as background (missed) or as 'primary' (mislabelled). Mislabelled distant lesions also explain the primary-class HD95 having a high mean but a low median.
- **Uncertainty** values are model-internal (heuristic entropy; MC-dropout entropy; bootstrap spread for the progression model). None is a calibrated probability of error.
- **Progression model** predicts a simulated outcome. Its discrimination only shows that the code recovers a signal deliberately planted in the generator. It has no metastasis-site labels and does not predict spread.
- **Saliency** is input-gradient sensitivity. It is not an explanation of reasoning or evidence of causation.
- **Dose** is a geometric illustration in relative units (no physics engine, no heterogeneity correction). Imported doses are displayed as stored and not verified.

## 8. Remaining research questions

- Does the pipeline transfer to a public, expert-annotated dataset (MSD Task06 Lung, NSCLC-Radiomics) under a patient-level split, and how large is the synthetic-to-real gap?
- Which uncertainty estimates (MC dropout, ensembles, test-time augmentation) are best calibrated against actual segmentation error on real data?
- Can longitudinal features predict a clinically defined endpoint (e.g. RANO progression in LUMIERE) with temporal and external validation, and how do imaging and clinical features compare?
- How should registration quality be scored so that change measurements are flagged when alignment fails?
