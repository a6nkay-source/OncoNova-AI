# Research roadmap

Ordered by how directly each item reduces the gap between this prototype and credible research results.

## Next (smallest steps with the most value)

1. **Train on MSD Task06 Lung.** The importer and manifest exist. Add resampling to a common spacing (e.g. 1.5 mm isotropic), lung-window normalisation, patch sampling with foreground bias, and sliding-window inference. Report test-split metrics through the same evaluation script and compare against the classical baseline.
2. **Fix the U-Net's rare classes.** Node-region and "other" lesions get Dice 0 now. Try class-weighted Dice/CE or focal loss, more training patients with those classes, and a dedicated lesion-detection metric (FROC) per class.
3. **DICOM SEG and RTSTRUCT import.** Convert expert contours (NSCLC-Radiomics GTV-1, LCTSC organs at risk) to label maps on the image grid, so real reference masks and organs at risk can feed Modules B and D.
4. **Uncertainty calibration.** Measure whether high-uncertainty voxels and regions actually coincide with segmentation errors (error-uncertainty AUROC, reliability diagrams). Compare MC dropout with deep ensembles and test-time augmentation.

## Then

5. **Real longitudinal labels.** Use LUMIERE (serial GBM MRI with expert RANO ratings) to train and temporally validate a progression model on a clinically defined endpoint. Respect its non-commercial terms.
6. **Survival modelling.** Add Cox and discrete-time models with time-dependent AUC and integrated Brier score. Use NSCLC-Radiomics overall survival with external validation on a second collection.
7. **Deformable registration** for lung follow-up (breathing), with a registration-quality score that blocks change measurements when alignment is poor.
8. **Image-based OOD detection** (feature-space density or Mahalanobis on encoder features) to replace the current spacing/intensity range checks.

## Later

9. **Validated external research dose engine.** Integrate an established open-source research dose engine as a separate, sandboxed service. Keep everything read-only and labelled research-only. Never add device or record-and-verify interfaces.
10. **Cohort and annotation tooling:** multi-rater annotation, inter-rater agreement, adjudication workflow, annotation provenance.
11. **Explainability studies:** compare saliency methods against perturbation tests, and report where saliency fails sanity checks.
12. **Multi-user deployment:** database-backed store, per-project access control, encrypted storage, structured audit export.

## Open research questions

- How large is the synthetic-to-real gap for each module, and which phantom realism (vessels, texture, motion, contrast) narrows it?
- Do growth-rate features from automatic segmentation predict progression as well as features from expert contours?
- Can an uncertainty threshold usefully select scans for expert review, and at what workload?
- Which metastasis-pattern questions can public data answer at all, given how few datasets have confirmed metastasis-site labels with external validation?
