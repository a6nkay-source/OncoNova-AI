# Dataset setup guide

Checked 2026-10-26. Dataset papers were found through PubMed and each one was followed to its official archive: The Cancer Imaging Archive (TCIA) collection pages, the Medical Segmentation Decathlon release on the AWS Open Data registry, Figshare (via the paper) and the NCI Imaging Data Commons tooling. File sizes come from the archive pages or from HTTP `Content-Length` headers (HEAD requests only; nothing was downloaded). Anything that could not be confirmed is marked **not stated**. Licences and access terms change, so check them again before downloading. The same catalogue drives the Datasets tab of the Models & Data page (`configs/datasets.yaml`).

> Use only data you are authorised to use, follow each dataset's usage policy and citation requirements, and never place downloaded images in source control (`data/` is git-ignored).

## Ground truth: opinion versus confirmation

Most public "tumour labels" are expert opinions (contours or ratings), not pathology-confirmed voxel labels. That limits what a model trained on them can claim:

| Ground-truth type | What it supports | What it does not support |
|---|---|---|
| Radiologist/oncologist contour | Segmentation agreement with that expert | Proof the region is cancer; correctness beyond inter-rater variability |
| Radiologist suspicion rating (LIDC-IDRI) | Modelling radiologist perception | Malignancy classification (ratings are not diagnoses) |
| Pathology/histology-confirmed diagnosis | Patient-level diagnostic labels | Voxel-level boundaries |
| Automated segmentation (LUMIERE) | Benchmarking against another algorithm | Treating it as ground truth |

## Candidate datasets

| Dataset | Cancer / region | Modality, format | Patients · images | Annotations | Ground truth | Licence · access | Size | Paper |
|---|---|---|---|---|---|---|---|---|
| **MSD Task06 Lung** ⭐ | NSCLC, thorax | CT · NIfTI | 95 (63 train labelled / 32 test unlabelled) | Tumour masks (train) | Single human rater, **not pathology-confirmed** | CC BY-SA 4.0 · open, no registration | 9.16 GB tar | [PMID 35840566](https://doi.org/10.1038/s41467-022-30695-9) |
| NSCLC-Radiomics (Lung1) | NSCLC, thorax | CT, RTSTRUCT, SEG · DICOM; clinical CSV | 422 · 1,265 series / 52,073 images | GTV-1 by a radiation oncologist + lung, heart, oesophagus | Expert contour, **not pathology-confirmed**; survival outcomes | **CC BY-NC 3.0** (non-commercial) · open | 35.78 GB | [PMID 24892406](https://doi.org/10.1038/ncomms5006) |
| LIDC-IDRI | Lung nodules, thorax | CT (also DX, CR) · DICOM + XML | 1,010 (table; description says 1,018 cases) · 1,308 series / 244,527 images | Outlines and ratings from 4 radiologists, two-phase read | **Radiologist opinion**; limited diagnosis subset (biopsy, resection, 2-year stability or unknown), pathology-confirmed share not stated | CC BY 3.0 · open | 133.16 GB | [PMID 21452728](https://doi.org/10.1118/1.3528204) |
| LCTSC | Thoracic cancer patients (OARs only) | CT, RTSTRUCT · DICOM | 60 · 120 series / 9,593 images | Lungs, heart, oesophagus, spinal cord | Clinical contours edited to RTOG 1106 | CC BY 3.0 · open | 5.15 GB | [PMID 30144101](https://doi.org/10.1002/mp.13141) |
| NSCLC Radiogenomics | NSCLC, thorax | CT, PET/CT, SEG · DICOM; AIM XML; CSV | 211 · 1,351 series / 286,754 images | Tumour masks (count of subjects with masks not stated) | Annotator type not stated; genomics from surgically excised tissue | CC BY 3.0 · open | 98 GB | [PMID 30325352](https://doi.org/10.1038/sdata.2018.202) |
| Head-Neck-PET-CT | Head and neck | PET, CT, RTSTRUCT, **RTPLAN, RTDOSE** · DICOM | 298 · 2,661 series / 123,271 images | GTV primary, GTV nodes | Clinical planning contours; outcomes: 40 distant metastases, 45 locoregional recurrences, 56 deaths (of 300) | Images under **NIH controlled access**; clinical data CC BY 3.0 | 72.46 GB | [PMID 28860628](https://doi.org/10.1038/s41598-017-10371-5) |
| LUMIERE | Glioblastoma, brain | MRI (T1, T1c, T2, FLAIR) · NIfTI, CSV | 91 · 638 study dates / 2,487 images | **Automated** segmentations (DeepBraTumIA, HD-GLIO-AUTO); expert RANO ratings for 616 timepoints | Expert RANO response (1 neuroradiologist); MGMT/IDH1 for a subset; survival | **Non-commercial only** (exact licence name not stated in the text reviewed) · Figshare | not stated | [PMID 36522344](https://doi.org/10.1038/s41597-022-01881-7) |
| BraTS | Glioma, brain | mpMRI · NIfTI | Varies by edition | Tumour sub-regions | Manual, up to 4 raters (inter-rater Dice 74-85% in 2012-13) | Edition-specific; not stated | MSD Task01 tar: 7.61 GB | [PMID 25494501](https://doi.org/10.1109/TMI.2014.2377694) |

Archive references: TCIA ([PMID 23884657](https://doi.org/10.1007/s10278-013-9622-7)) and the NCI Imaging Data Commons (`idc-index` 0.13.4 on PyPI, released 2026-10-20, Python ≥ 3.10, no credentials required). Dataset DOIs: NSCLC-Radiomics 10.7937/K9/TCIA.2015.PF0M9REI, LIDC-IDRI 10.7937/K9/TCIA.2015.LO9QL9SX, LCTSC 10.7937/K9/TCIA.2017.3R3FVZ08, NSCLC Radiogenomics 10.7937/K9/TCIA.2017.7hs46erv, Head-Neck-PET-CT 10.7937/K9/TCIA.2017.8oje5q00.

**Discrepancies found while checking:** LIDC-IDRI's page gives 1,010 subjects in its table and 1,018 cases in its description. The AWS registry entry for MSD says "CC-BY-SA 4.0 International" but links to the CC BY 4.0 URL; the MSD paper says CC-BY-SA 4.0. Head-Neck-PET-CT images are controlled access, even though its clinical spreadsheets are CC BY 3.0.

Mapping to modules: segmentation (B) → MSD Task06, NSCLC-Radiomics, LCTSC (organs at risk). Longitudinal and progression (C) → LUMIERE (serial MRI with expert response labels). Survival → NSCLC-Radiomics. Research dose import (D) → Head-Neck-PET-CT (RTDOSE), controlled access.

## Recommended starter: MSD Task06 Lung

Why this one:

1. **Laptop-sized:** one 9.16 GB tarball, 95 volumes, versus 35-133 GB for the TCIA collections.
2. **Ready-made masks in NIfTI:** loads straight into OncoNova (`load_nifti`) with no RTSTRUCT/SEG conversion, matching the thoracic-CT domain of the demo models.
3. **Fully open:** no registration, CC BY-SA 4.0 (share-alike obligations apply to redistributed derivatives).

Caveats: labels come from a single rater and are not pathology-confirmed. The 32 test cases have no released labels, so all evaluation uses a patient-level split of the 63 labelled cases (the release's `dataset.json` says 63; an earlier version of this guide said 64). MSD lung data came from TCIA, so do not combine it with other TCIA lung collections without checking for shared patients, or the same patient could leak between training and test.

### Download and import

```bash
mkdir -p data/external/msd
```

```bash
curl -L -o data/external/msd/Task06_Lung.tar https://msd-for-monai.s3-us-west-2.amazonaws.com/Task06_Lung.tar
```

Check it before extracting: 9,163,696,640 bytes, MD5 `8afd997733c7fc0432f71255ba4e52dc` (the value MONAI's `DecathlonDataset` checks).

```bash
md5 -q data/external/msd/Task06_Lung.tar
```

Extract with Python's path-checking `data` filter, which refuses absolute paths, `..` and links:

```bash
.venv/bin/python -I -c "import tarfile; tarfile.open('data/external/msd/Task06_Lung.tar').extractall('data/external/msd', filter='data')"
```

```bash
cd backend && ../.venv/bin/python scripts/import_msd.py ../data/external/msd/Task06_Lung --manifest-id msd-task06-lung
```

The importer reads `dataset.json` and validates every training pair: readable NIfTI, image/label geometry, allowed label values. It records spacing and lesion voxel counts, excludes failures with a reason, assigns a seeded patient-level split, and writes `data/external/msd-task06-lung/manifest.json`. With seed 42 the split is 44 train, 9 val and 10 test patients, and none were excluded. The volumes are 512×512×(112-636) voxels at 0.60-0.98 mm in-plane and 0.62-2.5 mm between slices; median tumour volume is 5.8 cc in the train split (smallest 0.74 cc).

### Train and compare (Stage 3)

```bash
cd backend && ../.venv/bin/python scripts/train_segmentation_msd.py
```

This trains three seeds of the U-Net on 37 train patients, chooses each checkpoint on 7 more, and writes an ensemble card. On an M2 it takes about 2 hours per seed while the Mac is awake. It caches the resampled scans in `data/external/msd-task06-lung/cache-*`, about 2 GB.

```bash
cd backend && ../.venv/bin/python scripts/evaluate_segmentation.py --dataset msd-task06-lung --split val --checkpoint-dir ../artifacts/models/seg-unet-msd-ens
```

```bash
cd backend && ../.venv/bin/python scripts/build_segmentation_comparison.py
```

The `test` split is locked: the evaluation script refuses it without `--unlock-test`. nnU-Net is trained separately in Colab (`notebooks/nnunet_msd_task06_colab.ipynb`) on the same patients. Results are in [SEGMENTATION_COMPARISON.md](SEGMENTATION_COMPARISON.md).

### Next: NSCLC-Radiomics subsets via IDC

For survival outcomes and organ-at-risk contours (Modules C and D) without the full 35.78 GB, `idc-index` can download a chosen subset of patients:

```bash
.venv/bin/pip install "idc-index>=0.13"
```

```python
from idc_index import IDCClient
client = IDCClient.client()
# Confirm the collection id first (IDC ids are lowercase, e.g. "rider_pilot"):
print([c for c in client.get_collections() if "nsclc" in c])
# Inspect before downloading: series and sizes for ten patients
df = client.sql_query("""
  SELECT PatientID, Modality, SeriesInstanceUID, series_size_MB
  FROM index WHERE collection_id = 'nsclc_radiomics'
  AND PatientID IN (SELECT DISTINCT PatientID FROM index WHERE collection_id = 'nsclc_radiomics' ORDER BY PatientID LIMIT 10)""")
print(df.groupby("Modality")["series_size_MB"].sum())
# Then download exactly those series:
# client.download_dicom_series(seriesInstanceUID=df["SeriesInstanceUID"].tolist(), downloadDir="data/external/nsclc")
```

Column names and `download_dicom_series` follow the `idc-index` README. The collection id is an assumption, so check it with `get_collections()` first.

The licence is **CC BY-NC 3.0 (non-commercial)** and the data citation is required. Converting RTSTRUCT/SEG to masks is not yet implemented (see ROADMAP).

## Synthetic data (built in)

| Manifest | Contents | How to regenerate |
|---|---|---|
| `synthetic-seg-v1` | 60 phantom patients × 2 CT scans with generator ground truth (primary, node-region, other lesion; lungs, heart, oesophagus, cord) | `make train-seg` (generates if missing) |
| `progression-cohort-v1` | 900 synthetic patients, 3 sites, simulated 180-day outcome and time-to-event | `make train-progression` |
| Demo cases `DEMO-001…005` | Five longitudinal phantom cases shown in the UI | Automatic at API start-up |

All are deterministic from their seeds and labelled DEMO DATA.
