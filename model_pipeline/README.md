# FPG Estimation from rPPG: Nested Cross-Validation Pipeline

Reproduces the results reported in the manuscript: Biphasic Spectral Signatures Enable Non-Linear Glucose Estimation from Smartphone Photoplethysmography.

## What's included

| File | Manuscript section |
|---|---|
| `processing.py` | Implements Remote PPG Signal Extraction to produce 7 channels|
| `preprocessing.py` | Signal Preprocessing (Steps 1-4) |
| `sqi.py` | Signal-to-noise calculation functions for channel selection) |
| `features.py` | Feature Engineering: Temporal Frequency Map |
| `model.py` | Model Architecture |
| `tuning.py` | Hyperparameter tuning (nested CV inner loop) |
| `train_eval.py` | Cross-Validation and Evaluation |
| `eval.py` | Cross-Validation and Evaluation (clinical accuracy metrics) |
| `fpg_utils.py` | shared utilities |
| `run_pipeline.py` | orchestrates all of the above, end to end |
| `dataset_df.parquet` | de-identified signal + demographic subset (see below) |


## What's *not* included, and why

Raw RGB recordings are not released, in line with participant
privacy protections under the study's IRB approval — consent for research
use does not extend to public redistribution of raw user data. `dataset_df.parquet`
contains only de-identified, already-extracted numeric rPPG signal for a subset of recordings, sufficient to run this pipeline end to end. Raw video loading and file-selection code (upstream of `dataset_df.parquet`) is also not included, since it operates only on data that isn't public, and it may reveal information about the raw dataset that is subject to strict confidentiality.

## `dataset_df.parquet` columns

| Column | Meaning |
|---|---|
| `Patient_ID` | Per-recording identifier (used for logging, not for grouping) |
| `license` | Per-subject identifier — recordings sharing this value are the same person. All train/validation/test splits are grouped by this column so that one subject's recordings never cross partitions |
| `time_cumulative`, `crcb`, `green`, `pos`, `pbv`, `crcg`, `cr/cg`, `ycgco` | Raw, pre-processing rPPG signal (list-valued per recording) |
| `age`, `gender` | Demographic covariates used as model input |
| `fpg` | Reference fasting plasma glucose (mg/dL) — the prediction target |
| `fpg_category` | Precomputed stratum: `<100`, `100-125`, `>=126` mg/dL |


## Setup

```bash
pip install -r requirements.txt
```

## Running

```bash
python run_pipeline.py
```

**To reproduce the paper's reported results**, 
By default, run_pipeline.py runs in quick-test mode (QUICK_TEST = True at the top of the file) — a fast, minimized-scale run (2 outer folds, 2 inner folds, 3 HPO trials) that confirms the pipeline runs end to end in well under a minute.

To reproduce the paper's actual reported results, edit run_pipeline.py and set QUICK_TEST = False. This runs the real configuration — 5 outer folds, 3 inner folds, 80 HPO trials per outer fold — and will take substantially longer (each outer fold runs an independent 80-trial search before its final model is trained - could take from several minutes to a couple of hours). 
Outputs (pooled metrics, per-fold hyperparameters, clinical accuracy tables, and trained model checkpoints) are written to final_baseline_outputs/ and cv_saved_models/.

Note on the included subset: dataset_df.parquet contains 76 recordings from 30 subjects — enough to exercise the full pipeline end to end, but too few for the real configuration to produce numbers resembling the paper's reported results, which are computed over the full dataset. Running the real config on this subset will complete without error, but treat its output as a correctness check of the code, not a reproduction of the results reported.


## Methodology notes

- **Nested cross-validation**: To prevent optimistic bias, hyperparameters are tuned independently using only the training data within each of the five outer folds (Cawley & Talbot, 2010, *JMLR*).
- **Subject-level splitting**: all recordings from a given subject are kept strictly within one partition (train, validation, or test) at every level of splitting, preventing subject-level leakage.


