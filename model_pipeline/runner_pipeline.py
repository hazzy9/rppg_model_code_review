"""
runner_pipeline.py

Purpose: Runs the following end-to-end pipeline:
- raw rPPG signal preprocessing
- SQI - based channel selection
- TF-Map feature engineering
- Nested cross-validation (includes hyperparameter optimization + training + evaluation)

Dataset: dataset_df.parquet 
    - contains the data used as inputs to the model:
    Includes the 7 rPPG signals extracted from raw RGB data + demographics (see README.md for details)

Usage of the runner code:
python runner_pipeline.py
"""

import os

import numpy as np
import pandas as pd

# importing relevant functions
from preprocessing import (
    MIN_SPAN_SEC, MAX_STEP_SEC, SIGNAL_COLS,
    filter_and_crop_tierA, trim_each_end,
    apply_temporal_filtering, integrity_filter,
)
from sqi import sliding_windows, compute_psd, find_f0, compute_sqi
from train_eval import run_final_baseline_3tfm


# Configurations
DATASET_PATH = "dataset_df.parquet"
FPS = 30
CHANNELS_ALL = ["crcb", "green", "pos", "pbv", "crcg", "cr/cg", "ycgco"]
CHANNELS_IMG = ["crcb", "pos", "pbv"]  #top-SQI channels 

SAVE_DIR = "final_baseline_outputs"
SAVE_MODEL_DIR = "cv_saved_models"

QUICK_TEST = True # set to False for the real run (HPO: 5 outer folds / 3 inner folds / 80 trials)
N_QUICK_LICENSES = 10 #note: dataset_df contains N=30

# SQI-related
SQI_FS = 30
SQI_WIN_SIZE = SQI_FS * 30   # 30s
SQI_STEP = SQI_FS * 15

def extract_record_sqi(signal):
    if signal is None or len(signal) < SQI_WIN_SIZE:
        return np.nan
    windows = sliding_windows(signal, SQI_WIN_SIZE, SQI_STEP)
    sqis = []
    for w in windows:
        w = w - np.mean(w)
        freqs, psd = compute_psd(w, SQI_FS)
        f0 = find_f0(freqs, psd)
        sqi = compute_sqi(freqs, psd, f0)
        if not np.isnan(sqi):
            sqis.append(sqi)
    return np.median(sqis) if sqis else np.nan


# main runner
def main():
    #load dataset
    merged_df = pd.read_parquet(DATASET_PATH)
    print(f"Loaded {DATASET_PATH}: {len(merged_df)} rows, "
          f"{merged_df['license'].nunique()} subjects")
    
# Signal Preprocessing (Corresponds to manuscript "Methods - Signal Preprocessing - Steps 1~4" in order)
    df, tierA_stats = filter_and_crop_tierA(merged_df, min_span_sec=MIN_SPAN_SEC, max_step_sec=MAX_STEP_SEC)
    print("Step 1 (tierA crop):", tierA_stats)
    
    df, trim_stats = trim_each_end(df, trim_sec=3.0)
    print("Step 2 (edge trim):", trim_stats)

    df, filter_stats = apply_temporal_filtering(df, fps=FPS, channels=SIGNAL_COLS)
    print("Step 3 (temporal filtering):", filter_stats)

    df, integrity_fail_log, integrity_stats = integrity_filter(df, channels=SIGNAL_COLS, time_col="time_cumulative")
    print("Step 4 (integrity check):", integrity_stats)
    for pid, ch, reason in integrity_fail_log[:20]:
        print(f"  excluded: {pid} / {ch}: {reason}")

    
    # SQI - calculation (from which the three channels were chosen)
    for ch in CHANNELS_ALL:
        df[f"SQI_{ch}"] = df[ch].apply(extract_record_sqi)
    
    sqi_cols = [f"SQI_{ch}" for ch in CHANNELS_ALL]
    sqi_summary = df[sqi_cols].describe()
    print("\nPer-channel SQI summary (Supplementary Table 6):")
    print(sqi_summary)
    os.makedirs(SAVE_DIR, exist_ok = True)
    sqi_summary.to_csv(f"{SAVE_DIR}/sqi_summary.csv")

    # final dataset for HPO + training + evaluation
    df_proc = df.reset_index(drop = True)
    print(f"\ndf_proc: {len(df_proc)} rows, {df_proc['license'].nunique()} subjects") #check readings & subjects count

    #sanity check
    required_cols = {"license", "age", "gender", "fpg", *CHANNELS_IMG}
    missing_cols = required_cols - set(df_proc.columns)
    assert not missing_cols, f"df_proc is missing required columns: {missing_cols}"


    # Run Nested CV (includes: HPO + model training + evaluation)
    if QUICK_TEST:
        print(f"\n{'='*60}\RUNNING A QUICK TEST -- abbreviated settings, ~{N_QUICK_LICENSES} subjects\n{'='*60}")
        smoke_licenses = df_proc["license"].drop_duplicates().iloc[:N_QUICK_LICENSES]
        df_run = df_proc.loc[df_proc["license"].isin(smoke_licenses)].reset_index(drop=True)
        n_outer, n_inner, n_trials = 2, 2, 3
        save_csv = False
    else:
        print(f"\n{'='*60}\nFULL RUN -- 5 outer folds, 3 inner folds, 80 HPO trials\n{'='*60}")
        df_run = df_proc
        n_outer, n_inner, n_trials = 5, 3, 80
        save_csv = True
 
    final_out = run_final_baseline_3tfm(
        df_proc=df_run,
        channels_img=CHANNELS_IMG,
        n_outer_folds=n_outer,
        n_inner_folds=n_inner,
        n_hpo_trials=n_trials,
        save_csv=save_csv,
        save_dir=SAVE_DIR,
        save_model_dir=(None if QUICK_TEST else SAVE_MODEL_DIR),
    )
 
    print("\n=== Overall metrics ===")
    for k, v in final_out["paper_metrics"].items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
 
    print("\n=== Per-outer-fold breakdown ===")
    print(final_out["per_fold_metrics"])
 
    print("\n=== Hyperparameters selected per outer fold ===")
    print(final_out["hp_per_fold"])
 
    print("\n=== Clarke Error Grid ===")
    for k, v in final_out["clarke"].items():
        print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")
 
    return final_out

 
 
if __name__ == "__main__":
    main()


