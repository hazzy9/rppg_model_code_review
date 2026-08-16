## Figure Reproduction Code

Code to reproduce every quantitative figure in the manuscript (Figures 1–3) from the processed data included in this repository.

Scope note: this repository covers figure reproduction only — it takes already-processed, derived data (PCA scores, spectrogram averages, bootstrap summary statistics) and turns it into the published plots. It does not include the upstream rPPG signal-processing or model-training pipeline that produced that derived data in the first place, since that data is not something we can share in raw form (see Data below).

### Contents
- figures_code_review.py # all Python figures (1A, 1C, 2A-G, 3A, 3B-D)
- figure1B_clarke_error_grid.R # Figure 1B (R, run separately)

- figure1_source_data.csv # Fig 1A, 1B
- figure1C_source_data.csv # Fig 1C
- figure2_source_data.csv # Fig 2A, 2B
- figure2cd_source_data.npz # Fig 2C, 2D
- figure2efg_source_data.npz # Fig 2E, 2F, 2G
- figure3_source_data.npz # Fig 3A, 3B-D
- README.md                                                   


### Data

Every file above is either:

Population-level, not per-subject (figure2cd_source_data.npz, figure2efg_source_data.npz's group-mean/stability arrays, phase1_..._summary.csv) — fitted-model parameters or across-subject bootstrap statistics, nothing traceable to an individual, or
Per-subject but capped and stratified (figure1_source_data.csv, figure2_source_data.csv, the per-subject arrays in figure2efg_source_data.npz, figure3_source_data.npz) — a 100-recording subsample, stratified by FPG range to preserve the original cohort's proportions, with no raw signal, timestamp, or subject identifier included.

No raw video, rPPG waveform, or personally identifying field is present in any file in this repository to protect individual privacy rights.


### System requirements

Python (Figures 1A, 1C, 2A–G, 3A, 3B–D)

Python 3.9 or later (tested on 3.12)
numpy, pandas, matplotlib, seaborn, scipy — no version pins required; tested against numpy 2.4, pandas 3.0, matplotlib 3.10, seaborn 0.13, scipy 1.17

R (Figure 1B only)

R 4.x
Packages: ega, ggplot2, dplyr, tidyr

Operating system: developed and tested on Linux; no OS-specific code paths, so macOS and Windows (with R/Python installed) are expected to work identically. No GPU or other non-standard hardware is needed — every script here runs on derived, already-small data.


### Installation guide
bash
git clone <this repository>
cd <repository>

## Python side
pip install numpy pandas matplotlib seaborn scipy

## R side (only needed for Figure 1B)
Rscript -e 'install.packages(c("ega", "ggplot2", "dplyr", "tidyr"))'

Typical install time on a normal desktop with a working internet connection: under 5 minutes for the Python packages; the R packages (particularly ega, which compiles from source on some platforms) can take up to 10–15 minutes on a first install.


- Demo

The data files in this repository are the demo dataset — there's no separate synthetic version, since none of them contain information sensitive enough to need one (see Data, above).


- To reproduce every Python figure:

bash
python3 figures_code.py

This produces 13 PNG files in the working directory: figure1A_agreement_plot.png, figure1C_calibration_stability.png, figure2A_pca_scatter.png through figure2G_quadratic_fit.png, figure3A_avg_tfm.png, and figure3B_cohens_d.png / figure3C_cohens_d.png / figure3D_cohens_d.png.

- To reproduce Figure 1B:

bash
Rscript figure1B_clarke_error_grid.R

This produces figure1B_clarke_error_grid.png plus a printed Clarke-zone summary table (points, %, A+B cumulative %) in the console.

Expected run time: under 15 seconds for all Python figures combined, on a normal desktop, no GPU. The R script runs in a few seconds.

Expected output: figures matching those published in the manuscript. Since the shared data is a 100-recording stratified subsample rather than the full cohort, exact pixel values (point positions, regression coefficients, color-map ranges) will differ slightly from the published version, but the overall pattern — direction of correlations, cluster locations, effect-size signs — should match.

- Instructions for use
Reproducing the manuscript figures

Each block in figures_code.py is self-contained and independent of the others — run the whole file for all figures, or copy out a single block to regenerate just one. Every block states at the top which data file it reads and which image it writes.

- Figure	Data file	Script
1A	figure1_source_data.csv	figures_code.py
1B	figure1_source_data.csv	figure1B_clarke_error_grid.R
1C	phase1_decay_uncal_vs_frozen_vs_decayed_summary.csv	figures_code.py
2A, 2B	figure2_source_data.csv	figures_code.py
2C, 2D	figure2cd_source_data.npz	figures_code.py
2E, 2F, 2G	figure2efg_source_data.npz	figures_code.py
3A, 3B–D	figure3_source_data.npz	figures_code.py
Running on your own data

Each block reads a plainly-schemad CSV or NPZ — to reuse a block on a new dataset, format your own data to match the columns/keys documented in that block's header comment (e.g. Figure 2A/2B need pc1, pc2, fpg columns; Figure 3A/3B–D need images (N,64,64,C), fpg_bin3 (N,), channels (C,) in an .npz). No part of the plotting code is hardcoded to this study's cohort beyond the FPG bin edges (<100 / 100–125 / ≥126 mg/dL, the standard ADA fasting-glucose categories), which can be edited directly in the script if a different binning is needed.
