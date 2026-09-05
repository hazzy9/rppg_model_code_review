"""
Figure reproduction code for peer review.

Every block below is self-contained - point it at the matching data file
(all included alongside this script) and run it on its own. Blocks don't
depend on each other or on execution order, so any single block can be
pulled out and run in isolation too.

Figure 1B is R, not Python: see figure1B_clarke_error_grid.R.

Data files used below:
  figure1_source_data.csv                              -> Fig 1A, 1B
  figure1C_source_data.csv                             -> Fig 1C
  figure2_source_data.csv                              -> Fig 2A, 2B
  figure2cd_source_data.npz                            -> Fig 2C, 2D
  figure2efg_source_data.npz                           -> Fig 2E, 2F, 2G
  figure3_source_data.npz                              -> Fig 3A, 3B-D
"""

# ============================================================
# Figure 1A - agreement between reference and predicted FPG
# ============================================================
# reads:  figure1_source_data.csv (y_true, y_pred_oof; n=100)
# writes: figure1A_agreement_plot.png

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

details_df = pd.read_csv("figure1_source_data.csv")

y_true = details_df["y_true"].to_numpy(float)
y_pred = details_df["y_pred_oof"].to_numpy(float)

vmin = np.min(np.concatenate([y_true, y_pred]))
vmax = np.max(np.concatenate([y_true, y_pred]))
pad = 0.05 * (vmax - vmin)
vmin -= pad
vmax += pad

fig, ax = plt.subplots(figsize=(5.5, 5.5))
ax.scatter(y_true, y_pred, s=18, alpha=0.6, edgecolor="none")
ax.plot([vmin, vmax], [vmin, vmax], color="black", linewidth=1.5, label="Identity (y = x)")
ax.set_xlim(vmin, vmax)
ax.set_ylim(vmin, vmax)
ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("Reference fasting plasma glucose (mg/dL)")
ax.set_ylabel("Predicted fasting plasma glucose (mg/dL)")
ax.legend(frameon=False, loc="lower right")

plt.tight_layout()
plt.savefig("figure1A_agreement_plot.png", dpi=300)
# plt.show()


# ============================================================
# Figure 1B - Clarke Error Grid
# ============================================================
# R, not Python - run figure1B_clarke_error_grid.R.
# Same source file as 1A: figure1_source_data.csv (y_true, y_pred_oof).


# ============================================================
# Figure 1C - calibration stability over time
# ============================================================
# reads:  figure1C_source_data.csv
# writes: figure1C_calibration_stability.png

import pandas as pd
import matplotlib.pyplot as plt

TIME_BIN_ORDER = ["0\u20133", "4\u20137", "8\u201314", "15\u201330", "31\u201360"]

df = pd.read_csv("figure1C_source_data.csv")
df["time_bin"] = pd.Categorical(df["time_bin"], categories=TIME_BIN_ORDER, ordered=True)
df = df.sort_values("time_bin")
x = df["time_bin"]

plt.figure(figsize=(9, 5))


def errbar(mean_col, lo_col, hi_col, label, color):
    y = df[mean_col]
    yerr = [y - df[lo_col], df[hi_col] - y]
    plt.errorbar(x, y, yerr=yerr, marker="o", capsize=3, label=label, color=color)


errbar("MARD_uncal_mean", "MARD_uncal_ci_low", "MARD_uncal_ci_high", "General (uncalibrated)", "gray")
errbar("MARD_decay_hl30_mean", "MARD_decay_hl30_ci_low", "MARD_decay_hl30_ci_high", "Decayed (half-life 30d)", "tab:green")

plt.xlabel("Days since last calibration")
plt.ylabel("MARD (%)")
plt.title("Calibration Stability Over Time", fontsize=14, fontweight="bold")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figure1C_calibration_stability.png", dpi=300, bbox_inches="tight")
# plt.show()


# ============================================================
# Figure 2A - PCA embedding (PC1 vs PC2)
# ============================================================
# reads:  figure2_source_data.csv (pc1, pc2, fpg; n=100)
# writes: figure2A_pca_scatter.png

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def assign_range(fpg):
    if fpg < 100:
        return "<100"
    elif fpg < 126:
        return "100-125"
    else:
        return ">=126"


df = pd.read_csv("figure2_source_data.csv")
df["fpg_bin"] = df["fpg"].apply(assign_range)

palette = {
    "<100": "#2ca02c",     # green
    "100-125": "#1f77b4",  # blue
    ">=126": "#d62728",    # red
}

plt.figure(figsize=(6, 5))

for bin_name, color in palette.items():
    sub = df[df["fpg_bin"] == bin_name]
    if len(sub) == 0:
        continue

    f = sub["fpg"].to_numpy()

    if bin_name == "<100":
        f_min, f_max = f.min(), f.max()
        alphas = 0.25 + 0.6 * (1 - (f - f_min) / max(f_max - f_min, 1e-6))
    elif bin_name == "100-125":
        center, half_width = 112.5, 12.5
        alphas = 0.25 + 0.6 * (1 - np.abs(f - center) / half_width)
        alphas = np.clip(alphas, 0.25, 0.85)
    else:  # >=126
        f_min, f_max = f.min(), f.max()
        norm = (f - f_min) / max(f_max - f_min, 1e-6)
        alphas = 0.45 + 0.4 * np.sqrt(norm)

    plt.scatter(sub["pc1"], sub["pc2"], s=40, c=color, alpha=alphas, edgecolors="none", label=bin_name)

plt.axhline(0, color="gray", lw=0.5, alpha=0.4)
plt.axvline(0, color="gray", lw=0.5, alpha=0.4)

plt.title("PCA", fontsize=15, fontweight="bold")
plt.xlabel("PC1")
plt.ylabel("PC2")

legend_handles = [
    Line2D([0], [0], marker="o", linestyle="", markersize=8,
           markerfacecolor=color, markeredgecolor=color, alpha=1.0, label=label)
    for label, color in palette.items()
]
plt.legend(handles=legend_handles, title="Glucose range", frameon=False)

plt.tight_layout()
plt.savefig("figure2A_pca_scatter.png", dpi=300, bbox_inches="tight")
# plt.show()


# ============================================================
# Figure 2B - correlation between PC1/PC2 and FPG
# ============================================================
# reads:  figure2_source_data.csv (same file as 2A)
# writes: figure2B_correlation.png

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

df = pd.read_csv("figure2_source_data.csv")

pc1 = df["pc1"].to_numpy()
pc2 = df["pc2"].to_numpy()
fpg = df["fpg"].to_numpy()

r1, p1 = pearsonr(pc1, fpg)
r2, p2 = pearsonr(pc2, fpg)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, pc, label, r, p in [(axes[0], pc1, "PC1", r1, p1), (axes[1], pc2, "PC2", r2, p2)]:
    ax.scatter(fpg, pc, alpha=0.35, s=18)
    sns.regplot(x=fpg, y=pc, scatter=False, line_kws=dict(color="black"), ax=ax)
    ax.set_xlabel("FPG (mg/dL)")
    ax.set_ylabel(label)
    ax.text(0.98, 0.98, f"R={r:.2f}\nP={p:.1e}", transform=ax.transAxes, ha="right", va="top")

fig.suptitle("Correlation between PC & FPG", fontsize=15, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig("figure2B_correlation.png", dpi=300, bbox_inches="tight")
# plt.show()


# ============================================================
# Figure 2C - frequency importance (PC1 vs PC2 loadings)
# ============================================================
# reads:  figure2cd_source_data.npz
#         components_top2  (2, n_features_kept) - PCA loadings for PC1, PC2
#         support_mask      (n_full,) bool        - which features survived the variance filter
#         scale_             (n_full,)             - StandardScaler.scale_ from the fit
#         tfm_shape          (C, F, T)
#         freqs              (F,) Hz
# writes: figure2C_frequency_importance.png

import numpy as np
import matplotlib.pyplot as plt

FMAX = 4.0


def reconstruct_loadings(components_top2, scale_, support_mask, tfm_shape):
    # undo the variance filter + standardization to get back to (PC, C, F, T)
    C, F, T = tfm_shape
    n_full = scale_.shape[0]
    full = np.zeros((components_top2.shape[0], n_full))
    full[:, support_mask] = components_top2
    full = full / scale_
    return full.reshape(-1, C, F, T)


data = np.load("figure2cd_source_data.npz", allow_pickle=True)
components_top2 = data["components_top2"]
support_mask = data["support_mask"]
scale_ = data["scale_"]
C, F, T = data["tfm_shape"]
freqs = data["freqs"]

L = reconstruct_loadings(components_top2, scale_, support_mask, (C, F, T))
mask = (freqs >= 0) & (freqs <= FMAX)

fig, ax = plt.subplots(figsize=(5, 3))
for pc_idx, label in [(0, "PC1"), (1, "PC2")]:
    fi = np.abs(L[pc_idx]).mean(axis=(0, 2))
    ax.plot(freqs[mask], fi[mask], lw=2, label=label)

ax.set_xlim(0, FMAX)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("|loading| averaged over time")
ax.set_title("Frequency importance", fontsize=13, fontweight="bold")
ax.grid(alpha=0.3)
ax.legend(frameon=False)

plt.tight_layout()
plt.savefig("figure2C_frequency_importance.png", dpi=300, bbox_inches="tight")
# plt.show()


# ============================================================
# Figure 2D - time-frequency PCA loading maps
# ============================================================
# reads:  figure2cd_source_data.npz (same file and schema as 2C, above)
# writes: figure2D_time_freq_loading.png

import numpy as np
import matplotlib.pyplot as plt

FMAX = 4.0


def reconstruct_loadings(components_top2, scale_, support_mask, tfm_shape):
    C, F, T = tfm_shape
    n_full = scale_.shape[0]
    full = np.zeros((components_top2.shape[0], n_full))
    full[:, support_mask] = components_top2
    full = full / scale_
    return full.reshape(-1, C, F, T)


data = np.load("figure2cd_source_data.npz", allow_pickle=True)
components_top2 = data["components_top2"]
support_mask = data["support_mask"]
scale_ = data["scale_"]
C, F, T = data["tfm_shape"]
freqs = data["freqs"]

L = reconstruct_loadings(components_top2, scale_, support_mask, (C, F, T))
mask = (freqs >= 0) & (freqs <= FMAX)
fmin_disp, fmax_disp = freqs[mask][0], freqs[mask][-1]

fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))

im = None
for ax, pc_idx, label in zip(axes, [0, 1], ["PC1", "PC2"]):
    tf_imp = np.abs(L[pc_idx]).mean(axis=0)
    tf_crop = tf_imp[mask, :]
    im = ax.imshow(tf_crop, origin="lower", aspect="auto",
                    extent=[0, tf_crop.shape[1], fmin_disp, fmax_disp])
    ax.set_title(label, fontsize=12)
    ax.set_xlabel("Time index")
    ax.set_ylim(0, FMAX)

axes[0].set_ylabel("Frequency (Hz)")

cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
fig.colorbar(im, cax=cbar_ax, label="|loading|")

fig.suptitle("Time-Frequency PCA loading", fontsize=13, fontweight="bold")
plt.tight_layout(rect=[0, 0, 0.9, 0.92])
plt.savefig("figure2D_time_freq_loading.png", dpi=300, bbox_inches="tight")
# plt.show()


# ============================================================
# Figure 2E - three-group comparison, significant clusters (CrCb)
# ============================================================
# reads:  figure2efg_source_data.npz
#         group_mean_g0/g1/g2  (F, T) mean log relative power per FPG group
#         anova_stability       (F, T) proportion of bootstrap draws where
#                                each pixel fell in a significant ANOVA cluster
#         f_axis, t_axis         (F,), (T,)
# writes: figure2E_significant_clusters.png

import numpy as np
import matplotlib.pyplot as plt

STAB_THR = 0.20
GROUP_LABELS = {0: "< 100", 1: "100 \u2013 125", 2: "\u2265 126"}

data = np.load("figure2efg_source_data.npz")
f_axis = data["f_axis"]
t_axis = data["t_axis"]
stab = data["anova_stability"]
group_mean = {g: data[f"group_mean_g{g}"] for g in [0, 1, 2]}

mask = stab >= STAB_THR

fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)

im = None
for ax, g in zip(axes, [0, 1, 2]):
    masked = np.where(mask, group_mean[g], np.nan)
    im = ax.imshow(masked, aspect="auto", origin="lower",
                    extent=[t_axis[0], t_axis[-1], f_axis[0], f_axis[-1]], cmap="viridis")
    ax.set_title(f"CrCb | {GROUP_LABELS[g]}")
    ax.set_xlabel("Time (s)")

axes[0].set_ylabel("Frequency (Hz)")

fig.suptitle("Three group comparison: significant clusters", fontsize=14, fontweight="bold")
cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
fig.colorbar(im, cax=cax, label="Mean log (relative power)")

plt.tight_layout(rect=[0, 0, 0.90, 0.93])
plt.savefig("figure2E_significant_clusters.png", dpi=300, bbox_inches="tight")
# plt.show()


# ============================================================
# Figure 2F - delta power between groups (CrCb, vs <100 reference)
# ============================================================
# reads:  figure2efg_source_data.npz (same file and schema as 2E, above)
# writes: figure2F_delta_power.png

import numpy as np
import matplotlib.pyplot as plt

STAB_THR = 0.20

data = np.load("figure2efg_source_data.npz")
f_axis = data["f_axis"]
t_axis = data["t_axis"]
stab = data["anova_stability"]
group_mean = {g: data[f"group_mean_g{g}"] for g in [0, 1, 2]}

mask = stab >= STAB_THR
ref = group_mean[0]

diffs = [(group_mean[g] - ref)[mask] for g in [1, 2]]
vmax = np.nanpercentile(np.abs(np.concatenate(diffs)), 95)
vmin = -vmax

fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)

titles = ["CrCb | (< 100; REF)", "CrCb | (100 \u2013 125) - REF", "CrCb | (\u2265 126) - REF"]
ims = []
for ax, g, title in zip(axes, [0, 1, 2], titles):
    diff = np.zeros_like(ref) if g == 0 else (group_mean[g] - ref)
    masked_diff = np.where(mask, diff, np.nan)
    im = ax.imshow(masked_diff, aspect="auto", origin="lower",
                    extent=[t_axis[0], t_axis[-1], f_axis[0], f_axis[-1]],
                    cmap="RdBu_r", vmin=vmin, vmax=vmax)
    ims.append(im)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")

axes[0].set_ylabel("Frequency (Hz)")

fig.suptitle("\u0394-power between clusters in three groups", fontsize=14, fontweight="bold")

plt.tight_layout()
cbar = fig.colorbar(ims[1], ax=axes, location="right", pad=0.02, shrink=0.9)
cbar.set_label("\u0394 Mean log (relative power)")

plt.savefig("figure2F_delta_power.png", dpi=300, bbox_inches="tight")
# plt.show()


# ============================================================
# Figure 2G - quadratic fit, cluster power vs FPG (CrCb, all subjects)
# ============================================================
# reads:  figure2efg_source_data.npz
# writes: figure2G_quadratic_fit.png


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f, t


def calculate_quadratic_stats_and_ci(x, y):
    n = len(x)
    if n < 4:
        return None

    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]

    coeffs = np.polyfit(x_sorted, y_sorted, 2)
    poly_func = np.poly1d(coeffs)
    y_pred = poly_func(x_sorted)

    ss_res = np.sum((y_sorted - y_pred) ** 2)
    ss_tot = np.sum((y_sorted - np.mean(y_sorted)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    R_val = np.sqrt(r2)

    p_params = 3
    dof_model = p_params - 1
    dof_error = n - p_params
    ms_model = (ss_tot - ss_res) / dof_model
    ms_error = ss_res / dof_error

    if ms_error == 0:
        p_value = np.nan
    else:
        f_stat = ms_model / ms_error
        p_value = f.sf(f_stat, dof_model, dof_error)

    s_err = np.sqrt(ss_res / dof_error)
    t_val = t.ppf(0.975, dof_error)

    X_mat = np.vander(x_sorted, 3)
    try:
        cov_mat = np.linalg.inv(X_mat.T @ X_mat) * s_err ** 2
        var_pred = np.sum((X_mat @ cov_mat) * X_mat, axis=1)
        ci_bound = t_val * np.sqrt(var_pred)
        ci_lower = y_pred - ci_bound
        ci_upper = y_pred + ci_bound
    except np.linalg.LinAlgError:
        ci_lower, ci_upper = y_pred, y_pred

    return {"R": R_val, "p": p_value, "x_fit": x_sorted, "y_fit": y_pred,
            "ci_lower": ci_lower, "ci_upper": ci_upper}


data = np.load("figure2efg_source_data.npz")
x = data["fpg"]
y = data["neg_log_likelihood"]

res = calculate_quadratic_stats_and_ci(x, y)

fig, ax = plt.subplots(figsize=(7, 5.5))

ax.scatter(x, y, alpha=0.6, s=50, edgecolors="white", color="steelblue", zorder=2, label="Subjects")
ax.plot(res["x_fit"], res["y_fit"], color="darkred", linewidth=2.5, label="Quadratic Fit", zorder=3)
ax.fill_between(res["x_fit"], res["ci_lower"], res["ci_upper"], color="red", alpha=0.12, zorder=1, label="95% CI")

ax.legend(loc="lower right", frameon=True, fontsize=9)

p_val = res["p"]
p_str = "<0.001" if p_val < 0.001 else f"={p_val:.3f}"
ax.text(0.97, 0.95, f"R = {res['R']:.3f}\n" + r"$\it{P}$" + f" {p_str}",
        transform=ax.transAxes, ha="right", va="top", fontsize=12, fontweight="bold")

ax.set_title("Quadratic fit: cluster power vs FPG", fontsize=13, fontweight="bold")
ax.set_xlabel("FPG (mg/dL)")
ax.set_ylabel("mean log relative power")
ax.grid(True, linestyle="--", alpha=0.3)

plt.tight_layout()
plt.savefig("figure2G_quadratic_fit.png", dpi=300, bbox_inches="tight")
# plt.show()


# ============================================================
# Figure 3A - average time-frequency maps (CrCb only)
# ============================================================
# reads:  figure3_source_data.npz
#         images    (N, 64, 64, C) TFM per recording, per channel
#         fpg_bin3  (N,) FPG bin label
#         channels  (C,) channel names - same file used by 3B-D below
# writes: figure3A_avg_tfm.png

import numpy as np
import matplotlib.pyplot as plt

TITLE_COLOR = "#152841"
CHANNEL_A = "crcb"
KEEP_FRAC = 0.35

BIN_ORDER = ["<100", "100\u2013125", "\u2265126"]
BIN_TITLES = {"<100": "< 100", "100\u2013125": "100 \u2013 125", "\u2265126": "\u2265 126"}


def zscore_spec(spec, eps=1e-6):
    mu, sd = spec.mean(), spec.std()
    return (spec - mu) / (sd + eps)


data = np.load("figure3_source_data.npz", allow_pickle=True)
images = data["images"]
fpg_bin3 = data["fpg_bin3"]
channels = list(data["channels"])

c_idx = channels.index(CHANNEL_A)

avg_specs = {}
for bin_name in BIN_ORDER:
    idx = np.where(fpg_bin3 == bin_name)[0]
    if len(idx) == 0:
        continue
    avg_specs[bin_name] = images[idx, :, :, c_idx].mean(axis=0)

avg_specs_norm = {b: zscore_spec(s) for b, s in avg_specs.items()}

fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))

for i, bin_name in enumerate(BIN_ORDER):
    ax = axes[i]
    spec = avg_specs_norm[bin_name]
    H = spec.shape[0]
    spec = spec[:int(H * KEEP_FRAC), :]

    ax.imshow(spec, aspect="auto", cmap="magma", interpolation="bicubic")
    ax.set_title(BIN_TITLES[bin_name], fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("Time index")
    if i == 0:
        ax.set_ylabel("Frequency (Hz)")

plt.tight_layout(rect=[0, 0, 1, 0.90])
plt.savefig("figure3A_avg_tfm.png", dpi=300, bbox_inches="tight")
# plt.show()


# ============================================================
# Figure 3B-D - Cohen's d effect-size maps (POS, CrCb, PBV)
# ============================================================
# reads:  figure3_source_data.npz (same file as 3A, above)
# writes: figure3B_cohens_d.png, figure3C_cohens_d.png, figure3D_cohens_d.png
#
# Pairwise Cohen's d between FPG groups:
#   B: >=126 vs <100      (diabetic vs normoglycaemic)
#   C: 100-125 vs <100     (prediabetic vs normoglycaemic)
#   D: >=126 vs 100-125    (diabetic vs prediabetic)
# Red = higher relative TFM power in the higher-glucose group, blue = lower.

import numpy as np
import matplotlib.pyplot as plt

TITLE_COLOR = "#152841"
KEEP_FRAC = 0.25
CLIM = 0.8

CHANNELS_IMG = ["pos", "crcb", "pbv"]
CHANNEL_DISPLAY = {"pos": "POS", "crcb": "CrCb", "pbv": "PBV"}

BIN_TITLE_LABEL = {"<100": "<100", "100\u2013125": "100-125", "\u2265126": "\u2265126"}


def cohens_d_map(X_a, X_b, eps=1e-6):
    """Pixelwise Cohen's d: (mean_b - mean_a) / pooled_std."""
    n_a, n_b = X_a.shape[0], X_b.shape[0]
    mean_a, mean_b = X_a.mean(axis=0), X_b.mean(axis=0)
    std_a, std_b = X_a.std(axis=0, ddof=1), X_b.std(axis=0, ddof=1)
    pooled_std = np.sqrt(((n_a - 1) * std_a**2 + (n_b - 1) * std_b**2) / (n_a + n_b - 2))
    return (mean_b - mean_a) / (pooled_std + eps)


def plot_cohens_d_panel(images, fpg_bin3, channels, bin_a, bin_b, panel_label,
                         out_path, keep_frac=KEEP_FRAC, clim=CLIM, figsize=(11, 3.6)):
    idx_a = np.where(fpg_bin3 == bin_a)[0]
    idx_b = np.where(fpg_bin3 == bin_b)[0]
    print(f"{bin_a}: {len(idx_a)} samples | {bin_b}: {len(idx_b)} samples")

    H, W = images.shape[1], images.shape[2]
    H_keep = int(H * keep_frac)

    fig, axes = plt.subplots(1, len(CHANNELS_IMG), figsize=figsize, sharey=True)

    im = None
    for ax, ch in zip(axes, CHANNELS_IMG):
        c_idx = channels.index(ch)
        X_a = images[idx_a][:, :, :, c_idx]
        X_b = images[idx_b][:, :, :, c_idx]

        dmap = cohens_d_map(X_a, X_b)

        im = ax.imshow(
            dmap[:H_keep, :], cmap="coolwarm", aspect="auto", interpolation="bicubic",
            vmin=-clim, vmax=clim, extent=[0, W, 0, H_keep],
        )

        ax.set_title(CHANNEL_DISPLAY[ch], fontsize=11)
        ax.set_xticks([0, 20, 40, 60])
        ax.set_yticks([])
        ax.set_xlabel("Time index")
        if ax is axes[0]:
            ax.set_ylabel("Relative Frequency\n(TFM bins, cropped)")

    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax, ticks=np.arange(-clim, clim + 0.01, 0.2))
    cbar.set_label("Cohen\u2019s $d$")

    fig.text(0.01, 0.97, f"({panel_label})", fontsize=14, fontweight="bold", ha="left", va="top")
    fig.suptitle(
        f"Effect size maps (Cohen\u2019s $d$; aligned) | "
        f"({BIN_TITLE_LABEL[bin_b]} \u2013 {BIN_TITLE_LABEL[bin_a]})",
        fontsize=12.5, fontweight="bold", color=TITLE_COLOR, y=1.03,
    )

    plt.tight_layout(rect=[0, 0, 0.9, 0.90])
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    # plt.show()


data = np.load("figure3_source_data.npz", allow_pickle=True)
images = data["images"]
fpg_bin3 = data["fpg_bin3"]
channels = list(data["channels"])

plot_cohens_d_panel(images, fpg_bin3, channels, bin_a="<100", bin_b="\u2265126",
                     panel_label="B", out_path="figure3B_cohens_d.png")
plot_cohens_d_panel(images, fpg_bin3, channels, bin_a="<100", bin_b="100\u2013125",
                     panel_label="C", out_path="figure3C_cohens_d.png")
plot_cohens_d_panel(images, fpg_bin3, channels, bin_a="100\u2013125", bin_b="\u2265126",
                     panel_label="D", out_path="figure3D_cohens_d.png")