#fpg_utils.py
import os, random, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from model import TinyCNN
from statsmodels.stats.proportion import proportion_confint

class ImgOnlyDataset(Dataset):
    def __init__(self, imgs, targets):
        self.x = torch.tensor(imgs).permute(0,3,1,2).contiguous().float()  # (N,C,H,W)
        self.y = torch.tensor(targets).float().reshape(-1,1)
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i], self.y[i]


def set_seeds_all(seed: int = 42):
    import random, os
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SimpleScaler:
    def __init__(self, mean=0.0, std=1.0, eps=1e-9):
        self.mean = float(mean)
        self.std = float(std)
        self.eps = float(eps)
 
    def fit(self, x):
        x = np.asarray(x, float)
        self.mean = float(np.nanmean(x))
        self.std = float(np.nanstd(x) + self.eps)
        return self
 
    def transform(self, x):
        x = np.asarray(x, float)
        return (x - self.mean) / self.std
 
    def fit_transform(self, x):
        return self.fit(x).transform(x)
 
 
def _rmse(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0:
        return np.nan
    return float(np.sqrt(np.mean((a - b) ** 2)))
 
 
def _mae(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0:
        return np.nan
    return float(np.mean(np.abs(a - b)))
 
 
def _mard_percent(a, b, eps=1e-8):
    """Mean Absolute Relative Difference (%) = mean(|yhat-y|/|y|)*100"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0:
        return np.nan
    denom = np.maximum(np.abs(a), eps)
    return float(np.mean(np.abs(b - a) / denom) * 100.0)
 
 
def compute_paper_metrics(y_true, y_pred):
    """Returns dict with: rmse_all, rmse_lt100, rmse_100_125, pctdiff_ge126, mard, mae."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
 
    m_lt100 = y_true < 100
    m_100_125 = (y_true >= 100) & (y_true < 126)
    m_ge126 = y_true >= 126
 
    out = {}
    out["rmse_all"] = _rmse(y_true, y_pred)
    out["rmse_lt100"] = _rmse(y_true[m_lt100], y_pred[m_lt100])
    out["rmse_100_125"] = _rmse(y_true[m_100_125], y_pred[m_100_125])
    out["pctdiff_ge126"] = _mard_percent(y_true[m_ge126], y_pred[m_ge126]) if np.any(m_ge126) else np.nan
    out["mard"] = _mard_percent(y_true, y_pred)
    out["mae"] = _mae(y_true, y_pred)
    return out
 
 
def _wilson_lb(k: int, n: int, alpha: float = 0.05, method: str = "wilson"):
    """One-sided (1-alpha) lower confidence bound (FDA iCGM convention)."""
    if n <= 0:
        return np.nan
    lb, _ = proportion_confint(count=k, nobs=n, alpha=2 * alpha, method=method)
    return float(lb)
 
 
def plot_yhat_vs_ytrue(y_true, y_pred, title="ŷ vs y (Predicted vs Actual)", save_path=None):
    y_true = np.asarray(y_true, float).ravel()
    y_pred = np.asarray(y_pred, float).ravel()
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[m], y_pred[m]
    if yt.size == 0:
        raise ValueError("No valid pairs to plot.")
 
    rmse = np.sqrt(mean_squared_error(yt, yp))
    mae = mean_absolute_error(yt, yp)
    r2 = r2_score(yt, yp)
    mard = float(np.mean(np.abs(yt - yp) / np.clip(yt, 1e-6, None)) * 100.0)
 
    lo = float(min(yt.min(), yp.min()))
    hi = float(max(yt.max(), yp.max()))
    margin = 0.05 * (hi - lo if hi > lo else 1.0)
    lo -= margin
    hi += margin
 
    plt.figure(figsize=(6, 6))
    plt.scatter(yt, yp, s=12, alpha=0.6)
    plt.plot([lo, hi], [lo, hi])
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.xlabel("Actual (y)")
    plt.ylabel("Predicted (ŷ)")
    plt.title(title)
    txt = f"RMSE: {rmse:.2f}\nMAE: {mae:.2f}\nR²: {r2:.3f}\nMARD: {mard:.2f}%"
    plt.gcf().text(0.98, 0.02, txt, ha="right", va="bottom")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()