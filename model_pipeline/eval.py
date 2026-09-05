#eval.py
import numpy as np
import pandas as pd
from fpg_utils import _wilson_lb


def clarke_error_grid(y_true, y_pred):
    """Source : Clarke Error Grid (Clarke et al., 1987)"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    zones = []
    for ref, pred in zip(y_true, y_pred):
        if (ref <= 70 and pred <= 70) or (0.8 * ref <= pred <= 1.2 * ref):
            zones.append("A")
        elif (ref >= 180 and pred <= 70) or (ref <= 70 and pred >= 180):
            zones.append("E")
        elif (70 <= ref <= 290 and pred >= ref + 110) or \
             (130 <= ref <= 180 and pred <= (7 / 5) * ref - 182):
            zones.append("C")
        elif (ref >= 240 and 70 <= pred <= 180) or \
             (ref <= 175 / 3 and 70 <= pred <= 180) or \
             (175 / 3 <= ref <= 70 and pred >= (6 / 5) * ref):
            zones.append("D")
        else:
            zones.append("B")

    counts = pd.Series(zones).value_counts(normalize=True) * 100
    return {z: counts.get(z, 0.0) for z in ["A", "B", "C", "D", "E"]} | {"Total samples": len(zones)}

def compute_icgm_style_metrics_v2(details_df, model_tag, pred_col="y_pred_oof_mean", alpha=0.05, ci_method="wilson"):
    df = details_df.loc[details_df["model"] == model_tag].copy()
    if df.empty:
        raise ValueError(f"No rows found for model tag: {model_tag}")

    y_true = df["y_true"].astype(float).to_numpy()
    y_pred = df[pred_col].astype(float).to_numpy()
    abs_err = np.abs(y_pred - y_true)
    rel_err = abs_err / np.clip(y_true, 1e-6, None)

    specs = [
        dict(glucose_range="70–180 mg/dL", mask=(y_true >= 70) & (y_true <= 180), tolerance="±10%", pass_mask=(rel_err <= 0.10), FDA_threshold=None),
        dict(glucose_range=">180 mg/dL", mask=(y_true > 180), tolerance="±10%", pass_mask=(rel_err <= 0.10), FDA_threshold=None),
        dict(glucose_range="<70 mg/dL", mask=(y_true < 70), tolerance="±15 mg/dL", pass_mask=(abs_err <= 15), FDA_threshold=0.85),
        dict(glucose_range="70–180 mg/dL", mask=(y_true >= 70) & (y_true <= 180), tolerance="±15%", pass_mask=(rel_err <= 0.15), FDA_threshold=0.70),
        dict(glucose_range=">180 mg/dL", mask=(y_true > 180), tolerance="±15%", pass_mask=(rel_err <= 0.15), FDA_threshold=0.80),
        dict(glucose_range="<70 mg/dL", mask=(y_true < 70), tolerance="±40 mg/dL", pass_mask=(abs_err <= 40), FDA_threshold=0.98),
        dict(glucose_range="70–180 mg/dL", mask=(y_true >= 70) & (y_true <= 180), tolerance="±40%", pass_mask=(rel_err <= 0.40), FDA_threshold=0.99),
        dict(glucose_range=">180 mg/dL", mask=(y_true > 180), tolerance="±40%", pass_mask=(rel_err <= 0.40), FDA_threshold=0.99),
    ]

    rows = []
    for s in specs:
        idx = s["mask"]
        n = int(np.sum(idx))
        k = int(np.sum(idx & s["pass_mask"]))
        pct = (k / n) if n > 0 else np.nan
        lb = _wilson_lb(k, n, alpha=alpha, method=ci_method) if n > 0 else np.nan
        thr = s["FDA_threshold"]
        rows.append({
            "glucose_range": s["glucose_range"], "tolerance": s["tolerance"], "n": n,
            "within_tolerance_%": (pct * 100) if n > 0 else np.nan,
            "lower_95%_bound_%": (lb * 100) if n > 0 else np.nan,
            "FDA_threshold_%": (thr * 100) if thr is not None else None,
            "pass": (lb >= thr) if (n > 0 and thr is not None) else None,
        })
    return pd.DataFrame(rows)


def compute_iso_style_metrics(details_df, model_tag, pred_col="y_pred_oof_mean", ref_col="y_true",
                                alpha=0.05, ci_method="wilson", twenty_break=100.0):
    df = details_df.loc[details_df["model"] == model_tag].copy()
    if df.empty:
        raise ValueError(f"No rows found for model tag: {model_tag}")

    y_true = df[ref_col].astype(float).to_numpy()
    y_pred = df[pred_col].astype(float).to_numpy()
    abs_err = np.abs(y_pred - y_true)
    rel_err_pct = abs_err / np.clip(y_true, 1e-6, None) * 100.0

    specs = [
        dict(rule="15/15", glucose_range="<100 mg/dL", mask=(y_true < 100), tolerance="±15 mg/dL", within=(abs_err <= 15)),
        dict(rule="15/15", glucose_range="≥100 mg/dL", mask=(y_true >= 100), tolerance="±15%", within=(rel_err_pct <= 15)),
        dict(rule=f"20/20 (break={int(twenty_break)})", glucose_range=f"<{int(twenty_break)} mg/dL", mask=(y_true < twenty_break), tolerance="±20 mg/dL", within=(abs_err <= 20)),
        dict(rule=f"20/20 (break={int(twenty_break)})", glucose_range=f"≥{int(twenty_break)} mg/dL", mask=(y_true >= twenty_break), tolerance="±20%", within=(rel_err_pct <= 20)),
    ]

    rows = []
    for s in specs:
        mask = s["mask"]
        n = int(np.sum(mask))
        if n == 0:
            rows.append({"rule": s["rule"], "glucose_range": s["glucose_range"], "tolerance": s["tolerance"], "n": 0, "within_tolerance_%": np.nan, "lower_95%_bound_%": np.nan})
            continue
        k = int(np.sum(s["within"][mask]))
        pct = 100.0 * k / n
        lb = _wilson_lb(k, n, alpha=alpha, method=ci_method)
        rows.append({"rule": s["rule"], "glucose_range": s["glucose_range"], "tolerance": s["tolerance"], "n": n, "within_tolerance_%": pct, "lower_95%_bound_%": 100.0 * lb})
    return pd.DataFrame(rows)