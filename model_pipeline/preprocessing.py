#preprocessing.py

import numpy as np
import pandas as pd
import re
import warnings
from scipy.signal import butter, filtfilt, detrend,find_peaks
from dateutil import parser


# parameters
MAX_STEP_SEC = 0.40
MIN_SPAN_SEC = 40.0
SIGNAL_COLS = ["crcb","green","pos","pbv","crcg","cr/cg","ycgco"]


def clean_date_string(date_str):
    if pd.isna(date_str):
        return None
    date_str = str(date_str).strip()
    date_str = re.sub(r'[\s:/\\\-]+', '.', date_str)
    # Remove double dots
    date_str = re.sub(r'\.{2,}', '.', date_str)

    try:
        dt = parser.parse(date_str, dayfirst=False, yearfirst=True)
        return dt.date() 
    except Exception:
        return None




def _first_continuous_segment(t, x, min_span_sec=MIN_SPAN_SEC, max_step_sec=MAX_STEP_SEC):
    """
    Return (i0, i1, span_sec) in ORIGINAL indices for the first continuous finite segment
    with span >= min_span_sec. 'Continuous' = adjacent finite samples with delta-t <= max_step_sec.
    If none, returns (-1, -1, 0.0).
    """
    t = np.asarray(t, float)
    x = np.asarray(x, float)
    if t.size < 2 or t.size != x.size:
        return -1, -1, 0.0

    idx_fin = np.flatnonzero(np.isfinite(x))
    if idx_fin.size < 2:
        return -1, -1, 0.0

    tf = t[idx_fin]
    dt = np.diff(tf)
    
    brk = np.concatenate(([True], dt > max_step_sec, [True]))
    bidx = np.flatnonzero(brk)

    for s, e in zip(bidx[:-1], bidx[1:]):
        if e - s >= 2:
            i0 = int(idx_fin[s])
            i1 = int(idx_fin[e-1])
            span = float(t[i1] - t[i0])
            if span >= min_span_sec:
                return i0, i1, span
    return -1, -1, 0.0

def _row_has_nonfinite(x):
    x = np.asarray(x, float)
    return (~np.isfinite(x)).any()

def _crop_and_rebase_row(row, i0, i1):

    out = {}
    # time
    t = np.asarray(row["time_cumulative"], float)
    t_seg = t[i0:i1+1].copy()
    t_seg -= t_seg[0]
    out["time_cumulative"] = t_seg

    for col in SIGNAL_COLS:
        xx = np.asarray(row[col], float)
        out[col] = xx[i0:i1+1].copy()
    return out

def filter_and_crop_tierA(merged_df,
                          min_span_sec=MIN_SPAN_SEC,
                          max_step_sec=MAX_STEP_SEC):

    keep_rows = []
    cropped_payload = []  
    kept_uncropped = kept_cropped = dropped = 0
    spans = []

    for i, row in merged_df.iterrows():
        cg = row["crcg"]
        t  = row["time_cumulative"]

        if not _row_has_nonfinite(cg):
            keep_rows.append(True)
            cropped_payload.append(None)
            kept_uncropped += 1
            continue

        i0, i1, span = _first_continuous_segment(t, cg,
                                                 min_span_sec=min_span_sec,
                                                 max_step_sec=max_step_sec)
        if i0 >= 0:
            keep_rows.append(True)
            payload = _crop_and_rebase_row(row, i0, i1)
            cropped_payload.append(payload)
            kept_cropped += 1
            spans.append(span)
        else:
            keep_rows.append(False)
            cropped_payload.append(None)
            dropped += 1

    keep_mask = np.array(keep_rows, dtype=bool)
    df_out = merged_df.loc[keep_mask].copy().reset_index(drop=True)

    j = 0  
    for k, kept in enumerate(keep_rows):
        if not kept:
            continue
        payload = cropped_payload[k]
        if payload is not None:
            df_out.at[j, "time_cumulative"] = payload["time_cumulative"]
            for col in SIGNAL_COLS:
                df_out.at[j, col] = payload[col]
        j += 1

    stats = {
        "kept_total": int(df_out.shape[0]),
        "kept_uncropped": int(kept_uncropped),
        "kept_cropped": int(kept_cropped),
        "dropped": int(dropped),
        "cropped_span_sec_min": float(np.min(spans)) if spans else None,
        "cropped_span_sec_median": float(np.median(spans)) if spans else None,
        "cropped_span_sec_max": float(np.max(spans)) if spans else None,
        "params": {"min_span_sec": float(min_span_sec), "max_step_sec": float(max_step_sec)}
    }
    return df_out, stats


def trim_each_end(df, trim_sec=1.5):
    """
    Cuts front and end of the signal by trim_sec
    """
    kept_rows = []
    dropped_rows = 0
    before_spans = []
    after_spans  = []

    for _, row in df.iterrows():
        t = np.asarray(row["time_cumulative"], float)
        if t.size == 0:
            dropped_rows += 1
            continue

        tmin, tmax = float(t[0]), float(t[-1])
        # guard: if span <= 2*trim_sec, trimming would empty it
        if (tmax - tmin) <= 2*trim_sec:
            dropped_rows += 1
            continue

        # build mask in absolute time of this row
        keep_mask = (t >= (tmin + trim_sec)) & (t <= (tmax - trim_sec))
        if not np.any(keep_mask):
            dropped_rows += 1
            continue

        # slice and rebase time
        t_new = t[keep_mask].copy()
        t_new -= t_new[0]

        new_row = row.copy()
        new_row["time_cumulative"] = t_new

        # slice all channels identically
        for col in SIGNAL_COLS:
            x = np.asarray(row[col], float)
            new_row[col] = x[keep_mask].copy()

        kept_rows.append(new_row)

        before_spans.append(tmax - tmin)
        after_spans.append(float(t_new[-1] - t_new[0]))

    df_out = pd.DataFrame(kept_rows).reset_index(drop=True)
    stats = {
        "rows_in":       int(df.shape[0]),
        "rows_kept":     int(df_out.shape[0]),
        "rows_dropped":  int(dropped_rows),
        "span_before_min": float(np.min(before_spans)) if before_spans else None,
        "span_before_med": float(np.median(before_spans)) if before_spans else None,
        "span_before_max": float(np.max(before_spans)) if before_spans else None,
        "span_after_min": float(np.min(after_spans)) if after_spans else None,
        "span_after_med": float(np.median(after_spans)) if after_spans else None,
        "span_after_max": float(np.max(after_spans)) if after_spans else None,
        "trim_sec_each_end": float(trim_sec),
    }
    return df_out, stats


def integrity_check(signal, time, atol=1e-9):
    """
    Checks data integrity:
      - time starts at 0
      - time is strictly monotonically increasing
      - no non-finite values remain in signal
    Returns (passed: bool, reason: str or None)
    """
    signal = np.asarray(signal)
    time = np.asarray(time)
 
    if len(time) == 0 or len(signal) == 0:
        return False, "empty signal/time"
 
    if not np.isclose(time[0], 0.0, atol=atol):
        return False, f"time does not start at 0 (starts at {time[0]})"
 
    if not np.all(np.diff(time) > 0):
        return False, "time is not strictly monotonically increasing"
 
    if not np.isfinite(signal).all():
        return False, "signal contains non-finite values"
 
    return True, None

# bandpass
def apply_bandpass(signal, fps, min_bpm=24, max_bpm=240):
    """Passband: 24-240 bpm = 0.4-4.0 Hz"""
    
    nyquist = 0.5 * fps
    low = min_bpm / 60.0 / nyquist
    high = max_bpm / 60.0 / nyquist


    if not (0 < low < 1) or not (0 < high < 1):
        return None
    b, a = butter(3, [low, high], btype='band')
    return filtfilt(b, a, signal)


# detrend
def _safe_detrend(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)

    if not np.isfinite(x).all():
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    if x.size < 4 or np.ptp(x) < 1e-12:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"scipy\.signal\._signaltools")
            return detrend(x, type="constant")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"scipy\.signal\._signaltools")
        return detrend(x, type="linear")

# preprocessing
def apply_temporal_filtering(df, fps,channels=SIGNAL_COLS):
    """
    * Applies: detrend + bandpass, per channel, per recording.
    Runs AFTER tierA crop (Step 1) + edge trim (Step 2), 
    BEFORE the integrity check (Step 4). 
    Passband: 24-240 bpm = 0.4-4.0 Hz
    
    """
    from preprocessing import _safe_detrend
    df = df.copy()
    fail_mask = pd.Series(False, index=df.index)

    def _filt(sig):
        x = np.asarray(sig, dtype=float)
        detrended = _safe_detrend(x)
        if detrended is None:
            return None
        return apply_bandpass(detrended, fps)

    for ch in channels:
        filtered = df[ch].apply(_filt)
        fail_mask |= filtered.apply(lambda x: x is None)
        df[ch] = filtered

    stats = {"n_input": len(df), "n_filter_failed": int(fail_mask.sum())}
    df = df.loc[~fail_mask].reset_index(drop=True)
    stats["n_output"] = len(df)
    return df, stats
    
def integrity_filter(df, channels=SIGNAL_COLS, time_col="time_cumulative"):
    """ 
    *Per-channel integrity check 
    (t0=0, strictly monotonic time, all-finite signal). 
    A recording is kept only if every channel passes.
    """

    keep = pd.Series(True, index=df.index)
    fail_log = []

    for idx, row in df.iterrows():
        time = row[time_col]
        for ch in channels:
            ok, reason = integrity_check(row[ch], time)
            if not ok:
                keep.at[idx] = False
                fail_log.append((row["Patient_ID"], ch, reason))
                break

    stats = {"n_input": len(df), "n_dropped": int((~keep).sum()), "n_output": int(keep.sum())}
    return df.loc[keep].reset_index(drop=True), fail_log, stats