#features.py

import numpy as np
from scipy.signal import stft
from skimage.transform import resize


def generate_tfm(signal, fs=30, nperseg=64, noverlap=48, output_size=(64, 64), eps = 1e-8):
    """
    Hann window, 75% overlap (nperseg=64, noverlap=48),
    log-power spectrogram S(f,t) = log(|Z(f,t)|^2 + eps).
    """
    _, _, Zxx = stft(signal, fs=fs, window = "hann", nperseg=nperseg, noverlap=noverlap)
    power = np.abs(Zxx) ** 2
    spectrogram = np.log(power + eps)
    resized = resize(spectrogram, output_size, mode="reflect", anti_aliasing=True)
    resized = (resized - np.min(resized)) / np.ptp(resized)
    return resized.astype(np.float32)

def safe_generate_tfm(signal, fs=30, nperseg=64, noverlap=48, output_size=(64, 64), eps=1e-8):
    signal = np.asarray(signal, dtype=float)
    if len(signal) < nperseg or np.isnan(signal).all() or np.std(signal) == 0:
        return np.zeros(output_size, dtype=np.float32)
    try:
        return generate_tfm(signal, fs=fs, nperseg=nperseg, noverlap=noverlap, output_size=output_size, eps=eps)
    except Exception:
        return np.zeros(output_size, dtype=np.float32)




def make_tfm_stack_from_list(row, channels, size=(64, 64)):
    imgs = [safe_generate_tfm(np.asarray(row.get(ch, []), float), output_size=size) for ch in channels]
    return np.stack(imgs, axis=-1)  # (H, W, len(channels))


def make_constant_plane(val, H=64, W=64):
    return np.full((H, W), float(val), dtype=np.float32)


def build_meta_planes_for_rows(df_rows, age_scaler, H=64, W=64):
    """[age_z, gender] planes """
    planes = []
    for _, r in df_rows.iterrows():
        age_z = float(age_scaler.transform([r["age"]])[0]) if "age" in r else 0.0
        age_plane = make_constant_plane(age_z, H, W)

        g = r.get("gender", 0)
        g = (1.0 if g.upper().startswith("M") else 0.0) if isinstance(g, str) else float(g)
        gender_plane = make_constant_plane(g, H, W)

        planes.append(np.stack([age_plane, gender_plane], axis=-1))  # (H,W,2)
    return np.stack(planes, axis=0)  # (N,H,W,2)