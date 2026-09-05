from scipy.signal import welch
import numpy as np

#helper function
def sliding_windows(x, win_size, step):
    return [
        x[i:i + win_size]
        for i in range(0, len(x) - win_size + 1, step)
    ]

#welch psd
def compute_psd(x, fs):
    freqs, psd = welch(
        x,
        fs=fs,
        nperseg=int(fs * 10),      # 10s segments
        noverlap=int(fs * 5),     # 50% overlap
        detrend="constant"
    )
    return freqs, psd

#domiant freq detection (f0)
def find_f0(freqs, psd, fmin=0.67, fmax=4.0):
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return None
    return freqs[mask][np.argmax(psd[mask])]


#sqi computation

def compute_sqi(freqs, psd, f0, bw=0.1):
    if f0 is None:
        return np.nan

    hr_band = (freqs >= 0.67) & (freqs <= 4.0)
    sig_band = (freqs >= f0 - bw) & (freqs <= f0 + bw)

    signal_power = np.sum(psd[sig_band])
    noise_power = np.sum(psd[hr_band]) - signal_power

    if noise_power <= 0:
        return np.nan

    return signal_power / noise_power

