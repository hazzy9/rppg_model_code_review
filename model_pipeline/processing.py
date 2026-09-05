#processing.py

"""
RGB channel processing: extracts the 7 candidate rPPG signal types from
raw R, G, B channel traces. Matches the manuscript's "RGB Channel
Extraction" and "Remote PPG Signal Extraction" sections.
"""


import numpy as np
import pandas as pd
import json

from signals import (
    extract_crcb_value, 
    extract_pos_ppg, 
    extract_pbv,
    extract_cr_plus_cg_value,
    extract_cr_div_cg_value,
    extract_ycgco_value_ycgco,
)



# Extracts the 7 types of signals from raw R,G,B signals
def process_all_signals(sigR, sigG, sigB, fps):
    crcb_signal = np.array([extract_crcb_value(r, g, b) for r, g, b in zip(sigR, sigG, sigB)])
    green_signal = np.array(sigG)
    crcg_signal = np.array([extract_cr_plus_cg_value(r,g,b) for r,g,b in zip(sigR, sigG, sigB)])
    cg_div_cr_signal = np.array([extract_cr_div_cg_value(r,g,b) for r,g,b in zip(sigR, sigG, sigB)])
    ycgco_signal = np.array([extract_ycgco_value_ycgco(r,g,b) for r,g,b in zip(sigR, sigG, sigB)])

    pos_signal = extract_pos_ppg(sigR, sigG, sigB, win_len=48)
    pbv_signal = extract_pbv(sigR, sigG, sigB, win_len=512, overlap=0.5)

    signals_df = pd.DataFrame({
        'crcb': crcb_signal,
        'green': green_signal,
        'pos': pos_signal,
        'pbv': pbv_signal,
        'crcg' : crcg_signal,
        'cr/cg' : cg_div_cr_signal,
        'ycgco' : ycgco_signal
    })
    
    return signals_df
