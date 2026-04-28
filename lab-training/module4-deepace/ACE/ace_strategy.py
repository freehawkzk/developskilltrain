"""ace_strategy.py
ACE 策略：将音频转换为电极图（离线整段处理）。
翻译自 MATLAB: ACEStrategy.m
"""
import numpy as np

from ace.common.load_map      import load_map
from ace.common.initialize_ace import initialize_ace
from ace.ace_process           import ace_process


def ace_strategy(x: np.ndarray, fs: int, n_band: int, n_maxima: int):
    """ACE 策略处理（离线）。

    Parameters
    ----------
    x        : np.ndarray  输入音频（fs 采样率）。
    fs       : int         采样率（Hz）。
    n_band   : int         频带数（NumberOfBands）。
    n_maxima : int         每帧选取的最大幅度通道数（Nmaxima）。

    Returns
    -------
    q : dict  电极图序列。
    p : dict  ACE MAP（含所有派生参数）。
    """
    voc_paras = {
        'NumberOfBands':  n_band,
        'Nmaxima':        n_maxima,
        'THR':            10,
        'MCL':            250,
        'BandGain':       0,
        'StimulationRate': 900,
    }
    p_all = load_map(voc_paras)
    p_all = initialize_ace(p_all)

    # 仅处理左侧
    p_all['General']['LeftOn'] = 1
    p_all['Left']['lr_select'] = 'left'

    # 重采样到 16 kHz
    if fs != 16000:
        from scipy.signal import resample_poly
        from math import gcd
        g  = gcd(16000, fs)
        x  = resample_poly(x, 16000 // g, fs // g)
        fs = 16000

    q, _ = ace_process(x, p_all['Left'])
    return q, p_all['Left']
