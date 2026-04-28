"""initialize_ace.py
ACE 参数初始化（从 MAP 派生所有处理参数）。
翻译自 MATLAB: ACE/CommonFunctions/initialize_ACE.m
"""
import copy
import numpy as np
from scipy.signal import freqz

from .fft_band_bins import fft_band_bins
from .lgf_utils     import lgf_alpha as _lgf_alpha


def initialize_ace(p: dict) -> dict:
    """从 MAP 字典派生所有 ACE 处理参数。

    Parameters
    ----------
    p : dict  包含 'Left'（和可选 'Right'）子字典的 MAP。

    Returns
    -------
    p : dict  原地更新后的 MAP（含 'General' 及各侧的派生参数）。
    """
    fs = 16000
    p  = copy.deepcopy(p)

    # ---- General --------------------------------------------------------
    p['General'] = {
        'block_size':    128,
        'frameDuration': 8,
        'durationSYNC':  6,
        'additionalGap': 1,
        'fft_size':      128,
        'num_bins':      128 // 2 + 1,          # 65
        'bin_freq':      fs / 128,               # 125.0 Hz
        'bin_freqs':     (fs / 128) * np.arange(128 // 2 + 1),
        'sub_mag':       -1e-10,
    }

    for side in ('Left', 'Right'):
        if side not in p:
            continue
        _init_side(p, side, fs)

    return p


def _init_side(p: dict, side: str, fs: int) -> None:
    s = p[side]
    G = p['General']

    s['block_size']    = G['block_size']         # 128
    s['nvects']        = s['pulses_per_frame_per_channel']
    s['analysis_rate'] = s['StimulationRate']
    s['block_shift']   = int(np.ceil(fs / s['analysis_rate']))  # ≈18 @ 900 Hz
    s['analysis_rate'] = fs / s['block_shift']

    s['sub_mag'] = G['sub_mag']

    # ---- Channel order (0-indexed 内部使用) -----------------------------
    nb = s['NumberOfBands']
    if s['ChannelOrderType'] == 'base-to-apex':
        s['channel_order'] = np.arange(nb - 1, -1, -1, dtype=int)  # [nb-1,...,0]
    else:
        s['channel_order'] = np.arange(nb, dtype=int)               # [0,...,nb-1]

    s['ranges']      = s['MCL'] - s['THR']
    s['global_gain'] = 1.0
    s['NHIST']       = s['block_size'] - s['block_shift']

    for key in ('fft_size', 'num_bins', 'bin_freq', 'bin_freqs'):
        s[key] = G[key]

    # ---- Analysis window ------------------------------------------------
    wtype = s['Window']
    if wtype == 'Hanning':
        a = [0.5, 0.5, 0.0, 0.0]
    elif wtype == 'Hamming':
        a = [0.54, 0.46, 0.0, 0.0]
    elif wtype == 'Blackman':
        a = [0.42, 0.5, 0.08, 0.0]
    else:
        a = [0.5, 0.5, 0.0, 0.0]
    n_idx = np.arange(s['block_size'], dtype=float)
    r     = 2.0 * np.pi * n_idx / s['block_size']
    s['window'] = a[0] - a[1]*np.cos(r) + a[2]*np.cos(2*r) - a[3]*np.cos(3*r)

    # ---- FFT filter weights --------------------------------------------
    band_bins_arr  = fft_band_bins(nb)
    s['band_bins'] = band_bins_arr

    num_bins = s['num_bins']
    weights  = np.zeros((nb, num_bins))
    bin_idx  = 2   # MATLAB 从 bin=3(1-indexed) 开始，即跳过 DC(0) 和 bin 1
    for band in range(nb):
        width = int(band_bins_arr[band])
        weights[band, bin_idx:bin_idx + width] = 1.0
        bin_idx += width

    # ---- 频率响应均衡校正 -----------------------------------------------
    _, h = freqz(s['window'] / 2.0, [1.0], worN=s['block_size'])
    power_resp = np.abs(h) ** 2

    P1 = power_resp[0]
    P2 = 2.0 * power_resp[1]
    P3 = power_resp[0] + 2.0 * power_resp[2]

    power_gains = np.empty(nb)
    for band in range(nb):
        w = int(band_bins_arr[band])
        if   w == 1: power_gains[band] = P1
        elif w == 2: power_gains[band] = P2
        else:        power_gains[band] = P3
    s['power_gains'] = power_gains

    # 归一化权重矩阵
    for band in range(nb):
        weights[band, :] /= power_gains[band]
    s['weights'] = weights

    # ---- 频带边界 -------------------------------------------------------
    cum_bins = np.concatenate([[1.5], 1.5 + np.cumsum(band_bins_arr)])
    s['crossover_freqs'] = cum_bins * s['bin_freq']
    s['band_widths']     = np.diff(s['crossover_freqs'])
    s['char_freqs']      = s['crossover_freqs'][:nb] + s['band_widths'] / 2.0

    # ---- 灵敏度与增益 ---------------------------------------------------
    s['sensitivity']  = 2.3
    s['scale_factor'] = 2.3 / 32768.0
    s['gains_dB']     = s['Gain'] + s['BandGains']          # shape (nb,)
    s['gains']        = 10.0 ** (s['gains_dB'] / 20.0)      # shape (nb,)

    # ---- 音量与拒绝通道数 -----------------------------------------------
    s['volume_level'] = s['Volume'] / 10.0
    s['num_rejected'] = nb - s['Nmaxima']

    # ---- 确定刺激模式代码 -----------------------------------------------
    impl_gen = {'CI24RE': 'CIC4', 'CI24R': 'CIC4', 'CI24M': 'CIC3'}.get(
        s.get('ImplantType', 'CI24RE'), 'CIC4')
    mode = s.get('StimulationMode', 'MP1+2')
    if impl_gen == 'CIC4':
        s['StimulationModeCode'] = 28
    else:
        s['StimulationModeCode'] = 30 if mode == 'MP1+2' else 28

    # ---- 对数压缩 alpha -------------------------------------------------
    if s['BaseLevel'] > 0:
        s['lgf_dynamic_range'] = 20.0 * np.log10(s['SaturationLevel'] / s['BaseLevel'])
    s['lgf_alpha'] = _lgf_alpha(s['Q'], s['BaseLevel'], s['SaturationLevel'])
