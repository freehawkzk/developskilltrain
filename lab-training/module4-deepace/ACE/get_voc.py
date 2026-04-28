"""get_voc.py
离线 GET 声码器：将 ACE 电极图转换为声码化音频。
翻译自 MATLAB: GETvoc.m

算法概要
--------
1. 根据频带中心频率计算正弦（GET）或噪声（GEN）载波。
2. 对每个脉冲按高斯包络叠加（最大值模式/求和模式）。
3. 对每个通道做 RMS 归一化（使声学 RMS ≈ 电流级别 RMS）。
4. 对所有通道求和后做去加重滤波，最后归一化峰值到 0.5。
"""
import numpy as np
from scipy.signal import lfilter


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------

def _fft_band_bins_local(num_bands: int) -> np.ndarray:
    """与 GETvoc.m 中内嵌的 FFT_band_bins 一致（仅支持 1–22 带）。"""
    _table = {
        22: [1,1,1,1,1,1,1, 1,1,2,2,2,2,3,3,4,4,5,5,6,7,8],
        21: [1,1,1,1,1,1,1, 1,2,2,2,2,3,3,4,4,5,6,6,7,8],
        20: [1,1,1,1,1,1,1, 1,2,2,2,3,3,4,4,5,6,7,8,8],
        19: [1,1,1,1,1,1,1, 2,2,2,3,3,4,4,5,6,7,8,9],
        18: [1,1,1,1,1,2, 2,2,2,3,3,4,4,5,6,7,8,9],
        17: [1,1,1,2,2, 2,2,2,3,3,4,4,5,6,7,8,9],
        16: [1,1,1,2,2, 2,2,2,3,4,4,5,6,7,9,11],
        15: [1,1,1,2,2, 2,2,3,3,4,5,6,8,9,13],
        14: [1,2,2,2, 2,2,3,3,4,5,6,8,9,13],
        13: [1,2,2,2, 2,3,3,4,5,7,8,10,13],
        12: [1,2,2,2, 2,3,4,5,7,9,11,14],
        11: [1,2,2,2, 3,4,5,7,9,12,15],
        10: [2,2,3, 3,4,5,7,9,12,15],
         9: [2,2,3, 3,5,7,9,13,18],
         8: [2,2,3, 4,6,9,14,22],
         7: [3,4, 4,6,9,14,22],
         6: [3,4, 6,9,15,25],
         5: [3,4, 8,16,31],
         4: [7, 8,16,31],
         3: [7, 15,40],
         2: [7, 55],
         1: [62],
    }
    if num_bands not in _table:
        raise ValueError(f'illegal number of bands: {num_bands}')
    return np.array(_table[num_bands], dtype=int)


def _calculate_weights(num_bands: int, num_bins: int):
    """构建频带权重矩阵（同 GETvoc.m 中 calculate_weights）。"""
    band_bins = _fft_band_bins_local(num_bands)
    w = np.zeros((num_bands, num_bins))
    bin_idx = 2   # 跳过 DC(0) 和 bin 1
    for band in range(num_bands):
        width = int(band_bins[band])
        w[band, bin_idx:bin_idx + width] = 1.0
        bin_idx += width
    return w, band_bins


def _gaussian_envelope(D: float, half_n: int, fs: float) -> np.ndarray:
    """高斯包络：exp(-π t² / D²)，t 范围 [-half_n/fs, half_n/fs]。"""
    t = np.arange(-half_n, half_n + 1) / fs
    return np.exp(-np.pi * t ** 2 / D ** 2)


def _rms(x: np.ndarray) -> float:
    if len(x) == 0:
        return 0.0
    return float(np.sqrt(np.mean(x.astype(float) ** 2)))


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def get_voc(electrodogram: dict,
            map_p: dict,
            vocoder_carrier: int,
            get_durations_factors,
            conv_type: int,
            carrier_freq_shift: float,
            get_fs: int) -> tuple:
    """离线 GET 声码器。

    Parameters
    ----------
    electrodogram       : dict  ACE 电极图，含 electrodes, current_levels, periods。
    map_p               : dict  ACE MAP（经 initialize_ace 处理的单侧）。
    vocoder_carrier     : int   1=正弦载波(GET), 2=噪声载波(GEN)。
    get_durations_factors : array-like  每通道高斯包络时长缩放因子（长度=NumCh）。
    conv_type           : int   1=最大值叠加(fake conv), 2=累加(real conv)。
    carrier_freq_shift  : float 载波频率偏移（Hz）。
    get_fs              : int   输出采样率（Hz）。

    Returns
    -------
    vocoded_sound   : np.ndarray, shape (T,)  声码化音频。
    modulated_bands : np.ndarray, shape (NumCh, T)  各通道调制信号。
    """
    num_ch  = map_p['NumberOfBands']
    num_max = map_p['Nmaxima']

    get_durations_factors = np.asarray(get_durations_factors, dtype=float)

    electrodes     = electrodogram['electrodes']
    current_levels = electrodogram['current_levels']
    ipp_us         = float(electrodogram['periods'])   # 脉冲间隔，微秒

    num_pulses = len(electrodes)
    T_total    = num_pulses * ipp_us / 1e6 + 0.06     # 总时长（s）
    n_samp     = round(T_total * get_fs)
    t          = np.arange(n_samp) / get_fs

    # ---- FFT 频带中心频率 ------------------------------------------------
    ace_fs   = 16000
    fftsize  = 128
    bin_res  = ace_fs / fftsize
    bin_freqs = bin_res * np.arange(fftsize // 2 + 1)

    weights, _ = _calculate_weights(num_ch, fftsize // 2 + 1)

    fc = np.zeros(num_ch)
    D  = np.zeros(num_ch)

    for n in range(num_ch):
        # MATLAB: weights(NumCh-n+1,:), n 从 1 开始 → Python: weights[num_ch-1-n, :]
        useful_bins = weights[num_ch - 1 - n, :] > 0
        fc[n] = np.sum(useful_bins * bin_freqs) / np.sum(useful_bins) + carrier_freq_shift
        D[n]  = get_durations_factors[n] / fc[n]

    # ---- 载波生成 -------------------------------------------------------
    voc_carrier_mat = np.zeros((num_ch, n_samp))

    if vocoder_carrier == 1:   # 正弦载波
        for n in range(num_ch):
            voc_carrier_mat[n, :] = np.sin(2 * np.pi * fc[n] * t) / num_max

    elif vocoder_carrier == 2:  # 噪声载波（多正弦叠加）
        block_size = 128
        for n in range(num_ch):
            useful_bins = weights[num_ch - 1 - n, :] > 0
            freqs_bins = bin_freqs[useful_bins]
            cutoffs = np.array([freqs_bins[0] - ace_fs / block_size / 2,
                                freqs_bins[-1] + ace_fs / block_size / 2]) + carrier_freq_shift
            bandwidth = cutoffs[1] - cutoffs[0]
            M = int(np.ceil(bandwidth * 0.1))
            for _ in range(M):
                freq  = np.random.rand() * bandwidth + cutoffs[0]
                phase = np.random.rand() * 2 * np.pi
                voc_carrier_mat[n, :] += np.sin(2 * np.pi * freq * t + phase)
    else:
        raise ValueError(f'Unsupported vocoder_carrier: {vocoder_carrier}')

    # ---- 声学级别转换（LGF 逆映射）--------------------------------------
    v_norm = current_levels / 255.0
    acoustic_level = ((1.0 + map_p['lgf_alpha']) ** v_norm - 1.0) / map_p['lgf_alpha']

    # ---- 高斯包络叠加 ---------------------------------------------------
    voc_env = np.zeros((num_ch, n_samp))

    for n in range(num_pulses):
        ch = int(electrodes[n])    # 1-indexed 电极号（0=空闲）
        if ch == 0:
            continue
        ch_idx   = ch - 1          # 0-indexed
        half_n   = 3 * round((D[ch_idx] / 2) * get_fs)
        gau_env  = acoustic_level[n] * _gaussian_envelope(D[ch_idx], half_n, get_fs)
        t_center = round((n + 1) * ipp_us / 1e6 * get_fs)   # MATLAB: n*IPP（1-indexed）

        for di in range(-half_n, half_n + 1):
            idx = t_center + di
            if 0 < idx <= n_samp:           # MATLAB: tempIndex+temp > 0（1-indexed）
                py_idx = idx - 1            # 转 0-indexed
                ei = di + half_n            # 包络数组索引
                if conv_type == 1:
                    voc_env[ch_idx, py_idx] = max(voc_env[ch_idx, py_idx], gau_env[ei])
                else:
                    voc_env[ch_idx, py_idx] += gau_env[ei]

    # ---- 调制 -----------------------------------------------------------
    modulated_bands = voc_env * voc_carrier_mat

    # ---- 每通道 RMS 归一化 ----------------------------------------------
    for n in range(num_ch):
        ch_el = n + 1   # 1-indexed 电极号
        rms_mod = _rms(modulated_bands[n, :])
        if rms_mod != 0:
            cl_n = current_levels[electrodes == ch_el]
            rms_cl = _rms(cl_n)
            modulated_bands[n, :] *= rms_cl / rms_mod

    # ---- 求和 -----------------------------------------------------------
    vocoded_sound = np.sum(modulated_bands, axis=0)

    # ---- 去加重滤波（与 ACE 预加重互补）---------------------------------
    de_b = np.array([0.4994, 0.4994])
    de_a = np.array([1.0000, -0.0012])
    vocoded_sound = lfilter(de_b, de_a, vocoded_sound)

    # ---- 峰值归一化到 0.5（与 MATLAB 一致：取正向最大值）------------------
    peak = np.max(vocoded_sound)
    if peak > 0:
        vocoded_sound = vocoded_sound / peak * 0.5

    return vocoded_sound, modulated_bands
