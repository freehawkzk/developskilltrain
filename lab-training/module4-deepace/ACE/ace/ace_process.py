"""ace_process.py
离线 ACE 前端处理（完整音频 → 电极图）。
翻译自 MATLAB: ACE/ACE_Process.m

关键流程
--------
1. 前端缩放 + 预加重滤波
2. 分帧（含重叠）+ 加汉宁窗 + FFT
3. 频带功率包络提取（加权求和）
4. 增益
5. 拒绝最小幅度通道
6. 对数压缩（LGF）
7. 整理为通道-幅值序列（按 channel_order 排列）
8. 通道映射 → 电极序列 + 电流级别
"""
import numpy as np
from scipy.signal import lfilter

from .common.logarithmic_compression import logarithmic_compression


def _matlab_buffer(x: np.ndarray, N: int, overlap: int) -> np.ndarray:
    """模拟 MATLAB buffer(x, N, overlap, []) ：零填充分帧。

    Returns
    -------
    frames : np.ndarray, shape (N, n_frames)
    """
    step = N - overlap
    x_len = len(x)
    if x_len == 0:
        return np.zeros((N, 0))
    n_frames = max(1, int(np.ceil((x_len - overlap) / step)))
    frames = np.zeros((N, n_frames))
    for i in range(n_frames):
        start = i * step
        end   = start + N
        chunk = x[start:min(end, x_len)]
        frames[:len(chunk), i] = chunk
    return frames


def ace_process(x: np.ndarray, p: dict):
    """离线 ACE 处理：将音频序列转换为电极图。

    Parameters
    ----------
    x : np.ndarray, shape (N,)  输入音频（16 kHz，已归一化）。
    p : dict  经 initialize_ace 处理的单侧 MAP（如 p_all['Left']）。

    Returns
    -------
    q : dict  电极图序列，字段：
              electrodes, modes, current_levels,
              phase_widths, phase_gaps, periods
    p : dict  （原样返回）
    """
    x = np.asarray(x, dtype=float).ravel()

    # ---- 前端缩放 + 预加重 -------------------------------------------
    front_end_scaling = 1.0590e3
    input_scaling     = 5.5325e-4
    pre_b = np.array([0.5006, -0.5006])
    pre_a = np.array([1.0000, -0.0012])

    y = x * front_end_scaling
    z = lfilter(pre_b, pre_a, y)
    x_scaled = z * input_scaling

    # ---- 分帧（含重叠） -----------------------------------------------
    block_size  = p['block_size']   # 128
    block_shift = p['block_shift']  # ≈18
    overlap     = block_size - block_shift
    frames = _matlab_buffer(x_scaled, block_size, overlap)
    if frames.shape[1] == 0:
        # 音频太短，返回空序列
        q = dict(electrodes=np.array([]), modes=p['StimulationModeCode'],
                 current_levels=np.array([]), phase_widths=p['PulseWidth'],
                 phase_gaps=p['IPG'],
                 periods=1.0/p['StimulationRate']/p['Nmaxima']*1e6)
        return q, p

    num_time_slots = frames.shape[1]

    # ---- 加窗 + FFT ---------------------------------------------------
    v = frames * p['window'][:, np.newaxis]
    U = np.fft.fft(v, axis=0)
    U = U[:p['num_bins'], :]       # 保留 0..num_bins-1（含 DC 到 Nyquist）

    # ---- 功率包络 -------------------------------------------------------
    V = np.abs(U) ** 2                          # (num_bins, T)
    E = p['weights'] @ V                        # (num_bands, T)
    E = np.sqrt(E)

    # ---- 增益 -----------------------------------------------------------
    E = E * p['gains'][:, np.newaxis]

    # ---- 拒绝最小幅度通道（每帧拒绝 num_rejected 个通道）--------------
    num_bands      = p['NumberOfBands']
    num_rejected   = p['num_rejected']
    if num_rejected > 0:
        for _ in range(num_rejected):
            # 将 NaN 暂时替换为 inf，以便 argmin 跳过已拒绝的通道
            E_masked = np.where(np.isnan(E), np.inf, E)
            k = np.argmin(E_masked, axis=0)          # (T,) row indices of min per column
            E[k, np.arange(num_time_slots)] = np.nan

    # ---- 对数压缩（LGF） -----------------------------------------------
    E, _, _ = logarithmic_compression(p, E)

    # ---- 整理序列（按 channel_order 排列，列主序展平）------------------
    # channel_order 已是 0-indexed，按 'base-to-apex' = [nb-1, nb-2, ..., 0]
    ch_order = p['channel_order']                          # (nb,) 0-indexed
    reord    = E[ch_order, :]                              # (nb, T)
    magnitudes = reord.ravel(order='F')                   # (nb*T,) 列主序展平
    # 对应通道号（1-indexed，与 MATLAB 一致）
    channels   = np.tile(ch_order + 1, num_time_slots)   # (nb*T,) 1-indexed

    # 移除 NaN（已被拒绝的通道）
    keep = ~np.isnan(magnitudes)
    channels   = channels[keep]
    magnitudes = magnitudes[keep]

    # ---- 通道映射 → 电极序列 ------------------------------------------
    # electrodes[i] = Electrodes[i-1] for i=1..nb, electrodes[nb+1]=0（空闲）
    electrodes_map    = np.append(p['Electrodes'], 0.0)   # (nb+1,)
    threshold_levels  = np.append(p['THR'], 0.0)
    comfort_levels    = np.append(p['MCL'], 0.0)

    # 空闲脉冲（channel == 0）→ 映射到最后一个元素（电极0）
    idle      = (channels == 0)
    ch_mapped = channels.copy().astype(int)
    ch_mapped[idle] = len(electrodes_map)                 # = nb+1 (1-indexed last)

    # 1-indexed 查表（Python 0-indexed：减 1）
    q_electrodes    = electrodes_map[ch_mapped - 1]
    q_t             = threshold_levels[ch_mapped - 1]
    ranges          = comfort_levels - threshold_levels
    q_r             = ranges[ch_mapped - 1]
    q_magnitudes    = np.minimum(magnitudes, 1.0)
    q_current_levels = np.round(q_t + q_r * p['volume_level'] * q_magnitudes)

    # 空闲脉冲电流级别清零，电极号清零
    q_is_idle = (q_magnitudes < 0)
    q_current_levels[q_is_idle] = 0.0
    q_electrodes[q_is_idle]     = 0.0

    q = {
        'electrodes':    q_electrodes,
        'modes':         p['StimulationModeCode'],
        'current_levels': q_current_levels,
        'phase_widths':  p['PulseWidth'],
        'phase_gaps':    p['IPG'],
        'periods':       1.0 / p['StimulationRate'] / p['Nmaxima'] * 1e6,
    }
    return q, p
