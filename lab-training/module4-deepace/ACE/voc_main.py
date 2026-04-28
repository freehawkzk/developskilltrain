"""voc_main.py
离线 GET/GEN 声码器演示（单文件）。
翻译自 MATLAB: VocMain.m

用法
----
    python voc_main.py
"""
import os, sys
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import resample_poly, welch
from math import gcd

# 将 scripts_py 加入路径
sys.path.insert(0, os.path.dirname(__file__))

from ace_strategy import ace_strategy
from get_voc      import get_voc


# ============================================================
# 参数（与 VocMain.m 保持一致）
# ============================================================
N_BAND_VOCODER       = 22        # 通道数
N_MAXIMA             = 8         # 每帧选最大幅度通道数
VOCODER_CARRIER      = 1         # 1=GET(正弦), 2=GEN(噪声)
CONV_TYPE            = 1         # 1=最大值叠加, 2=累加
CARRIER_FREQ_SHIFT   = 0         # 载波频率偏移（Hz）

GET_DURATIONS_FACTORS = (3 + (N_BAND_VOCODER - np.arange(1, N_BAND_VOCODER + 1))).astype(float)

GET_FS = 16000 if CARRIER_FREQ_SHIFT == 0 else 48000

# 输入文件（相对于本脚本上一级目录）
_BASE = os.path.join(os.path.dirname(__file__), '..')
SOUND_PATH = os.path.normpath(os.path.join(_BASE, './data/audio', 'T006_CB8_02553x3.wav'))


def _rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype=float) ** 2)))


def _myspectrogram(ax, s, fs, title=''):
    """频谱图（与 myspectrogram.m / toframes 完全一致）。"""
    Tw, Ts = 18, 1                                  # 帧长/帧移 (ms)
    nfft = 1024                                      # FFT 点数，与 MATLAB 一致
    Nw = round(fs * Tw * 0.001)                      # 帧长（样本）
    Ns = round(fs * Ts * 0.001)                      # 帧移（样本）

    # ---- 与 MATLAB toframes 一致的信号归一化 ----
    sig = np.array(s, dtype=float)
    smax = np.max(np.abs(sig)) / 0.999
    if smax > 0:
        sig = sig / smax

    # ---- 零填充（与 MATLAB toframes 一致）----
    D = len(sig) % Ns
    G = (int(np.ceil(Nw / Ns)) - 1) * Ns
    sig = np.concatenate([np.zeros(G), sig, np.zeros(Nw - D)])
    M = (len(sig) - Nw) // Ns + 1

    # ---- Hamming 窗 ----
    win = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(Nw) / (Nw - 1))

    # ---- 分帧 + 加窗 + FFT ----
    frames = np.zeros((M, nfft))
    for i in range(M):
        frames[i, :Nw] = sig[i * Ns: i * Ns + Nw] * win
    S = np.fft.fft(frames, n=nfft, axis=1)
    S = np.abs(S) ** 2 / Nw       # periodogram
    S = np.sqrt(S)                 # 幅度谱
    S = S[:, :nfft // 2 + 1].T    # 只保留正频率，转为 (freq, time)

    # ---- 全局归一化 + dB ----
    S = S / np.max(S)
    S[S < 1e-10] = 1e-10
    S = 20 * np.log10(S)

    F = np.arange(nfft // 2 + 1) * fs / nfft
    T = np.arange(M) * Ns / fs

    ax.pcolormesh(T, F, S, vmin=-59, vmax=-1, cmap='jet', shading='auto')
    ax.set_ylim(0, fs / 2)
    ax.set_xlabel('time (s)')
    ax.set_ylabel('frequency (Hz)')
    if title:
        ax.set_title(title)


def main():
    # ---- 读取音频 -------------------------------------------------------
    if not os.path.exists(SOUND_PATH):
        print(f'音频文件不存在: {SOUND_PATH}')
        print('请修改 SOUND_PATH 或将音频文件放到 audio/ 目录。')
        return

    x, fs = sf.read(SOUND_PATH, always_2d=False)
    if x.ndim > 1:
        x = x[:, 0]

    # 重采样到 16 kHz
    if fs != 16000:
        g = gcd(16000, fs)
        x = resample_poly(x, 16000 // g, fs // g)
        fs = 16000

    # 归一化到 0.36
    x = x / np.max(np.abs(x)) * 0.36
    t = np.arange(len(x)) / fs

    # ---- ACE 处理 --------------------------------------------------------
    print('正在运行 ACE 策略...')
    q, p = ace_strategy(x, fs, N_BAND_VOCODER, N_MAXIMA)

    # ---- GET 声码器 ------------------------------------------------------
    print('正在运行 GET 声码器...')
    vocoded_sound, _ = get_voc(q, p, VOCODER_CARRIER, GET_DURATIONS_FACTORS,
                                CONV_TYPE, CARRIER_FREQ_SHIFT, GET_FS)

    o = vocoded_sound.ravel()

    # 输出 RMS 匹配到原始信号
    rms_x = _rms(x)
    rms_o = _rms(o)
    if rms_o > 0:
        o = o * rms_x / rms_o
    t1 = np.arange(len(o)) / GET_FS

    # ---- 保存 WAV --------------------------------------------------------
    out_path = os.path.normpath(os.path.join(_BASE, 'compare_offline_py.wav'))
    sf.write(out_path, o.astype(np.float32), GET_FS)
    print(f'已保存声码化音频: {out_path}')

    # ---- 绘图 -----------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('GET Vocoder — Offline')

    axes[0, 0].plot(t, x)
    axes[0, 0].set_xlim(0, t[-1]); axes[0, 0].set_ylim(-0.7, 0.7)
    axes[0, 0].set_title('Original audio'); axes[0, 0].set_xlabel('time (s)')

    axes[0, 1].plot(t1, o)
    axes[0, 1].set_xlim(0, t1[-1]); axes[0, 1].set_ylim(-0.7, 0.7)
    axes[0, 1].set_title('GET vocoded'); axes[0, 1].set_xlabel('time (s)')

    _myspectrogram(axes[1, 0], x,  fs,     'Original — Spectrogram')
    _myspectrogram(axes[1, 1], o, GET_FS, 'GET vocoded — Spectrogram')

    # 电极图（与 MATLAB plot_electrodogram 一致）
    ax_el = axes[0, 2]
    n_pulses = len(q['electrodes'])
    pulse_times = np.arange(1, n_pulses + 1) * q['periods'] / 1e6
    for idx in range(n_pulses):
        el = int(q['electrodes'][idx])
        if el == 0:
            continue
        ch = 23 - el                    # channel = 23 - electrode
        cl_norm = q['current_levels'][idx] / 255.0
        ax_el.vlines(pulse_times[idx], ch, ch + cl_norm, colors='k', linewidth=0.5)
    ax_el.set_xlabel('time (s)'); ax_el.set_ylabel('Electrode')
    ax_el.set_title('ACE Electrodogram')
    ax_el.set_ylim(0, 23)
    # Y 轴刻度标签：channel → electrode（与 MATLAB 一致）
    yticks = np.arange(1, 23)
    ax_el.set_yticks(yticks)
    ax_el.set_yticklabels(23 - yticks)

    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.show()

    # ---- 播放（可选）------------------------------------------------------
    try:
        import sounddevice as sd
        print(f'正在播放声码化音频（{GET_FS} Hz）...')
        sd.play(o.astype(np.float32), samplerate=GET_FS)
        sd.wait()
    except ImportError:
        print('提示：安装 sounddevice 可启用实时播放。')


if __name__ == '__main__':
    main()
