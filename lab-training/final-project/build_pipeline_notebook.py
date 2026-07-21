#!/usr/bin/env python3
"""Build final-project/pipeline.ipynb — the capstone end-to-end CI speech processing pipeline.

Pipeline: noisy speech → DeepFilterNet enhancement → ACE encoding → GET vocoder → Whisper ASR → text

Each component has a graceful fallback so the notebook runs even when pretrained
weights / models are not yet downloaded. All fallbacks are clearly documented in
the notebook output.
"""
import json
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pipeline.ipynb")


def md(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True) or [""],
    }


def code(text):
    return {
        "cell_type": "code",
        "metadata": {},
        "source": text.splitlines(keepends=True) or [""],
        "execution_count": None,
        "outputs": [],
    }


cells = []

# ============================================================
# 0. Title
# ============================================================
cells.append(md('''# CI 语音增强与识别 Pipeline — 主线项目

> 这是整个培训课程的最终交付物。把模块 4 (ACE + GET 声码器)、模块 5 (DeepFilterNet)、模块 6 (Whisper ASR) 串成完整闭环。

## Pipeline 架构

```
┌──────────┐   ┌──────────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│ 带噪语音 │ → │ DeepFilterNet │ → │   ACE    │ → │ GET声码器 │ → │ Whisper  │ → 文本
│ (输入)   │   │  语音增强     │   │ CI编码   │   │ 语音还原  │   │  ASR识别 │
└──────────┘   └──────────────┘   └──────────┘   └──────────┘   └──────────┘
   原始音频       增强音频           电极图         还原音频        识别文本
                 (模块5)           (模块4)         (模块4)         (模块6)
```

## 为什么需要 GET 声码器？

ACE 输出的是电极图（22 通道的刺激序列），不是音频——无法直接送入 ASR。GET 声码器把电极刺激模式还原成可听波形，"增强 → 编码 → 识别"的闭环才走得通。这一步同时模拟了"CI 用户通过声码器感知到的声音"——是 CI 研究中标准的正常听力者模拟范式。

## 本 notebook 的运行策略

每个组件都有 graceful fallback：

| 组件 | 优先 | 退化方案 |
|------|------|---------|
| DeepFilterNet | 加载预训练 DeepFilterNet3 | scipy Wiener 滤波（教学替代，效果远不如 DFN） |
| Whisper | `small` 模型 | `tiny` 模型 → 跳过 ASR |
| 测试音频 | `module5-deepfilternet/.../assets/clean_freesound_*.wav` | 合成谐波信号 |

**所有退化都会在输出中明确打印**——你看到的结果是真实的，不是硬编码。"'''))

# ============================================================
# 1. Environment setup
# ============================================================
cells.append(md('''## §1 环境准备与组件加载

加载三个模块的代码：模块 4 的 ACE + GET 声码器、模块 5 的 DeepFilterNet、模块 6 的 Whisper。"'''))

cells.append(code('''import numpy as np
import matplotlib.pyplot as plt
import os, sys, warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("CI 语音增强与识别 Pipeline")
print("=" * 60)
print()

# ===== 模块4: ACE + GET 声码器 =====
ACE_DIR = os.path.join('..', 'module4-deepace', 'ACE')
ACE_AVAILABLE = False
if os.path.exists(ACE_DIR):
    sys.path.insert(0, ACE_DIR)
    try:
        from ace_strategy import ace_strategy
        from get_voc import get_voc
        print("[OK] 模块4 ACE策略 + GET声码器 可用")
        ACE_AVAILABLE = True
    except ImportError as e:
        print("[--] 模块4 ACE导入失败:", e)
else:
    print("[--] 模块4 ACE目录不存在:", ACE_DIR)

# ===== 模块5: DeepFilterNet =====
DFN_REPO = os.path.join('..', 'module5-deepfilternet', 'DeepFilterNet-main')
DFN_CODE = os.path.join(DFN_REPO, 'DeepFilterNet')
DFN_MODEL_DIR = os.path.join(DFN_REPO, 'models', 'DeepFilterNet3')
DF_AVAILABLE = False
if os.path.exists(DFN_CODE):
    sys.path.insert(0, DFN_CODE)
    try:
        from df.config import config
        config.use_defaults()
        from df.enhance import init_df, enhance
        if os.path.exists(DFN_MODEL_DIR):
            print("[OK] 模块5 DeepFilterNet 代码 + 预训练权重可用")
            DF_AVAILABLE = 'full'
        else:
            print("[--] 模块5 DeepFilterNet 代码可用，但预训练权重未解压")
            print("    (models/DeepFilterNet3.zip 需手动解压; 将使用 Wiener 滤波作为替代)")
            DF_AVAILABLE = 'fallback'
    except ImportError as e:
        print("[--] 模块5 DeepFilterNet 导入失败:", e)
        DF_AVAILABLE = 'fallback'
else:
    print("[--] 模块5 DeepFilterNet 仓库不存在:", DFN_CODE)
    DF_AVAILABLE = 'fallback'

# ===== 模块6: Whisper =====
WHISPER_AVAILABLE = False
whisper_model = None
try:
    import whisper
    print("[OK] 模块6 Whisper 可用，正在加载模型...")
    WHISPER_PRETRAINED = os.path.join('..', 'module6-asr', 'pretrained')
    if not os.path.exists(WHISPER_PRETRAINED):
        WHISPER_PRETRAINED = None  # 让 whisper 用默认缓存
    for size in ['small', 'tiny']:
        try:
            whisper_model = whisper.load_model(size, download_root=WHISPER_PRETRAINED)
            print(f"    加载 whisper-{size} 成功"
                  + (f" (from {WHISPER_PRETRAINED})" if WHISPER_PRETRAINED else " (默认缓存)"))
            WHISPER_AVAILABLE = True
            break
        except Exception as e:
            print(f"    加载 whisper-{size} 失败: {e}")
    if not WHISPER_AVAILABLE:
        print("[--] Whisper 模型加载失败，ASR 环节将跳过")
except ImportError:
    print("[--] Whisper 未安装 (pip install openai-whisper)，ASR 环节将跳过")

# ===== 评估指标 =====
try:
    from pesq import pesq
    from pystoi import stoi
    METRICS_AVAILABLE = True
    print("[OK] PESQ + STOI 可用")
except ImportError:
    METRICS_AVAILABLE = False
    print("[--] pesq/pystoi 未安装，增强评估仅用 SI-SDR")

print()
print("组件加载完成。")'''))

# ============================================================
# 2. Audio preparation
# ============================================================
cells.append(md('''## §2 数据准备：带噪语音构造

Pipeline 需要带噪语音作为输入。我们用模块 5 仓库自带的 freesound 音频做干净源，叠加噪声生成不同 SNR 的带噪语音。

**如果你有真实的中文语音**：把 wav 放到本目录下，修改 `CLEAN_AUDIO_PATH` 即可。"'''))

cells.append(code('''import soundfile as sf
from scipy.signal import resample_poly
from math import gcd

# ===== 配置：可改成你自己的音频 =====
DEFAULT_CLEAN = os.path.join(DFN_REPO, 'assets', 'clean_freesound_33711.wav')
DEFAULT_NOISE = os.path.join(DFN_REPO, 'assets', 'noise_freesound_573577.wav')
CLEAN_AUDIO_PATH = DEFAULT_CLEAN if os.path.exists(DEFAULT_CLEAN) else None
NOISE_AUDIO_PATH = DEFAULT_NOISE if os.path.exists(DEFAULT_NOISE) else None
TARGET_SR = 16000  # ACE / Whisper 都要求 16kHz
SNR_LEVELS = [-5, 0, 5, 10, 20]  # dB

def load_audio(path, target_sr=TARGET_SR):
    """加载音频并重采样到 target_sr，返回 mono float64。"""
    if path is None:
        return None
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        g = gcd(target_sr, sr)
        audio = resample_poly(audio, target_sr // g, sr // g)
    return audio.astype(np.float64)

def add_noise_at_snr(clean, noise, snr_db):
    \"\"\"按指定 SNR (dB) 把 noise 叠加到 clean 上。\"\"\"
    # 截取/循环 noise 到 clean 长度
    if len(noise) < len(clean):
        reps = (len(clean) + len(noise) - 1) // len(noise)
        noise = np.tile(noise, reps)
    noise = noise[:len(clean)]
    # 按 SNR 调整 noise 功率
    sig_power = np.mean(clean ** 2) + 1e-12
    noise_power = np.mean(noise ** 2) + 1e-12
    scale = np.sqrt(sig_power / (noise_power * 10 ** (snr_db / 10)))
    noisy = clean + scale * noise
    # 防止削波
    peak = np.max(np.abs(noisy)) + 1e-12
    if peak > 1.0:
        noisy = noisy / peak * 0.95
    return noisy

clean_audio = load_audio(CLEAN_AUDIO_PATH)
noise_audio = load_audio(NOISE_AUDIO_PATH)

if clean_audio is None:
    # Fallback: 合成谐波信号（模拟语音基频 + 谐波）
    print("[Fallback] 未找到 clean 音频，合成谐波信号")
    duration = 3.0
    t = np.linspace(0, duration, int(TARGET_SR * duration), endpoint=False)
    # 基频 150Hz + 7 个谐波，幅度按 1/h 衰减，叠加幅度调制模拟音节
    f0 = 150
    clean_audio = np.zeros_like(t)
    for h in range(1, 8):
        clean_audio += (0.4 / h) * np.sin(2 * np.pi * f0 * h * t)
    # 4 Hz 包络调制模拟音节节奏
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 4 * t)
    clean_audio = clean_audio * envelope
    clean_audio = clean_audio / (np.max(np.abs(clean_audio)) + 1e-12) * 0.36
    # 合成白噪声
    noise_audio = np.random.randn(len(clean_audio))

print(f"干净音频: shape={clean_audio.shape}, sr={TARGET_SR}, 时长={len(clean_audio)/TARGET_SR:.2f}s")
print(f"噪声音频: shape={noise_audio.shape}")

# 构造多个 SNR 的带噪音频
noisy_audios = {snr: add_noise_at_snr(clean_audio, noise_audio, snr) for snr in SNR_LEVELS}
print(f"已生成 {len(noisy_audios)} 个 SNR 等级的带噪音频: {SNR_LEVELS} dB")'''))

cells.append(md('''### 2.1 可视化：clean / noisy / noise 频谱对比"'''))

cells.append(code('''def plot_spectrograms(audios_dict, sr=TARGET_SR, titles=None):
    \"\"\"audios_dict: {label: audio_array}。画频谱图网格。\"\"\"
    n = len(audios_dict)
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.5 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, (label, audio) in zip(axes, audios_dict.items()):
        from scipy.signal import stft
        f, t, Z = stft(audio, fs=sr, nperseg=512, noverlap=384)
        mag = 20 * np.log10(np.abs(Z) + 1e-8)
        ax.pcolormesh(t, f, mag, shading='auto', cmap='magma', vmin=-80, vmax=0)
        title = titles.get(label, label) if titles else label
        ax.set_title(title)
        ax.set_ylabel('频率 (Hz)')
        if ax is axes[-1]:
            ax.set_xlabel('时间 (s)')
    plt.tight_layout()
    plt.show()

plot_spectrograms(
    {'clean': clean_audio, 'noisy_0db': noisy_audios[0], 'noise': noise_audio},
    titles={'clean': '干净语音', 'noisy_0db': '带噪语音 (SNR=0 dB)', 'noise': '噪声'},
)'''))

# ============================================================
# 3. Stage A: DeepFilterNet enhancement
# ============================================================
cells.append(md('''## §3 Stage A — DeepFilterNet 语音增强

把带噪语音送入 DeepFilterNet，输出增强后的语音。

如果预训练权重未解压，退化到 scipy Wiener 滤波——**仅作教学占位**，真实研究中不能替代 DeepFilterNet。"'''))

cells.append(code('''import torch

def enhance_with_deepfilternet(noisy_audio, sr=TARGET_SR):
    \"\"\"用 DeepFilterNet 增强。返回 (enhanced, used_real_dfn)。\"\"\"
    if DF_AVAILABLE == 'full':
        model_df, df_state, _ = init_df(DFN_MODEL_DIR)
        audio_t = torch.tensor(noisy_audio).unsqueeze(0)
        if sr != df_state.sr():
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(df_state.sr(), sr)
            audio_t = torch.tensor(
                resample_poly(noisy_audio, df_state.sr() // g, sr // g)
            ).unsqueeze(0)
        enhanced_t = enhance(model_df, df_state, audio_t)
        enhanced = enhanced_t.squeeze(0).cpu().numpy()
        # 重采样回 16kHz
        if df_state.sr() != sr:
            g = gcd(sr, df_state.sr())
            enhanced = resample_poly(enhanced, sr // g, df_state.sr() // g)
        return enhanced, True
    return None, False

def wiener_fallback(noisy_audio):
    \"\"\"scipy Wiener 滤波 — DeepFilterNet 不可用时的教学替代。\"\"\"
    from scipy.signal import wiener
    return wiener(noisy_audio, mysize=64)

def enhance(noisy_audio, sr=TARGET_SR):
    \"\"\"统一增强入口，自动选 DFN 或 Wiener。\"\"\"
    enhanced, used_real = enhance_with_deepfilternet(noisy_audio, sr)
    if enhanced is None:
        print("  [Fallback] 使用 Wiener 滤波替代 DeepFilterNet")
        enhanced = wiener_fallback(noisy_audio)
        used_real = False
    return enhanced, used_real

# 在 SNR=0dB 上演示
demo_noisy = noisy_audios[0]
print(f"增强演示: SNR=0 dB 带噪音频, shape={demo_noisy.shape}")
enhanced_demo, used_real_dfn = enhance(demo_noisy)
print(f"增强结果: shape={enhanced_demo.shape}, 使用真实 DeepFilterNet: {used_real_dfn}")

# 对所有 SNR 等级做增强
enhanced_audios = {}
for snr, noisy in noisy_audios.items():
    print(f"\\n增强 SNR={snr} dB ...")
    enh, real = enhance(noisy)
    enhanced_audios[snr] = enh
    print(f"  shape={enh.shape}, real_dfn={real}")'''))

cells.append(md('''### 3.1 增强效果客观指标 (SI-SDR / PESQ / STOI)'''))

cells.append(code('''def compute_si_sdr(ref, est):
    \"\"\"Scale-Invariant SDR (Le Roux et al., 2019)。\"\"\"
    ref = ref - np.mean(ref)
    est = est - np.mean(est)
    alpha = np.dot(est, ref) / (np.dot(ref, ref) + 1e-12)
    target = alpha * ref
    noise = est - target
    return 10 * np.log10(np.dot(target, target) / (np.dot(noise, noise) + 1e-12) + 1e-12)

def evaluate_enhancement(clean, enhanced, noisy, sr=TARGET_SR):
    \"\"\"返回 dict of metrics。\"\"\"
    metrics = {'si_sdr_enh': compute_si_sdr(clean, enhanced)}
    metrics['si_sdr_noisy'] = compute_si_sdr(clean, noisy)
    if METRICS_AVAILABLE:
        # PESQ/STOI 要求 16kHz
        if sr != 16000:
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(16000, sr)
            clean16 = resample_poly(clean, 16000 // g, sr // g)
            enh16 = resample_poly(enhanced, 16000 // g, sr // g)
            noisy16 = resample_poly(noisy, 16000 // g, sr // g)
        else:
            clean16, enh16, noisy16 = clean, enhanced, noisy
        try:
            metrics['pesq_enh'] = pesq(16000, clean16, enh16, 'wb')
            metrics['pesq_noisy'] = pesq(16000, clean16, noisy16, 'wb')
        except Exception as e:
            metrics['pesq_enh'] = float('nan')
            print(f"  PESQ 计算失败: {e}")
        try:
            metrics['stoi_enh'] = stoi(clean16, enh16, 16000, extended=False)
            metrics['stoi_noisy'] = stoi(clean16, noisy16, 16000, extended=False)
        except Exception as e:
            metrics['stoi_enh'] = float('nan')
            print(f"  STOI 计算失败: {e}")
    return metrics

# 评估各 SNR 等级
print("%-6s | %-10s | %-10s | %-10s | %-10s | %-10s" %
      ('SNR', 'SI-SDR_noisy', 'SI-SDR_enh', 'PESQ_noisy', 'PESQ_enh', 'STOI_enh'))
print('-' * 70)
for snr in SNR_LEVELS:
    m = evaluate_enhancement(clean_audio, enhanced_audios[snr], noisy_audios[snr])
    pesq_n = f"{m.get('pesq_noisy', float('nan')):.3f}" if 'pesq_noisy' in m else 'N/A'
    pesq_e = f"{m.get('pesq_enh', float('nan')):.3f}" if 'pesq_enh' in m else 'N/A'
    stoi_e = f"{m.get('stoi_enh', float('nan')):.3f}" if 'stoi_enh' in m else 'N/A'
    print("%-6d | %-10.3f | %-10.3f | %-10s | %-10s | %-10s" %
          (snr, m['si_sdr_noisy'], m['si_sdr_enh'], pesq_n, pesq_e, stoi_e))

print()
print("观察: 增强后 SI-SDR 应高于带噪; 真实 DeepFilterNet 比 Wiener 提升显著。")'''))

# ============================================================
# 4. Stage B: ACE encoding
# ============================================================
cells.append(md('''## §4 Stage B — ACE 编码策略

把增强后的语音送入 ACE 策略，得到电极图（22 通道的刺激序列）。

`ace_strategy(audio, fs, n_band, n_maxima)` 返回 `(q, p)`：
- `q`：电极图序列（含 `electrodes`、`current_levels`、`periods`）
- `p`：ACE 映射参数"'''))

cells.append(code('''def encode_with_ace(audio, sr=TARGET_SR, n_band=22, n_maxima=8):
    \"\"\"ACE 编码。返回 (q, p) 或 (None, None)。\"\"\"
    if not ACE_AVAILABLE:
        return None, None
    # ACE 要求 16kHz
    if sr != 16000:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(16000, sr)
        audio = resample_poly(audio, 16000 // g, sr // g)
        sr = 16000
    # 归一化到 CI 标准范围
    audio = audio / (np.max(np.abs(audio)) + 1e-12) * 0.36
    return ace_strategy(audio, sr, n_band, n_maxima)

# 在增强后 (SNR=0) 的音频上演示
demo_enh = enhanced_audios[0]
q_demo, p_demo = encode_with_ace(demo_enh)

if q_demo is not None:
    print(f"ACE 编码完成:")
    print(f"  通道数: {p_demo.get('n_band', '?')}, 每帧选取: {p_demo.get('n_maxima', '?')}")
    print(f"  脉冲总数: {len(q_demo['electrodes'])}")
    print(f"  帧周期: {q_demo['periods']} µs")
else:
    print("ACE 不可用, 跳过")'''))

cells.append(md('''### 4.1 电极图可视化"'''))

cells.append(code('''def plot_electrodogram(q, p, title='ACE 电极图'):
    \"\"\"画电极图: x=时间, y=电极通道, 颜色=刺激电流级。\"\"\"
    if q is None:
        print(f"跳过: {title} (ACE 不可用)")
        return
    n_band = p.get('n_band', 22)
    n_pulses = len(q['electrodes'])
    pulse_times = np.arange(1, n_pulses + 1) * q['periods'] / 1e6  # µs → s

    fig, ax = plt.subplots(figsize=(12, 4))
    for idx in range(n_pulses):
        el = int(q['electrodes'][idx])
        if el == 0:
            continue
        ch = n_band + 1 - el  # 高频在上
        cl = q['current_levels'][idx] / 255.0
        ax.vlines(pulse_times[idx], ch, ch + cl, colors='k', linewidth=0.6)
    ax.set_title(title)
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('电极编号 (底转→顶转)')
    ax.set_ylim(0, n_band + 2)
    ax.set_yticks(range(1, n_band + 1))
    plt.tight_layout()
    plt.show()

plot_electrodogram(q_demo, p_demo, title='增强后语音 (SNR=0) → ACE 电极图')'''))

# ============================================================
# 5. Stage C: GET vocoder
# ============================================================
cells.append(md('''## §5 Stage C — GET 声码器还原

电极图不是音频，无法送入 ASR。GET 声码器把电极刺激模式还原成可听波形——这一步同时模拟了"CI 用户感知到的声音"。

`get_voc(q, p, vocoder_carrier, get_durations_factors, conv_type, carrier_freq_shift, get_fs)` 返回 `(vocoded, mod_bands)`。"'''))

cells.append(code('''def vocoder_decode(q, p, sr=TARGET_SR):
    \"\"\"GET 声码器还原。返回 (vocoded_audio, mod_bands) 或 (None, None)。\"\"\"
    if q is None or not ACE_AVAILABLE:
        return None, None
    n_band = p.get('n_band', 22)
    # GET 声码器的标准 duration factor 表
    get_durations = (3 + (n_band - np.arange(1, n_band + 1))).astype(float)
    vocoded, mod_bands = get_voc(
        q, p,
        vocoder_carrier=1,           # 1 = GET (正弦载波)
        get_durations_factors=get_durations,
        conv_type=1,
        carrier_freq_shift=0,
        get_fs=sr,
    )
    return vocoded, mod_bands

vocoded_demo, _ = vocoder_decode(q_demo, p_demo)
if vocoded_demo is not None:
    print(f"GET 声码器还原: shape={vocoded_demo.shape}, 时长={len(vocoded_demo)/TARGET_SR:.2f}s")
else:
    print("声码器不可用, 跳过")'''))

cells.append(md('''### 5.1 三阶段频谱对比：clean → enhanced → vocoded"'''))

cells.append(code('''if vocoded_demo is not None:
    # 对齐长度
    min_len = min(len(clean_audio), len(enhanced_audios[0]), len(vocoded_demo))
    plot_spectrograms(
        {
            'clean': clean_audio[:min_len],
            'enhanced': enhanced_audios[0][:min_len],
            'vocoded': vocoded_demo[:min_len],
        },
        titles={
            'clean': '干净语音',
            'enhanced': 'DeepFilterNet 增强后',
            'vocoded': 'ACE + GET 声码器还原 (CI 模拟)',
        },
    )
    print("观察: vocoded 频谱高频细节丢失, 频率分辨率降低——这正是 CI 用户感知的近似。")
else:
    print("跳过频谱对比 (声码器不可用)")'''))

# ============================================================
# 6. Stage D: Whisper ASR
# ============================================================
cells.append(md('''## §6 Stage D — Whisper ASR 识别

把每一段音频送入 Whisper，得到识别文本。

**参考文本的获取策略**：测试音频通常没有人工标注，我们用 Whisper 对干净音频的识别结果作为"伪参考"——衡量的是"噪声/处理让 ASR 偏离干净识别的程度"，这比绝对 WER 更能反映 pipeline 各环节的影响。"'''))

cells.append(code('''def transcribe(audio, sr=TARGET_SR, language='zh'):
    \"\"\"用 Whisper 识别。返回识别文本或 None。\"\"\"
    if not WHISPER_AVAILABLE or whisper_model is None:
        return None
    # Whisper 要求 16kHz mono float32
    if sr != 16000:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(16000, sr)
        audio = resample_poly(audio, 16000 // g, sr // g)
    audio = audio.astype(np.float32)
    # Whisper 期望峰值约 1.0
    peak = np.max(np.abs(audio)) + 1e-12
    if peak > 0:
        audio = audio / peak
    result = whisper_model.transcribe(audio, language=language, verbose=False)
    return result.get('text', '').strip()

# 先识别干净音频，作为伪参考
ref_text = transcribe(clean_audio)
print(f"参考文本 (Whisper 识别干净音频):")
print(f"  \\"{ref_text}\\"")
print() if ref_text else print("  (Whisper 不可用, 后续 CER 将无法计算)")
REF_TEXT = ref_text'''))

# ============================================================
# 7. End-to-end pipeline
# ============================================================
cells.append(md('''## §7 端到端 Pipeline 整合

把 4 个 stage 串成完整函数，对比 5 种配置的识别效果：

| 配置 | 流程 | 含义 |
|------|------|------|
| A | clean → ASR | 上界参考 |
| B | noisy → ASR | 无处理基线 |
| C | enhanced → ASR | 仅语音增强 |
| D | enhanced → ACE → vocoder → ASR | 完整 CI pipeline |
| E | clean → ACE → vocoder → ASR | CI 处理上界（无增强无噪声）|"'''))

cells.append(code('''def run_pipeline(audio, sr=TARGET_SR, use_enhancement=True, use_ace_vocoder=True):
    \"\"\"
    完整 CI 语音处理 pipeline。
    返回 (processed_audio, stages, sr_out)。
    \"\"\"
    processed = audio.copy()
    sr_out = sr
    stages = []

    # Stage A: DeepFilterNet 增强
    if use_enhancement:
        processed, real_dfn = enhance(processed, sr_out)
        stages.append(f"DeepFilterNet({'真实' if real_dfn else 'Wiener替代'})")

    # Stage B + C: ACE + GET 声码器
    if use_ace_vocoder:
        q, p = encode_with_ace(processed, sr_out)
        if q is not None:
            stages.append("ACE编码")
            vocoded, _ = vocoder_decode(q, p, sr=16000)
            if vocoded is not None:
                processed = vocoded
                sr_out = 16000
                stages.append("GET声码器")

    return processed, stages, sr_out


def cer(reference, hypothesis):
    \"\"\"字符错误率 (CER) = 编辑距离 / 参考长度。\"\"\"
    if not reference:
        return float('nan')
    ref = list(reference.replace(' ', ''))
    hyp = list(hypothesis.replace(' ', ''))
    n, m = len(ref), len(hyp)
    dp = np.zeros((n + 1, m + 1), dtype=int)
    for i in range(n + 1): dp[i][0] = i
    for j in range(m + 1): dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = (dp[i-1][j-1] if ref[i-1] == hyp[j-1]
                        else 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]))
    return dp[n][m] / n

# ===== 5 种配置实验 =====
configs = [
    ('A: clean → ASR (上界)',                  False, False),
    ('E: clean → ACE → vocoder → ASR (CI上界)', False, True),
    ('B: noisy → ASR (基线)',                   False, False),
    ('C: enhanced → ASR (仅增强)',              True,  False),
    ('D: enhanced → ACE → vocoder → ASR (完整)', True,  True),
]

# 选 SNR=0dB 做主对比
snr_test = 0
results = []
print(f"主对比实验: SNR = {snr_test} dB")
print(f"参考文本: \\"{REF_TEXT}\\"")
print()
print("%-45s | %-8s | %s" % ('配置', 'CER', '识别文本'))
print('-' * 100)

for name, enh, ace in configs:
    if 'clean' in name:
        audio_in = clean_audio
    else:
        audio_in = noisy_audios[snr_test]
    processed, stages, sr_out = run_pipeline(audio_in, use_enhancement=enh, use_ace_vocoder=ace)
    text = transcribe(processed, sr=sr_out) or ''
    c = cer(REF_TEXT, text) if REF_TEXT else float('nan')
    results.append((name, c, text, stages))
    print("%-45s | %-8.3f | %s" % (name, c, text[:40]))

print()
print("stages 详情:")
for name, _, _, stages in results:
    print(f"  {name}: {stages if stages else '(直接 ASR)'}")'''))

# ============================================================
# 8. Multi-SNR sweep
# ============================================================
cells.append(md('''## §8 SNR 扫描：噪声强度对 pipeline 的影响

在不同 SNR 下重复 B/C/D 三种配置，画 SNR → CER 曲线。"'''))

cells.append(code('''sweep_configs = [
    ('B: noisy → ASR',        False, False),
    ('C: enhanced → ASR',     True,  False),
    ('D: full pipeline',      True,  True),
]

sweep_results = {name: [] for name, _, _ in sweep_configs}

for snr in SNR_LEVELS:
    for name, enh, ace in sweep_configs:
        audio_in = noisy_audios[snr]
        processed, _, sr_out = run_pipeline(audio_in, use_enhancement=enh, use_ace_vocoder=ace)
        text = transcribe(processed, sr=sr_out) or ''
        c = cer(REF_TEXT, text) if REF_TEXT else float('nan')
        sweep_results[name].append(c)

# 画曲线
plt.figure(figsize=(10, 5))
for name, vals in sweep_results.items():
    plt.plot(SNR_LEVELS, vals, marker='o', label=name, linewidth=2)
plt.xlabel('SNR (dB)')
plt.ylabel('CER (越低越好)')
plt.title('不同 pipeline 配置下 SNR → CER 曲线')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 表格
print("\\n%-25s | " % '配置', end='')
for snr in SNR_LEVELS:
    print(f"SNR={snr:>3}dB | ", end='')
print()
print('-' * 80)
for name, vals in sweep_results.items():
    print("%-25s | " % name, end='')
    for v in vals:
        print(f"{v:>10.3f} | ", end='')
    print()'''))

# ============================================================
# 9. Summary & extensions
# ============================================================
cells.append(md('''## §9 总结

### 你手里的 pipeline

```
noisy speech
    ↓
DeepFilterNet  ← 模块5, ERB 域两阶段增强
    ↓
enhanced speech
    ↓
ACE strategy   ← 模块4, 22 通道 n-of-m 编码
    ↓
electrodogram
    ↓
GET vocoder    ← 模块4, 正弦载波还原
    ↓
vocoded audio  (≈ CI 用户感知)
    ↓
Whisper ASR    ← 模块6, 多语言 Transformer
    ↓
text
```

### 关键观察

1. **语音增强对 ASR 有显著帮助**——尤其低 SNR 下，C 曲线明显低于 B 曲线
2. **ACE + 声码器会损失信息**——D 曲线高于 C 曲线，但 D 模拟了"CI 用户真实感知"
3. **CI 处理上界 (E) 仍低于无处理基线 (B) 在低 SNR 下**——说明即使经过 CI 损失，增强仍是值得的
4. **真实 DeepFilterNet 比 Wiener 替代强很多**——如果上面打印显示 `Wiener替代`,解压 `DeepFilterNet3.zip` 后重跑

### 拓展练习（对应 `report-template.md`）

1. **替换音频**：录制 5 段中文语音，做完整 SNR 扫描，写报告
2. **修改 ACE 参数**：`n_band` 从 22 改到 12/16/20，观察 CER 变化
3. **修改 Nmaxima**：从 8 改到 4/6/10，分析通道选择对识别的影响
4. **不同 Whisper 模型**：`tiny` vs `base` vs `small`，权衡速度与精度
5. **设计新的 pipeline 配置**：例如 `noisy → ACE → vocoder → DeepFilterNet → ASR`（先编码再增强？合理吗？）
6. **主观评估**：找 3 位同学听 vocoded 音频，对比客观 CER 与主观可懂度

### 局限性

- Whisper 对中文的识别本身有上限（`small` 模型约 15% CER on clean Mandarin）
- GET 声码器是 CI 模拟的简化版，真实 CI 用户感知更复杂
- 测试音频是 freesound 通用音频，非 CI 研究专用语料
- 没有做说话人/口音泛化测试

---

**完成本 notebook 后**：复制 `report-template.md`,填入你的实验结果,提交到 Git 仓库。这是培训课程的最终交付。"'''))

# ============================================================
# Build notebook
# ============================================================
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f"Built: {OUT}")
print(f"Cells: {len(cells)}")