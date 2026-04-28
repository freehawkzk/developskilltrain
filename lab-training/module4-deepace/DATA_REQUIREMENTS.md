# 模块4 数据集需求

本文档说明模块4"DeepACE模型解析"所需的数据集，以及如何合成 mini 数据集用于快速训练演示。

---

## 一、DeepACE 数据格式说明

通过阅读 `DeepACE_torch/dataset.py` 源码，DeepACE 的训练数据格式如下：

### 输入（mixture）
- 格式：WAV 文件，16kHz，单声道
- 内容：带噪语音（语音 + 噪声的混合）
- 路径：`data/train/mixture/` 和 `data/valid/mixture/`

### 目标（target）
- 格式：`.mat` 文件（MATLAB格式），包含一个变量 `lgf`
- 内容：ACE 策略处理后的电极图（electrodogram），形状 `(time_steps, 22)`
- 22 对应 CI 的 22 个电极通道
- `stim_rate`（刺激速率）默认 1000 Hz，即 `block_shift = 16000/1000 = 16` 样本/帧
- 路径：`data/train/targetLGF/` 和 `data/valid/targetLGF/`

### 数据生成流程

```
干净语音 + 噪声 → 混合(mixture WAV) → ACE策略处理 → 电极图(target .mat)
```

关键点：target 不是直接对 mixture 做 ACE，而是对**干净语音**做 ACE 得到的理想电极图。模型的任务是：从带噪语音直接预测出干净语音的电极图，跳过传统的 ACE 流程。

---

## 二、公开数据集

### 1. 语音数据（用于生成 mixture）

可复用模块3已下载的数据集：

| 数据集 | 用途 | 说明 |
|--------|------|------|
| Speech Commands | 语音源 | 已在模块3下载 |
| VoiceBank-DEMAND | 语音增强标准测试集 | 可选，含多种SNR的带噪语音 |

如果模块3已完成，Speech Commands 应已在 `module3-audio-classification/data/speech_commands/`。

### 2. DeepACE 原始论文数据（可选）

DeepACE 论文作者使用的训练数据：
- **TIMIT** 或 **LibriSpeech** 作为语音源
- **DNS Challenge** 噪声库

如果希望复现论文结果，需要使用完整数据集。对于**课堂教学**，使用下方合成的小数据集即可。

---

## 三、Mini 数据集合成脚本

以下脚本自动生成一个 mini 数据集（训练集约20条，验证集约5条），确保 5 分钟内能跑完一个 epoch。脚本会：

1. 从 Speech Commands 抽取干净语音
2. 生成噪声并混合
3. 用本模块的 ACE 代码生成 target 电极图

**前置条件**：
- Speech Commands 数据集已下载到 `module3-audio-classification/data/speech_commands/`（或自行指定路径）
- ACE 代码可用（`ACE/` 目录下的代码）

### 合成脚本：`scripts/prepare_mini_dataset.py`

```python
#!/usr/bin/env python3
"""
生成 DeepACE 训练用的 mini 数据集。

输出目录结构:
  DeepACE_torch/data/
  ├── train/
  │   ├── mixture/     # 带噪语音 WAV
  │   └── targetLGF/   # ACE电极图 .mat
  ├── valid/
  │   ├── mixture/
  │   └── targetLGF/
  └── test/
      └── mixture/

每条数据: 4秒, 16kHz单声道
训练集: 20条, 验证集: 5条, 测试集: 3条
"""

import os
import sys
import glob
import random
import numpy as np
import soundfile as sf
import scipy.io

# ============================================================
# 配置
# ============================================================
SEED = 42
SR = 16000          # 采样率
DURATION = 4.0       # 音频时长（秒）
N_BAND = 22           # ACE频带数
N_MAXIMA = 8          # ACE每帧选取的最大通道数
N_TRAIN = 20          # 训练样本数
N_VALID = 5           # 验证样本数
N_TEST = 3            # 测试样本数
SNR_RANGE = (-5, 15)  # 噪声SNR范围(dB)

# 路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODULE4_DIR = os.path.dirname(SCRIPT_DIR)
ACE_DIR = os.path.join(MODULE4_DIR, "ACE")
DEEPACE_DIR = os.path.join(MODULE4_DIR, "DeepACE_torch")

# Speech Commands 路径（可修改）
SPEECH_COMMANDS_DIR = os.path.join(
    os.path.dirname(MODULE4_DIR),
    "module3-audio-classification", "data", "speech_commands"
)

# 输出路径
OUTPUT_DIR = os.path.join(DEEPACE_DIR, "data")
os.makedirs(os.path.join(OUTPUT_DIR, "train", "mixture"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "train", "targetLGF"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "valid", "mixture"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "valid", "targetLGF"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "test", "mixture"), exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)


def load_speech_commands():
    """加载 Speech Commands 中的语音文件"""
    if not os.path.exists(SPEECH_COMMANDS_DIR):
        print("Speech Commands 未找到，使用合成语音")
        return None

    files = []
    for word_dir in os.listdir(SPEECH_COMMANDS_DIR):
        word_path = os.path.join(SPEECH_COMMANDS_DIR, word_dir)
        if os.path.isdir(word_path) and word_dir != "_background_noise_":
            for f in os.listdir(word_path):
                if f.endswith(".wav"):
                    files.append(os.path.join(word_path, f))

    random.shuffle(files)
    print("找到 %d 个语音文件" % len(files))
    return files


def generate_silence(duration, sr):
    """生成静音信号"""
    return np.zeros(int(sr * duration))


def generate_tone_speech(duration, sr):
    """生成模拟语音信号（多个谐波叠加）"""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = np.zeros_like(t)
    # 基频 + 谐波，模拟语音
    for _ in range(random.randint(2, 5)):
        freq = random.uniform(100, 600)
        amp = random.uniform(0.1, 0.4)
        signal += amp * np.sin(2 * np.pi * freq * t)
    # 加微弱噪声模拟自然语音
    signal += 0.02 * np.random.randn(len(t))
    # 归一化
    signal = signal / (np.max(np.abs(signal)) + 1e-8) * 0.8
    return signal


def generate_noise(duration, sr, noise_type="white"):
    """生成不同类型的噪声"""
    n_samples = int(sr * duration)
    if noise_type == "white":
        noise = np.random.randn(n_samples) * 0.3
    elif noise_type == "pink":
        # 简单的粉红噪声近似
        white = np.random.randn(n_samples)
        noise = np.cumsum(white) / np.sqrt(n_samples) * 0.3
    elif noise_type == "babble":
        # 模拟多人说话噪声
        noise = np.zeros(n_samples)
        for _ in range(random.randint(3, 6)):
            t = np.linspace(0, duration, n_samples, endpoint=False)
            freq = random.uniform(80, 500)
            amp = random.uniform(0.05, 0.15)
            noise += amp * np.sin(2 * np.pi * freq * t + random.uniform(0, 2*np.pi))
    else:
        noise = np.random.randn(n_samples) * 0.3
    return noise


def add_noise_at_snr(speech, noise, snr_db):
    """在指定SNR下混合语音和噪声"""
    if len(noise) < len(speech):
        noise = np.tile(noise, (len(speech) // len(noise)) + 1)[:len(speech)]
    else:
        noise = noise[:len(speech)]

    speech_power = np.mean(speech ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power < 1e-10:
        return speech.copy()

    snr_linear = 10 ** (snr_db / 10)
    noise_scaled = noise * np.sqrt(speech_power / (noise_power * snr_linear + 1e-8))
    mixture = speech + noise_scaled
    mixture = mixture / (np.max(np.abs(mixture)) + 1e-8) * 0.9
    return mixture


def process_ace(waveform, sr=16000):
    """用 ACE 策略处理音频，生成电极图"""
    # 添加 ACE 目录到 Python 路径
    if ACE_DIR not in sys.path:
        sys.path.insert(0, ACE_DIR)

    try:
        from ace_strategy import ace_strategy
        q, p = ace_strategy(waveform, sr, N_BAND, N_MAXIMA)
        # q 是电极图序列，提取 lgf（电刺激幅度）
        # ACE 输出 q 包含多个字段，电极图在 q['elecData'] 或类似字段中
        # 具体格式需要查看 ace_process.py 的输出
        if isinstance(q, dict) and 'lgf' in q:
            electrodogram = q['lgf']
        elif isinstance(q, dict):
            # 尝试不同的键名
            for key in ['elecData', 'electrodogram', 'data', 'stim']:
                if key in q:
                    electrodogram = q[key]
                    break
            else:
                # 如果都没有，从 q 中取第一个数组
                for key, val in q.items():
                    if isinstance(val, np.ndarray) and val.ndim >= 2:
                        electrodogram = val
                        break
                else:
                    print("WARNING: ACE输出格式未知，使用模拟电极图")
                    electrodogram = None
        else:
            electrodogram = None
    except Exception as e:
        print("ACE处理失败: %s, 使用模拟电极图" % str(e))
        electrodogram = None

    if electrodogram is None:
        # 如果ACE处理失败，生成模拟电极图
        electrodogram = generate_simulated_electrodogram(waveform, sr)

    return electrodogram


def generate_simulated_electrodogram(waveform, sr=16000, n_channels=22, stim_rate=1000):
    """生成模拟的电极图（当ACE代码不可用时的备用方案）"""
    import librosa

    block_shift = int(np.ceil(sr / stim_rate))
    n_frames = int(np.ceil(len(waveform) / block_shift))

    # 提取梅尔频谱作为基础
    mel_spec = librosa.feature.melspectrogram(
        y=waveform, sr=sr, n_fft=512, hop_length=block_shift,
        n_mels=n_channels, fmin=0, fmax=sr//2
    )
    mel_spec = librosa.power_to_db(mel_spec + 1e-10)
    # 归一化到 [0, 1]
    mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min() + 1e-8)
    # 确保帧数匹配
    if mel_spec.shape[1] > n_frames:
        mel_spec = mel_spec[:, :n_frames]
    elif mel_spec.shape[1] < n_frames:
        mel_spec = np.pad(mel_spec, ((0,0),(0, n_frames - mel_spec.shape[1])))

    # 模拟 N-of-M 通道选择
    n_maxima = N_MAXIMA
    for col in range(mel_spec.shape[1]):
        frame = mel_spec[:, col].copy()
        # 只保留最大的 n_maxima 个通道
        if n_maxima < n_channels:
            threshold = np.sort(frame)[-n_maxima]
            frame[frame < threshold] = 0
        mel_spec[:, col] = frame

    # 转为 (time_steps, channels) 格式
    electrodogram = mel_spec.T  # (T, 22)
    return electrodogram


def create_sample(speech_files, idx, sr=16000, duration=4.0):
    """创建一条训练样本：mixture WAV + target .mat"""
    target_len = int(sr * duration)

    # 加载或生成干净语音
    if speech_files and idx < len(speech_files):
        import librosa
        speech, orig_sr = librosa.load(speech_files[idx], sr=None)
        if orig_sr != sr:
            speech = librosa.resample(speech, orig_sr=orig_sr, target_sr=sr)
    else:
        speech = generate_tone_speech(duration, sr)

    # 截断或循环到目标长度
    if len(speech) > target_len:
        start = random.randint(0, len(speech) - target_len)
        speech = speech[start:start + target_len]
    elif len(speech) < target_len:
        speech = np.tile(speech, (target_len // len(speech)) + 1)[:target_len]

    # 添加噪声
    snr = random.uniform(*SNR_RANGE)
    noise_type = random.choice(["white", "pink", "babble"])
    noise = generate_noise(duration, sr, noise_type)
    mixture = add_noise_at_snr(speech, noise, snr)

    # ACE 处理干净语音得到 target 电极图
    electrodogram = process_ace(speech, sr)

    return mixture, electrodogram


def main():
    print("=" * 60)
    print("DeepACE Mini Dataset Generator")
    print("=" * 60)

    # 加载语音文件
    speech_files = load_speech_commands()

    # 生成训练集
    print("\n生成训练集 (%d 条)..." % N_TRAIN)
    for i in range(N_TRAIN):
        mixture, elec = create_sample(speech_files, i)
        # 保存 mixture WAV
        mix_path = os.path.join(OUTPUT_DIR, "train", "mixture", "%04d.wav" % i)
        sf.write(mix_path, mixture.astype(np.float32), SR)
        # 保存 target .mat (electrodogram: (T, 22))
        tgt_path = os.path.join(OUTPUT_DIR, "train", "targetLGF", "%04d.mat" % i)
        scipy.io.savemat(tgt_path, {"lgf": elec})
        if (i + 1) % 5 == 0:
            print("  已完成 %d/%d" % (i+1, N_TRAIN))

    # 生成验证集
    print("\n生成验证集 (%d 条)..." % N_VALID)
    for i in range(N_VALID):
        mixture, elec = create_sample(speech_files, N_TRAIN + i)
        mix_path = os.path.join(OUTPUT_DIR, "valid", "mixture", "%04d.wav" % i)
        sf.write(mix_path, mixture.astype(np.float32), SR)
        tgt_path = os.path.join(OUTPUT_DIR, "valid", "targetLGF", "%04d.mat" % i)
        scipy.io.savemat(tgt_path, {"lgf": elec})
        print("  已完成 %d/%d" % (i+1, N_VALID))

    # 生成测试集（只有 mixture，没有 target）
    print("\n生成测试集 (%d 条)..." % N_TEST)
    for i in range(N_TEST):
        mixture, _ = create_sample(speech_files, N_TRAIN + N_VALID + i)
        mix_path = os.path.join(OUTPUT_DIR, "test", "mixture", "%04d.wav" % i)
        sf.write(mix_path, mixture.astype(np.float32), SR)
        print("  已完成 %d/%d" % (i+1, N_TEST))

    # 验证
    print("\n数据集验证:")
    train_mix = glob.glob(os.path.join(OUTPUT_DIR, "train", "mixture", "*.wav"))
    train_tgt = glob.glob(os.path.join(OUTPUT_DIR, "train", "targetLGF", "*.mat"))
    valid_mix = glob.glob(os.path.join(OUTPUT_DIR, "valid", "mixture", "*.wav"))
    valid_tgt = glob.glob(os.path.join(OUTPUT_DIR, "valid", "targetLGF", "*.mat"))
    test_mix = glob.glob(os.path.join(OUTPUT_DIR, "test", "mixture", "*.wav"))

    print("  训练集: %d mixture, %d target" % (len(train_mix), len(train_tgt)))
    print("  验证集: %d mixture, %d target" % (len(valid_mix), len(valid_tgt)))
    print("  测试集: %d mixture" % len(test_mix))

    # 检查一条数据
    if train_mix:
        mix, sr = sf.read(train_mix[0])
        tgt = scipy.io.loadmat(train_tgt[0])
        elec = tgt["lgf"]
        print("\n样本示例:")
        print("  Mixture: shape=%s, sr=%d, duration=%.1fs" % (mix.shape, sr, len(mix)/sr))
        print("  Target electrodogram: shape=%s (time_steps x channels)" % (elec.shape,))
        print("  期望: mixture=(%d,), target=(%d, 22)" % (int(SR*DURATION), int(SR*DURATION/16)))

    print("\n完成! 数据保存在: %s" % OUTPUT_DIR)
    print("可以运行训练: cd DeepACE_torch && python train.py")


if __name__ == "__main__":
    main()
```

---

## 四、数据集最终目录结构

运行合成脚本后，`DeepACE_torch/data/` 目录应如下：

```
DeepACE_torch/data/
├── train/
│   ├── mixture/
│   │   ├── 0000.wav     # 带噪语音，4秒，16kHz
│   │   ├── 0001.wav
│   │   └── ...          # 共20个
│   └── targetLGF/
│       ├── 0000.mat     # ACE电极图，含 'lgf' 变量，shape=(T, 22)
│       ├── 0001.mat
│       └── ...          # 共20个
├── valid/
│   ├── mixture/
│   │   └── ...          # 共5个
│   └── targetLGF/
│       └── ...          # 共5个
└── test/
    └── mixture/
        └── ...          # 共3个（无target）
```

---

## 五、准备顺序

1. **（如已有模块3数据）** 直接运行 `python scripts/prepare_mini_dataset.py`
2. **（如无 Speech Commands）** 脚本会自动降级使用合成语音，无需任何外部数据
3. 运行后验证输出文件数量和格式是否正确
4. mini 数据集总共约 10MB，训练一个 epoch 约 5-30 秒

## 六、注意事项

- **ACE 代码兼容性**：合成脚本会尝试调用 `ACE/ace_strategy.py` 生成真实电极图。如果 ACE 代码运行失败（缺少依赖等），脚本会自动降级为使用梅尔频谱模拟电极图，确保流程不中断
- **target 格式**：.mat 文件必须包含 `lgf` 变量，形状为 `(time_steps, 22)`，这是 `DeepACE_torch/dataset.py` 中 `load_electrodogram()` 硬编码的格式
- **stim_rate**：config.yaml 中 `stim_rate: 1000`，意味着 `block_shift = 16000/1000 = 16`，4秒音频对应 `4000/16 = 250` 帧
