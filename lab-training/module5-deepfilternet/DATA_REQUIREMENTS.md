# 模块5 数据集需求

本文档说明模块5"DeepFilterNet模型解析"所需的数据集和使用方式。

---

## 一、DeepFilterNet 数据格式说明

DeepFilterNet 的推理（增强）只需要：
- **输入**：带噪语音 WAV 文件，任意采样率（代码会自动重采样到48kHz）
- **输出**：增强后的干净语音 WAV 文件

训练数据格式（供参考，本模块不需要训练）：
- **语音源**：干净语音，存储为 HDF5 格式
- **噪声源**：各类环境噪声，存储为 HDF5 格式
- **训练时动态混合**：训练过程中实时生成 (clean + noise → noisy) 配对

### 关键配置参数（DfParams 默认值）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| SR | 48000 | 采样率 (Hz) |
| FFT_SIZE | 960 | FFT 窗口大小 |
| HOP_SIZE | 480 | STFT 步长 |
| NB_ERB | 32 | ERB 频带数 |
| NB_DF | 96 | Deep Filtering 频率 bin 数 |
| DF_ORDER | 5 | Deep Filtering 阶数 |

---

## 二、已包含的素材

DeepFilterNet 开源代码仓库 (`DeepFilterNet-main/`) 已包含以下可直接使用的素材：

### 1. 预训练模型权重

位于 `DeepFilterNet-main/models/` 目录：

| 文件 | 模型版本 | 说明 |
|------|---------|------|
| DeepFilterNet3.zip | DeepFilterNet3 | 最新版，推荐使用 |
| DeepFilterNet2.zip | DeepFilterNet2 | 第二版 |
| DeepFilterNet.zip | DeepFilterNet | 第一版 |

使用前需解压：
```bash
cd DeepFilterNet-main/models/
unzip DeepFilterNet3.zip
```

### 2. 示例音频

位于 `DeepFilterNet-main/assets/` 目录：

| 文件 | 说明 |
|------|------|
| clean_freesound_33711.wav | 干净语音样本 |
| noise_freesound_2530.wav | 噪声样本 |
| noise_freesound_573577.wav | 噪声样本 |
| noisy_snr0.wav | 0dB SNR 带噪语音示例 |
| rir_sim_*.wav | 房间脉冲响应（用于混响模拟） |

### 3. 数据处理脚本

- `assets/dataset.cfg` — 数据集配置文件
- `DeepFilterNet/df/scripts/prepare_data.py` — 训练数据准备脚本
- `scripts/download_process_dns4.sh` — DNS Challenge 数据集下载脚本

---

## 三、课堂演示数据需求

本模块的教学重点在推理和评估，不需要完整训练数据。课堂演示需要：

### 必须准备（课前完成）

1. **解压预训练模型**
   ```bash
   cd DeepFilterNet-main/models/
   unzip DeepFilterNet3.zip
   ```

2. **安装 DeepFilterNet Python 包**
   ```bash
   cd DeepFilterNet-main/DeepFilterNet/
   pip install -e .
   ```
   注意：安装需要 Rust 编译环境和 `libdf` 库。如果 Rust 不可用，可以使用预编译的 wheel 或直接使用 `deep-filter` pip 包：
   ```bash
   pip install deep-filter
   ```

3. **准备不同 SNR 的测试音频**（3-5段）
   - 可使用 assets 中的音频文件
   - 或使用 `scripts/prepare_test_samples.py`（见下方）生成

### 可选准备

- **VoiceBank-DEMAND 数据集**：标准语音增强测试集，包含预配对的 (clean, noisy) 音频
  - 下载：https://datashare.ed.ac.uk/handle/10283/2791
  - 用于评估 DeepFilterNet 在标准 benchmark 上的表现

---

## 四、测试样本生成脚本

以下脚本自动从 assets 中的音频生成不同 SNR 的测试样本：

### 脚本：`scripts/prepare_test_samples.py`

```python
#!/usr/bin/env python3
"""
生成不同 SNR 的语音增强测试样本。
使用 DeepFilterNet assets 目录中的音频文件。
"""
import numpy as np
import soundfile as sf
import os

SR = 48000
SNR_LEVELS = [-5, 0, 5, 10, 15, 20]
DURATION = 5.0  # 秒

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODULE5_DIR = os.path.dirname(SCRIPT_DIR)
ASSETS_DIR = os.path.join(MODULE5_DIR, "DeepFilterNet-main", "assets")
OUTPUT_DIR = os.path.join(MODULE5_DIR, "test_samples")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_audio(path, sr=SR, duration=DURATION):
    """加载音频并截取指定时长"""
    audio, orig_sr = sf.read(path)
    if len(audio.shape) > 1:
        audio = audio[:, 0]
    if orig_sr != sr:
        # 简单重采样
        import librosa
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr)
    target_len = int(sr * duration)
    if len(audio) > target_len:
        audio = audio[:target_len]
    elif len(audio) < target_len:
        audio = np.tile(audio, (target_len // len(audio)) + 1)[:target_len]
    return audio

def mix_at_snr(clean, noise, snr_db):
    """按指定 SNR 混合"""
    clean_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power < 1e-10:
        return clean.copy()
    snr_linear = 10 ** (snr_db / 10)
    noise_scaled = noise * np.sqrt(clean_power / (noise_power * snr_linear + 1e-8))
    mixture = clean + noise_scaled
    mixture = mixture / (np.max(np.abs(mixture)) + 1e-8) * 0.9
    return mixture

def main():
    # 加载音频
    clean_path = os.path.join(ASSETS_DIR, "clean_freesound_33711.wav")
    noise_path = os.path.join(ASSETS_DIR, "noise_freesound_573577.wav")

    if not os.path.exists(clean_path):
        print("错误：找不到 assets/clean_freesound_33711.wav")
        print("请确认 DeepFilterNet-main 已完整下载")
        return

    clean = load_audio(clean_path)
    noise = load_audio(noise_path)

    # 保存干净语音
    sf.write(os.path.join(OUTPUT_DIR, "clean.wav"), clean, SR)
    print("已保存: clean.wav")

    # 生成不同 SNR 的混合
    for snr in SNR_LEVELS:
        mixture = mix_at_snr(clean, noise, snr)
        fname = "noisy_snr%d.wav" % snr
        sf.write(os.path.join(OUTPUT_DIR, fname), mixture.astype(np.float32), SR)
        print("已保存: %s" % fname)

    print("\n完成！测试样本保存在: %s" % OUTPUT_DIR)
    print("可运行推理: cd DeepFilterNet-main && python -m df.enhance %s" % OUTPUT_DIR)

if __name__ == "__main__":
    main()
```

---

## 五、快速启动步骤

```bash
# 1. 解压预训练模型
cd module5-deepfilternet/DeepFilterNet-main/models/
unzip DeepFilterNet3.zip

# 2. 安装 DeepFilterNet（推荐使用 pip 预编译版）
pip install deep-filter

# 3. 生成测试样本
cd module5-deepfilternet
python scripts/prepare_test_samples.py

# 4. 运行推理
cd DeepFilterNet-main
python -m df.enhance ../test_samples/ -o ../enhanced/

# 5. 对比增强前后
# 使用 notebook 中的评估代码计算 PESQ/STOI/SI-SDR
```

---

## 六、注意事项

- **Rust 编译依赖**：DeepFilterNet 的核心数据处理（STFT、ERB等）使用 Rust 实现。如果从源码安装需要 Rust 工具链。使用 `pip install deep-filter` 可跳过 Rust 编译
- **采样率**：DeepFilterNet 默认 48kHz，与 DeepACE 的 16kHz 不同。推理时会自动重采样
- **GPU 内存**：推理可在 CPU 上实时运行（~2% CPU），训练需要 GPU
- **评估指标工具**：PESQ 需要 `pesq` 包（`pip install pesq`），STOI 已包含在 DeepFilterNet 代码中
