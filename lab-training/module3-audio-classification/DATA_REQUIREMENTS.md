# 模块3 数据集需求

本文档列出模块3"面向声音的分类模型"所需的全部数据集。公开数据集需手动下载，自建数据集需按说明准备。

---

## 一、公开数据集

### 1. ESC-50：环境声分类数据集

**用途：** 第1-2次课的核心实验数据，用于音频特征对比和CRNN分类实验。

**内容：** 50类环境声，每类40条录音，共2000条，每条5秒，44.1kHz单声道。

**下载地址：**
- GitHub Release: https://github.com/karoldvl/ESC-50/archive/master.zip
- 备用（Zenodo）: https://zenodo.org/record/3989107/files/ESC-50.zip

**存放路径：**
```
module3-audio-classification/data/ESC-50/
├── meta/
│   └── esc50.csv          # 标签文件
└── audio/
    └── 1-xxxx-A-xx.wav    # 2000个wav文件
```

**验证：** 下载后解压，确认 `meta/esc50.csv` 存在且有2000行数据记录，`audio/` 目录下有2000个wav文件。

---

### 2. Speech Commands：语音命令识别数据集

**用途：** 第2次课的补充实验（如课堂时间允许），用于对比环境声分类与语音命令分类的差异。第3次课VAD实验的基础数据来源——从中抽取语音片段作为正样本。

**内容：** 35类语音命令（如"yes""no""up""down"等），每类数千条，约105000条1秒录音，16kHz单声道。

**下载地址：**
- 官方：https://storage.cloud.google.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz
- 备用（TensorFlow Datasets）: https://www.tensorflow.org/datasets/catalog/speech_commands

**存放路径：**
```
module3-audio-classification/data/speech_commands/
├── _background_noise_/
│   └── ...                # 背景噪声文件
├── yes/
│   └── *.wav
├── no/
│   └── *.wav
├── up/
│   └── *.wav
├── ...                    # 其余32类目录
├── TESTING_LIST.txt
├── VALIDATION_LIST.txt
└── TRAINING_LIST.txt
```

**验证：** 确认有35个子目录 + 1个 `_background_noise_` 目录，`TRAINING_LIST.txt` 等划分文件存在。

---

## 二、自建数据集

### 1. 语音/噪声二分类数据集（speech_noise_dataset）

**用途：** 第3次课VAD模型训练、模块2主线项目延伸。此数据集是课程主线项目的基础——语音增强前的"检测"环节。

#### 数据要求

| 项目 | 要求 |
|------|------|
| 语音样本 | 至少2000条，来自不同说话人（男女各半），不同内容，不同信噪比 |
| 噪声样本 | 至少2000条，涵盖白噪声、粉红噪声、环境噪声（街道、餐厅、风声等） |
| 格式 | WAV，16kHz，单声道 |
| 时长 | 每条0.5-2秒 |
| 标签 | `speech`（语音）或 `noise`（噪声） |

#### 建设方式

**方案A：从现有公开数据集组装（推荐，最省力）**

1. **语音部分**：从 Speech Commands 数据集中抽取
   ```bash
   # 随机抽取2000条语音命令文件
   # Speech Commands 中每个 .wav 都是人说的短词，天然是"语音"样本
   find speech_commands/ -name "*.wav" -not -path "*/_background_noise_*" | shuf | head -2000
   ```

2. **噪声部分**：从以下来源获取
   - Speech Commands 中的 `_background_noise_/` 目录（约7条长噪声，需切片）
   - ESC-50 中标注为噪声的类别（如"chainsaw""drilling""engine"等）
   - NOISEX 数据集（经典噪声库）：http://staff.ustc.edu.cn/~yln/nosieex/noisex.zip

3. **组装脚本**（下载完原始数据后运行）：
   ```python
   # prepare_speech_noise.py — 自动组装语音/噪声数据集
   import os, glob, random, shutil
   import numpy as np
   import soundfile as sf

   SPEECH_COMMANDS_DIR = "../speech_commands/"
   ESC50_DIR = "../ESC-50/audio/"
   NOISEX_DIR = "../noisex/"  # 可选
   OUTPUT_DIR = "./speech_noise_dataset/"
   
   TARGET_SR = 16000
   CLIP_DURATION = 1.0  # 秒
   N_SAMPLES_PER_CLASS = 2000

   def load_and_resample(path, sr=16000):
       """加载音频并重采样到目标采样率"""
       data, orig_sr = sf.read(path)
       if orig_sr != sr:
           import librosa
           data = librosa.resample(data, orig_sr=orig_sr, target_sr=sr)
       if data.ndim > 1:
           data = data.mean(axis=1)  # 转单声道
       return data, sr

   def split_long_audio(path, clip_duration=1.0, sr=16000):
       """将长音频切割成固定时长片段"""
       data, _ = load_and_resample(path, sr)
       clip_len = int(clip_duration * sr)
       clips = []
       for start in range(0, len(data) - clip_len, clip_len):
           clips.append(data[start:start + clip_len])
       return clips

   def main():
       os.makedirs(f"{OUTPUT_DIR}/speech", exist_ok=True)
       os.makedirs(f"{OUTPUT_DIR}/noise", exist_ok=True)

       # === 语音样本 ===
       speech_files = glob.glob(f"{SPEECH_COMMANDS_DIR}/*/*.wav")
       speech_files = [f for f in speech_files if "_background_noise_" not in f]
       random.shuffle(speech_files)
       
       count = 0
       for f in speech_files[:N_SAMPLES_PER_CLASS]:
           data, sr = load_and_resample(f)
           out_path = f"{OUTPUT_DIR}/speech/speech_{count:04d}.wav"
           sf.write(out_path, data, sr)
           count += 1
       print(f"语音样本: {count} 条")

       # === 噪声样本 ===
       noise_clips = []
       
       # 来源1: Speech Commands 的背景噪声
       bg_noise_files = glob.glob(f"{SPEECH_COMMANDS_DIR}/_background_noise_/*.wav")
       for f in bg_noise_files:
           clips = split_long_audio(f)
           noise_clips.extend(clips)
       
       # 来源2: ESC-50 中的噪声类
       # ESC-50噪声类: chainsaw, drilling, engine, jackhammer, siren, car_horn 等
       noise_esc50_labels = [1, 6, 10, 11, 18, 19, 20, 36, 37, 38, 39, 40]
       # 需要读取 esc50.csv 获取对应文件
       import csv
       with open(f"../ESC-50/meta/esc50.csv") as f:
           reader = csv.DictReader(f)
           for row in reader:
               if int(row['target']) in noise_esc50_labels:
                   path = f"../ESC-50/audio/{row['filename']}"
                   clips = split_long_audio(path)
                   noise_clips.extend(clips)
       
       # 来源3: NOISEX（如果有的话）
       if os.path.exists(NOISEX_DIR):
           noisex_files = glob.glob(f"{NOISEX_DIR}/*.wav")
           for f in noisex_files:
               clips = split_long_audio(f)
               noise_clips.extend(clips)
       
       random.shuffle(noise_clips)
       noise_clips = noise_clips[:N_SAMPLES_PER_CLASS]
       
       for i, clip in enumerate(noise_clips):
           out_path = f"{OUTPUT_DIR}/noise/noise_{i:04d}.wav"
           sf.write(out_path, clip, TARGET_SR)
       print(f"噪声样本: {len(noise_clips)} 条")

   if __name__ == "__main__":
       main()
   ```

**方案B：完全自建**

如果实验室有录音条件，可以：
1. 录制多位说话人的语音（朗读短句、自由对话），切片为0.5-2秒片段
2. 在不同环境录制噪声（安静的办公室、嘈杂的走廊、室外等）
3. 格式统一为16kHz单声道WAV

---

### 2. 带噪语音可懂度分类数据集（intelligibility_dataset）

**用途：** 第3次课实战2——带噪语音可懂度分类。输入一段带噪语音，模型判断"CI用户能否听懂"。

#### 数据要求

| 项目 | 要求 |
|------|------|
| 正样本（可懂） | 带噪语音，SNR较高（如 +5dB, +10dB, +15dB），语音内容仍可辨认 |
| 负样本（不可懂） | 带噪语音，SNR较低（如 -5dB, -10dB），语音内容被噪声严重遮蔽 |
| 格式 | WAV，16kHz，单声道 |
| 时长 | 每条1-3秒 |
| 标签 | `intelligible`（可懂）或 `unintelligible`（不可懂） |
| 数量 | 每类至少500条 |

#### 建设方式

**推荐：从干净语音 + 噪声合成（自动生成，无需人工录制）**

1. 准备干净语音：从 Speech Commands 中抽取1000条语音文件
2. 准备噪声源：从上述噪声数据集或 NOISEX 中获取噪声
3. 在不同 SNR 下合成带噪语音：
   ```python
   # prepare_intelligibility.py — 合成带噪语音可懂度数据集
   import os, glob, random
   import numpy as np
   import soundfile as sf

   SPEECH_DIR = "../speech_commands/"      # 干净语音源
   NOISE_DIR = "../speech_noise_dataset/noise/"  # 噪声源
   OUTPUT_DIR = "./intelligibility_dataset/"
   TARGET_SR = 16000

   # SNR设置：正值→可懂，负值→不可懂
   INTELLIGIBLE_SNRS = [5, 10, 15]        # dB
   UNINTELLIGIBLE_SNRS = [-10, -5]        # dB

   def load_audio(path, sr=16000):
       data, orig_sr = sf.read(path)
       if orig_sr != sr:
           import librosa
           data = librosa.resample(data, orig_sr=orig_sr, target_sr=sr)
       if data.ndim > 1:
           data = data.mean(axis=1)
       return data

   def add_noise_at_snr(speech, noise, snr_db):
       """在指定SNR下将噪声加到语音上"""
       # 裁剪或循环噪声使其与语音等长
       if len(noise) < len(speech):
           noise = np.tile(noise, (len(speech) // len(noise)) + 1)[:len(speech)]
       else:
           noise = noise[:len(speech)]
       
       speech_power = np.mean(speech ** 2)
       noise_power = np.mean(noise ** 2)
       
       # 计算缩放因子
       snr_linear = 10 ** (snr_db / 10)
       noise_scaled = noise * np.sqrt(speech_power / (noise_power * snr_linear + 1e-8))
       
       return speech + noise_scaled

   def main():
       os.makedirs(f"{OUTPUT_DIR}/intelligible", exist_ok=True)
       os.makedirs(f"{OUTPUT_DIR}/unintelligible", exist_ok=True)

       # 收集语音和噪声文件
       speech_files = glob.glob(f"{SPEECH_DIR}/*/*.wav")
       speech_files = [f for f in speech_files if "_background_noise_" not in f]
       noise_files = glob.glob(f"{NOISE_DIR}/*.wav")
       
       random.shuffle(speech_files)
       random.shuffle(noise_files)

       # === 可懂样本 ===
       count = 0
       for speech_path in speech_files[:500]:
           speech = load_audio(speech_path)
           snr = random.choice(INTELLIGIBLE_SNRS)
           noise_path = random.choice(noise_files)
           noise = load_audio(noise_path)
           
           noisy = add_noise_at_snr(speech, noise, snr)
           # 归一化防止削波
           noisy = noisy / (np.max(np.abs(noisy)) + 1e-8) * 0.9
           
           out_path = f"{OUTPUT_DIR}/intelligible/intel_{count:04d}.wav"
           sf.write(out_path, noisy.astype(np.float32), TARGET_SR)
           count += 1
       print(f"可懂样本: {count} 条")

       # === 不可懂样本 ===
       count = 0
       for speech_path in speech_files[500:1000]:
           speech = load_audio(speech_path)
           snr = random.choice(UNINTELLIGIBLE_SNRS)
           noise_path = random.choice(noise_files)
           noise = load_audio(noise_path)
           
           noisy = add_noise_at_snr(speech, noise, snr)
           noisy = noisy / (np.max(np.abs(noisy)) + 1e-8) * 0.9
           
           out_path = f"{OUTPUT_DIR}/unintelligible/unintel_{count:04d}.wav"
           sf.write(out_path, noisy.astype(np.float32), TARGET_SR)
           count += 1
       print(f"不可懂样本: {count} 条")

   if __name__ == "__main__":
       main()
   ```

#### 可懂度阈值说明

"可懂/不可懂"的划分基于 SNR 阈值，这是CI研究中的常见做法：
- SNR ≥ +5dB：大多数听者（包括CI用户）可以理解语音内容 → 标记为"可懂"
- SNR ≤ -5dB：即使是正常听者也难以理解 → 标记为"不可懂"
- SNR 在 -5dB 到 +5dB 之间的样本**故意不包含**，避免边界模糊

如果实验室有CI可懂度评分数据，可以替代自动合成版本，效果更好。

---

## 三、数据集最终目录结构

下载和准备完成后，`data/` 目录应如下：

```
module3-audio-classification/data/
├── ESC-50/
│   ├── meta/
│   │   └── esc50.csv
│   └── audio/
│       └── *.wav              # 2000个文件
├── speech_commands/
│   ├── _background_noise_/
│   ├── yes/
│   ├── no/
│   ├── ...                    # 35个命令目录
│   ├── TESTING_LIST.txt
│   ├── VALIDATION_LIST.txt
│   └── TRAINING_LIST.txt
├── speech_noise_dataset/       # 自建（方案A或B）
│   ├── speech/
│   │   └── speech_*.wav
│   └── noise/
│       └── noise_*.wav
└── intelligibility_dataset/    # 自建（自动合成）
    ├── intelligible/
    │   └── intel_*.wav
    └── unintelligible/
        └── unintel_*.wav
```

---

## 四、准备顺序建议

1. **先下载 ESC-50** — 最小（约60MB），下载快，模块3-1和3-2的课堂实验都需要它
2. **再下载 Speech Commands** — 较大（约2.3GB），但VAD和可懂度数据集都依赖它作为语音源
3. **运行 `prepare_speech_noise.py`** — 生成语音/噪声二分类数据集（需要ESC-50和Speech Commands已就位）
4. **运行 `prepare_intelligibility.py`** — 生成带噪语音可懂度数据集（需要上一步的噪声数据集已就位）

全部数据准备完成后总大小约3-4GB。
