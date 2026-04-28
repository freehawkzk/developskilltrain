# 模块6 数据集需求

本文档说明模块6"ASR技术简介"所需的数据集和使用方式。

---

## 一、ASR推理所需数据

### Whisper模型

本模块使用OpenAI的Whisper模型进行语音识别。Whisper是预训练模型，无需额外训练数据。

**模型选择**：

| 模型 | 参数量 | VRAM | 中文效果 | 推荐场景 |
|------|--------|------|---------|---------|
| tiny | 39M | ~1 GB | 一般 | 快速演示 |
| base | 74M | ~1 GB | 一般 | 课堂演示 |
| small | 244M | ~2 GB | 较好 | 推荐使用 |
| medium | 769M | ~5 GB | 好 | 进阶实验 |
| large-v3 | 1550M | ~10 GB | 最好 | 最佳效果 |

**推荐**：课堂使用 `small` 模型（中文效果较好，GPU要求不高）。

### 测试音频

需要以下类型的测试音频：

1. **干净中文语音**：用于评估基线识别率
   - 来源：可使用模块5生成的测试样本中的clean.wav
   - 或自行录制/下载的中文语音

2. **带噪中文语音**：不同SNR等级
   - 可使用模块5的 `scripts/prepare_test_samples.py` 生成
   - SNR等级：-5, 0, 5, 10, 15, 20 dB

3. **增强后语音**：DeepFilterNet处理后的输出
   - 从模块5的推理结果获取

4. **英文语音**（可选）：用于对比中英文识别差异

---

## 二、数据准备方式

### 方式A：使用模块5的已有数据（推荐）

```bash
# 复用模块5的测试样本
cp -r ../module5-deepfilternet/test_samples/ ./test_audio/
cp -r ../module5-deepfilternet/enhanced/ ./test_audio/enhanced/
```

### 方式B：自行生成测试数据

```python
#!/usr/bin/env python3
"""生成ASR测试数据（中文TTS或录音）"""
# 可以使用 edge-tts 或其他中文TTS工具生成测试语音
# pip install edge-tts
# edge-tts --text "今天天气很好" --voice zh-CN-XiaoxiaoNeural --file test.wav
```

### 方式C：使用公开数据集

- **Aishell**：中文语音识别标准测试集（178小时）
- **LibriSpeech**：英文语音识别标准测试集
- **Common Voice**：多语言开源语音数据集

---

## 三、课前准备

### 必须准备（课前完成）

1. **安装Whisper**
   ```bash
   pip install openai-whisper
   # 或使用faster-whisper（更快，推荐）
   pip install faster-whisper
   ```

2. **预下载模型权重**
   ```python
   import whisper
   model = whisper.load_model("small")  # 首次运行自动下载
   ```
   建议课前预先下载，避免课上等待。

3. **准备测试音频**
   - 至少准备3段干净中文语音（每段5-10秒）
   - 准备对应的不同SNR带噪版本

### 可选准备

- **安装jieba**（中文分词，用于CER计算）：`pip install jieba`
- **安装torchaudio**：用于音频加载和处理
- **下载Aishell测试集**：用于更全面的评估

---

## 四、GET声码器

第2次课的Pipeline整合需要将ACE电极图还原为可听音频，才能送入ASR评估。

### GET声码器位置

GET声码器的代码位于 `module4-deepace/ACE/` 目录：

| 文件 | 说明 |
|------|------|
| `get_voc.py` | GET声码器核心实现 |
| `voc_main.py` | 声码器演示脚本 |
| `ace_strategy.py` | ACE策略入口（声码器需要其输出的电极图和MAP参数） |

### GET声码器参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| vocoder_carrier | 1 | 1=GET(正弦载波), 2=GEN(噪声载波) |
| conv_type | 1 | 1=最大值叠加, 2=累加 |
| carrier_freq_shift | 0 | 载波频率偏移 (Hz) |
| get_fs | 16000 | 输出采样率 |
| N_BAND | 22 | 通道数 |
| N_MAXIMA | 8 | 每帧选取的最大幅度通道数 |

### 使用示例

```python
import sys
sys.path.insert(0, '../module4-deepace/ACE/')
from ace_strategy import ace_strategy
from get_voc import get_voc

# Step 1: ACE编码
q, p = ace_strategy(audio, 16000, n_band=22, n_maxima=8)

# Step 2: GET声码器还原
GET_DUR = (3 + (22 - np.arange(1, 23))).astype(float)
vocoded, mod_bands = get_voc(q, p, 1, GET_DUR, 1, 0, 16000)
# vocoded 即为还原后的音频，可送入ASR
```

---

## 五、Pipeline整合需求

第2次课需要整合前面所有模块的输出。确保以下模块可运行：

| 模块 | 需要的输出 | 状态确认 |
|------|-----------|---------|
| 模块5 DeepFilterNet | 增强后的音频文件 | `enhanced/` 目录 |
| 模块4 ACE策略 | 电极图 + MAP参数 | `ace_strategy` 可运行 |
| 模块4 GET声码器 | 还原音频 | `get_voc` 可运行 |
| 模块6 ASR | 识别文本 + WER/CER | 本模块 |

---

## 六、注意事项

- **中文识别**：Whisper对中文的识别效果取决于模型大小。tiny/base对中文效果一般，建议使用small或以上
- **GPU要求**：Whisper推理推荐GPU。CPU也可运行但较慢（small模型约3x实时）
- **音频格式**：Whisper接受wav/mp3等常见格式，会自动重采样到16kHz
- **中文评估**：中文ASR评估用CER（字符错误率）而非WER（词错误率），因为中文没有空格分词
