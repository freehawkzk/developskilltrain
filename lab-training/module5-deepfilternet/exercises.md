# 模块5 编程练习

## 练习1：实现简化版谱减法（第1次课配套）

**目标**：理解谱减法的原理和局限。

**任务**：在STFT域实现谱减法，要求：
1. 将输入信号分帧，每帧做STFT
2. 在噪声段（前0.5秒）估计噪声功率谱
3. 从带噪谱中减去噪声功率谱，半波整流
4. 用原始相位重建时域信号

**框架代码**：
```python
import numpy as np
from scipy import signal as sig

def spectral_subtraction(noisy, sr, noise_frames=5):
    """
    简化版谱减法。
    
    参数:
        noisy: 带噪语音信号 (1D numpy array)
        sr: 采样率
        noise_frames: 用于估计噪声的帧数
    
    返回:
        enhanced: 增强后的信号
    """
    # TODO: 设置STFT参数
    # 提示: nperseg=480, noverlap=360
    
    # TODO: 计算STFT
    # f, t, Zxx = sig.stft(...)
    
    # TODO: 估计噪声功率谱（取前noise_frames帧的平均）
    # noise_psd = ...
    
    # TODO: 谱减法（减去噪声功率谱，半波整流）
    # enhanced_mag = ...
    
    # TODO: 用原始相位重建
    # enhanced_stft = ...
    
    # TODO: ISTFT重建时域信号
    # _, enhanced = sig.istft(...)
    
    return enhanced
```

**验证**：用notebook中的合成信号测试，计算SI-SDR。

**预期结果**：谱减法能去除部分噪声，但会产生"音乐噪声"（频谱上的随机亮点）。

---

## 练习2：实现SI-SDR计算函数（第1次课配套）

**目标**：深入理解SI-SDR指标。

**任务**：从头实现SI-SDR，不使用现成库。

**参考公式**：
```
e_target = <s_hat, s> / <s, s> * s
e_noise = s_hat - e_target
SI-SDR = 10 * log10(<e_target, e_target> / <e_noise, e_noise>)
```

**框架代码**：
```python
def compute_si_sdr(estimated, reference):
    """
    计算Scale-Invariant SDR。
    
    参数:
        estimated: 估计信号 (1D numpy array)
        reference: 参考信号 (1D numpy array)
    
    返回:
        si_sdr: SI-SDR值 (dB)
    """
    # TODO: 去均值
    # reference = reference - np.mean(reference)
    # estimated = estimated - np.mean(estimated)
    
    # TODO: 计算最优缩放因子 alpha
    # alpha = np.dot(estimated, reference) / (np.dot(reference, reference) + 1e-8)
    
    # TODO: 计算e_target和e_noise
    # s_target = alpha * reference
    # e_noise = estimated - s_target
    
    # TODO: 计算SI-SDR
    # si_sdr = 10 * np.log10(...)
    
    return si_sdr
```

**测试用例**：
```python
# 测试1: 完全匹配
ref = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000))
assert compute_si_sdr(ref, ref) > 50  # 应该非常大

# 测试2: 缩放不变性
assert abs(compute_si_sdr(2 * ref, ref) - compute_si_sdr(ref, ref)) < 0.01

# 测试3: 纯噪声
noise = np.random.randn(len(ref)) * 0.1
assert compute_si_sdr(noise, ref) < 0  # 应该是负值
```

---

## 练习3：修改ERB频带数量（第3次课配套）

**目标**：理解ERB频带数对模型性能和CI适配的影响。

**任务**：
1. 修改 `DfParams` 中的 `NB_ERB` 参数（在 `config.py` 中）
2. 用修改后的模型运行推理
3. 对比不同频带数下的增强效果

**实验步骤**：

```bash
# Step 1: 备份原始配置
cp DeepFilterNet-main/DeepFilterNet/df/config.py config_backup.py

# Step 2: 修改NB_ERB
# 在 config.py 中将 NB_ERB = 32 改为以下值依次测试
# - NB_ERB = 22  (匹配典型CI电极数)
# - NB_ERB = 16  (更少频带)
# - NB_ERB = 48  (更多频带)
```

**记录表格**：

| NB_ERB | SI-SDR (dB) | 推理时间 (ms/秒音频) | 模型参数量 | 备注 |
|--------|-------------|---------------------|-----------|------|
| 32 (原始) | | | | |
| 22 | | | | |
| 16 | | | | |
| 48 | | | | |

**思考**：为什么减少频带可能不影响甚至改善CI场景的效果？

---

## 练习4：实现Enhance-then-Encode流水线（第3次课配套）

**目标**：将语音增强与ACE编码串联。

**任务**：将DeepFilterNet的增强输出作为ACE的输入，评估增强对编码效果的影响。

**框架代码**：
```python
import torch
import sys
sys.path.append('../module4-deepace/')

from deepace import CISessionConfig, ACESession, ACEOutput
from df.enhance import init_df, enhance

def enhance_then_encode(noisy_path, config=None):
    """
    Enhance-then-Encode 流水线。
    
    参数:
        noisy_path: 带噪语音文件路径
        config: CI配置（如果None则使用默认配置）
    
    返回:
        result: dict，包含各阶段输出和评估指标
    """
    if config is None:
        config = CISessionConfig(n_channels=22, fs=16000)
    
    # TODO: Step 1 - DeepFilterNet增强
    # model, df_state, _ = init_df()
    # enhanced, _ = enhance(model, df_state, noisy_path)
    
    # TODO: Step 2 - 重采样到16kHz（如果需要）
    # DeepFilterNet输出48kHz，ACE需要16kHz
    # import torchaudio.transforms as T
    # resampler = T.Resample(48000, 16000)
    # enhanced_16k = resampler(torch.tensor(enhanced))
    
    # TODO: Step 3 - ACE编码
    # session = ACESession(config)
    # ace_output = session.process(enhanced_16k.numpy())
    
    # TODO: Step 4 - 计算评估指标
    # - 对比有/无增强的通道选择模式
    # - 计算SI-SDR
    
    return result

# 对比实验
# result_no_enhance = encode_only(noisy_path)
# result_with_enhance = enhance_then_encode(noisy_path)
```

**评估指标**：
- 通道选择的稳定性（帧间通道切换频率）
- 增强前后的SI-SDR
- 电极激活模式的可懂度预测

---

## 练习5：SNR扫描评估（第3次课配套）

**目标**：系统评估DeepFilterNet在不同信噪比下的表现。

**任务**：用 `scripts/prepare_test_samples.py` 生成不同SNR的测试样本，运行推理并计算指标。

```python
import numpy as np
from pesq import pesq  # pip install pesq
from pystoi import stoi  # pip install pystoi

def evaluate_enhancement(clean, enhanced, noisy, sr):
    """
    计算完整评估指标。
    
    返回:
        dict: {pesq, stoi, si_sdr}
    """
    # PESQ: 宽带模式
    pesq_score = pesq(sr, clean, enhanced, 'wb')
    
    # STOI
    stoi_score = stoi(clean, enhanced, sr, extended=False)
    
    # SI-SDR
    si_sdr = compute_si_sdr(enhanced, clean)
    
    return {
        'pesq': pesq_score,
        'stoi': stoi_score,
        'si_sdr': si_sdr
    }

# 扫描不同SNR
snr_levels = [-5, 0, 5, 10, 15, 20]
results = {}
for snr in snr_levels:
    # TODO: 运行推理并评估
    pass

# 画图：x轴为SNR，y轴为各指标
```

**预期输出**：三张图（PESQ/STOI/SI-SDR vs SNR），每张图包含"带噪-干净"和"增强-干净"两条曲线。
