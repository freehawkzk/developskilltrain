# 模块6 编程练习

## 练习1：实现CER计算函数（第1次课配套）

**目标**：深入理解字符错误率的计算方法。

**任务**：从零实现CER计算，包括编辑距离算法。

**框架代码**：
```python
def levenshtein_distance(ref, hyp):
    """
    计算两个序列之间的编辑距离。
    
    参数:
        ref: 参考序列 (list)
        hyp: 假设序列 (list)
    
    返回:
        distance: 编辑距离 (int)
    """
    # TODO: 实现动态规划编辑距离
    pass

def compute_cer(reference, hypothesis):
    """
    计算中文字符错误率。
    
    参数:
        reference: 参考文本 (str)
        hypothesis: 识别结果 (str)
    
    返回:
        cer: 字符错误率 (float, 0~1)
    """
    # TODO: 按字符拆分，调用编辑距离，计算CER
    pass
```

**测试用例**：
```python
assert compute_cer("你好世界", "你好世界") == 0.0
assert compute_cer("你好世界", "你好是界") == 0.25  # 1个替换/4个字
assert compute_cer("你好世界", "你好") == 0.5       # 2个删除/4个字
assert compute_cer("你好", "你好世界") == 1.0       # 2个插入/2个字
```

---

## 练习2：Whisper识别与评估（第2次课配套）

**目标**：学会使用Whisper进行语音识别，并评估识别质量。

**任务**：
1. 使用Whisper识别至少3段音频
2. 计算每段的CER
3. 分析识别错误类型（替换、删除、插入）

**框架代码**：
```python
import whisper

def evaluate_asr(model, audio_path, reference_text):
    """
    使用Whisper识别音频并计算CER。
    
    参数:
        model: Whisper模型
        audio_path: 音频文件路径
        reference_text: 参考文本
    
    返回:
        result: dict，包含识别文本、CER、分段信息
    """
    # TODO: 识别音频
    # result = model.transcribe(audio_path, language="zh")
    
    # TODO: 计算CER
    # cer = compute_cer(reference_text, result["text"])
    
    # TODO: 分析错误类型
    pass

# 批量评估
# audio_dir = "../test_audio/"
# for f in os.listdir(audio_dir):
#     if f.endswith('.wav'):
#         result = evaluate_asr(model, os.path.join(audio_dir, f), ref_text)
#         print(f"  {f}: CER={result['cer']:.2f}")
```

**评估记录表格**：

| 音频文件 | 参考文本 | 识别结果 | CER | 主要错误类型 |
|---------|---------|---------|-----|------------|
| clean.wav | | | | |
| noisy_snr10.wav | | | | |
| noisy_snr5.wav | | | | |
| noisy_snr0.wav | | | | |
| enhanced.wav | | | | |

---

## 练习3：SNR-ASR曲线绘制（第2次课配套）

**目标**：系统评估噪声对ASR的影响。

**任务**：
1. 使用不同SNR的带噪语音进行识别
2. 绘制"SNR vs CER"曲线
3. 对比有/无语音增强的曲线

**框架代码**：
```python
def snr_asr_experiment(model, clean_path, noise_path, snr_levels):
    """
    SNR-ASR评估实验。
    
    参数:
        model: Whisper模型
        clean_path: 干净语音路径
        noise_path: 噪声路径
        snr_levels: SNR等级列表
    
    返回:
        results: dict，包含每个SNR的CER
    """
    import numpy as np
    from scipy.io import wavfile
    
    results = {}
    
    for snr in snr_levels:
        # TODO: 生成指定SNR的带噪语音
        # noisy = mix_at_snr(clean, noise, snr)
        
        # TODO: 保存临时文件
        # wavfile.write(temp_path, sr, noisy.astype(np.float32))
        
        # TODO: ASR识别并计算CER
        # result = model.transcribe(temp_path, language="zh")
        # cer = compute_cer(reference, result["text"])
        
        results[snr] = cer
        print("  SNR=%3ddB: CER=%.2f" % (snr, cer))
    
    return results

# 绘制SNR-CER曲线
# plt.plot(snrs, cers, 'o-')
# plt.xlabel('SNR (dB)')
# plt.ylabel('CER')
# plt.title('SNR vs ASR识别准确率')
```

---

## 练习4：端到端Pipeline整合（第2次课配套，从零式）

**目标**：独立完成所有模块的串联，包括GET声码器步骤。

**任务**：不使用框架代码，从零实现以下Pipeline并评估：

```
带噪语音 → DeepFilterNet增强 → ACE编码 → GET声码器还原 → ASR识别 → 计算CER
```

**要求**：
1. 自己写代码加载各模块
2. 自己处理采样率转换（DeepFilterNet输出48kHz → ACE需要16kHz → 声码器输出16kHz）
3. 正确调用GET声码器（需要ACE输出的electrodogram和map_p）
4. 自己设计对比实验（至少4种配置）
5. 自己画图展示结果

**评估矩阵**：

| Pipeline配置 | 音频1 CER | 音频2 CER | 音频3 CER | 平均CER |
|-------------|-----------|-----------|-----------|---------|
| 带噪→ASR | | | | |
| 增强→ASR | | | | |
| 增强→ACE→声码器→ASR | | | | |
| 干净→ASR | | | | |

**预期成果**：
- 一份可运行的Python脚本/notebook
- 包含至少3种Pipeline配置的对比实验
- SNR-CER曲线图（有/无增强两条线）
- 简短分析（200字）：增强对ASR识别率的影响

---

## 练习5：CI用户视角分析（思考题，不编程）

**目标**：培养从CI用户角度思考问题的能力。

思考以下问题，写一份简短分析（300字以内）：

1. ASR识别率高的语音，CI用户一定能听懂吗？试举一个反例。

2. 模块4的DeepACE将语音编码为22个通道的电极刺激。如果我们用Whisper识别ACE编码后的重建语音，CER可能很高。这意味着ACE"不好"吗？为什么？

3. 如果你要设计一个"CI专用语音增强"的研究课题，你会如何定义目标函数？（提示：不仅仅是最小化SI-SDR）
