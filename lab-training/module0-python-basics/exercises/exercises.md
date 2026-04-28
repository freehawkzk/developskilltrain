# 模块0 课后练习

## 第1次课课后练习

### 基础（必做）

完成一个音频工具函数库，包含以下三个函数：

```python
# 文件名：signal_utils.py

import numpy as np
import matplotlib.pyplot as plt

def generate_sine_wave(frequency, duration, sample_rate=16000, amplitude=0.5):
    """生成正弦波
    
    参数:
        frequency: 频率 (Hz)
        duration: 时长 (秒)
        sample_rate: 采样率 (Hz)
        amplitude: 振幅
    
    返回:
        t: 时间轴 (numpy数组)
        wave: 波形数据 (numpy数组)
    """
    # TODO: 实现这个函数
    pass

def generate_composite_wave(freqs, amps, duration, sample_rate=16000):
    """生成复合波（多个正弦波叠加）
    
    参数:
        freqs: 频率列表
        amps: 对应的振幅列表
        duration: 时长
        sample_rate: 采样率
    
    返回:
        t: 时间轴
        composite: 复合波
    """
    # TODO: 实现这个函数
    pass

def plot_waveform(t, wave, title='Waveform'):
    """画出波形
    
    参数:
        t: 时间轴
        wave: 波形数据
        title: 图表标题
    """
    # TODO: 实现这个函数
    pass
```

**验收标准：**
- 三个函数都能正确运行
- 能生成440Hz的正弦波并画图
- 能生成C大三和弦（C4+E4+G4）的复合波并画图
- 代码提交到Git仓库，commit message写清楚

### 进阶（选做）

写一个函数 `generate_chord(root_note, chord_type, duration, sample_rate=16000)`，它：
- 输入根音（如'C4'）和和弦类型（如'major', 'minor'）
- 输出对应的和弦波形
- 提示：需要一个音符名到频率的映射表

参考频率：
- C4=261.63, D4=293.66, E4=329.63, F4=349.23, G4=392.00, A4=440.00, B4=493.88

---

## 第2次课课后练习

### 基础（必做）

1. 完善 `Signal` 类，添加以下方法：
   - `duration` 属性：返回信号的时长（秒）
   - `num_samples` 属性：返回样本数
   - `resample(new_sr)` 方法：重采样到新的采样率

2. 编写 `AudioFileDataset` 类，继承 `torch.utils.data.Dataset`：
   - `__init__`：接收一个音频文件目录路径
   - `__len__`：返回文件数量
   - `__getitem__`：加载指定索引的音频文件并返回 (waveform, filename)

**验收标准：**
- Signal 类的所有方法都能正确运行
- AudioFileDataset 类能通过 `len()` 和索引访问
- 代码提交到Git仓库

### 进阶（选做）

用面向对象的方式重写第1次课的练习：
- `Signal` 是基类
- `SineWave` 继承 `Signal`
- `CompositeWave` 继承 `Signal`（可以包含多个 `SineWave`）
- 所有类都有 `plot()` 和 `compute_spectrum()` 方法

---

## 第3次课课后练习

### 基础（必做）

1. 将前两次课的代码重构为模块化的项目：
   ```
   audio_tools/
   ├── __init__.py
   ├── signal.py          # Signal类
   ├── generators.py      # SineWave, CompositeWave, NoisySignal
   └── visualization.py   # 画图工具函数
   ```

2. 在一个新的notebook中用import引用自己的模块：
   ```python
   from audio_tools.signal import Signal
   from audio_tools.generators import SineWave, NoisySignal
   ```

3. 用Git提交，至少3个commit：
   - "初始化项目结构"
   - "实现Signal和SineWave类"
   - "完成模块化重构"

**验收标准：**
- 目录结构正确
- import能够正常工作
- Git历史有3个以上的commit

### 进阶（选做）

1. 为 `SineWave` 类添加一个 `to_wav(filepath)` 方法，使用 `soundfile` 库将波形保存为wav文件
2. 为 `Signal` 类添加 `__repr__` 方法，打印信号的基本信息
3. 写一个简单的单元测试：验证 `SineWave(440).frequency == 440`

---

## 编程能力诊断测试

完成模块0后，请用30分钟独立完成以下任务（不看答案）：

**任务**：写一个 `HarmonicWave` 类，继承 `Signal`，生成包含基频和谐波的复合波。

要求：
1. 构造函数接收 `fundamental_freq`（基频）、`num_harmonics`（谐波数量，默认5）、`amplitudes`（各谐波的振幅列表，默认等比递减）
2. 实现 `get_harmonic_freqs()` 方法，返回所有谐波频率的列表
3. 实现 `plot_spectrum()` 方法，画出频谱（用 `compute_spectrum()` 的结果）
4. 每个方法都要有docstring
5. 代码格式规范，变量名有意义

```python
# 参考接口
class HarmonicWave(Signal):
    def __init__(self, fundamental_freq, num_harmonics=5, ...):
        ...
    
    def get_harmonic_freqs(self):
        ...
    
    def plot_spectrum(self):
        ...
```

**自评标准：**
- 30分钟内能独立完成：编程能力良好，可以免修模块0的后两次课
- 需要提示才能完成：编程基础需要加强，建议参加全部课程
- 无法完成：编程基础薄弱，模块0必须参加
