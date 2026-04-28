# 模块5：DeepFilterNet模型解析

## 课程安排

| 课次 | 主题 | 课时 | 必修 |
|------|------|------|------|
| 第1次 | 语音增强基础 + DeepFilterNet论文精读 | 2.5h | 是 |
| 第2次 | DeepFilterNet代码解析与运行 | 2.5h | 是 |
| 第3次 | DeepFilterNet修改实验 + 与CI结合 | 2.5h | 可选 |

## 学习目标

完成本模块后，学生应能够：
- 理解语音增强问题的定义和评估指标（PESQ、STOI、SI-SDR）
- 理解DeepFilterNet的核心设计思想（两阶段、ERB域）
- 运行DeepFilterNet推理并评估增强效果
- 将DeepFilterNet与ACE策略结合
- （进阶）分析DeepFilterNet在CI/助听器场景下的适用性和局限

## 编程练习设计模式

本模块采用**组合式**练习——将DeepFilterNet与之前模块的组件组合。

---

## 第1次课：语音增强基础 + DeepFilterNet论文精读（2.5小时）

### 教学目标

理解语音增强问题的定义和评估指标，理解DeepFilterNet的核心设计思想。

### 内容与节奏

| 时间 | 内容 | 形式 | 要点 |
|------|------|------|------|
| 0:00-0:20 | 语音增强问题定义：从带噪语音中恢复干净语音 | 讲授 | 用CI场景引入："CI用户在噪声中理解语音的困难，正是语音增强要解决的核心问题" |
| 0:20-0:45 | 语音增强评估指标：PESQ、STOI、SI-SDR | 讲授 | 每个指标都要讲"它衡量什么、它的局限是什么" |
| 0:45-1:10 | 传统语音增强方法回顾：谱减法、维纳滤波 | 讲授 | "DeepFilterNet的'Deep Filter'是对传统滤波方法的深度学习化" |
| 1:10-1:20 | 休息 | — | — |
| 1:20-1:50 | DeepFilterNet论文精读：架构设计 | 共读 | 重点：为什么用两个阶段（Enhancement + Deep Filtering）？为什么用ERB域？ |
| 1:50-2:20 | DeepFilterNet论文精读：训练策略与实验 | 共读 | 训练数据构造方式（DNS Challenge数据集）、损失函数设计、与baseline的对比 |
| 2:20-2:30 | 讨论：DeepFilterNet与CI/助听器的关系 | 互动 | "如果要把DeepFilterNet用在助听器上，需要解决什么问题？" |

### 关键设计

DeepFilterNet中选择ERB（Equivalent Rectangular Bandwidth）域而非普通的梅尔域，这个选择与听觉感知直接相关。ERB刻度模拟了人耳的频率分辨率，跟CI研究中的频率-电极映射有内在联系。要让学生理解这个设计选择不仅是工程上的，更是听觉科学上的。

### 配套Notebook

- `notebooks/01-se-enhancement-basics.ipynb`

### 课前准备

- DeepFilterNet论文PDF提前分发
- 预习文档：`预习-第1次课.md`

---

## 第2次课：DeepFilterNet代码解析与运行（2.5小时）

### 教学目标

能读懂DeepFilterNet的代码结构，能运行推理，理解数据流。

### 内容与节奏

| 时间 | 内容 | 形式 | 要点 |
|------|------|------|------|
| 0:00-0:15 | 论文疑问回顾 | 互动 | — |
| 0:15-0:40 | 代码结构总览 + 与DeepACE的对比 | 讲授 | 两个模型放在一起对比架构，让学生看到encoder-decoder模式的通用性 |
| 0:40-1:10 | 核心模块解析：ERB特征提取→Encoder→Deep Filtering→Decoder | 讲授+代码 | 再次画数据流图，标注张量形状变化 |
| 1:10-1:20 | 休息 | — | — |
| 1:20-1:50 | 运行推理：用预训练模型处理带噪语音 | 实操 | 提供几段不同SNR的带噪语音，让学生听处理前后的差异 |
| 1:50-2:20 | 输出分析：语谱图对比 + 客观指标评估（PESQ/STOI/SI-SDR） | 实操+讨论 | 主线项目语音增强模块 |
| 2:20-2:30 | 讨论：增强后的语音听起来如何？客观指标和主观听感是否一致？ | 互动 | 引出主观评价的重要性——这对CI研究尤其关键 |

### 关键设计

代码解析环节，最重要的不是逐行读懂每一行，而是让学生建立"数据流"的心智模型——"一段音频输入进去，经过了哪些变换，最后输出了什么"。建议画一个详细的数据流图，标注每一步的张量形状变化。

### 配套Notebook

- `notebooks/02-code-analysis.ipynb`

### 课前准备

- 安装DeepFilterNet（`pip install deep-filter`）
- 解压预训练模型（`models/DeepFilterNet3.zip`）
- 预习文档：`预习-第2次课.md`

---

## 第3次课（可选）：DeepFilterNet修改实验 + 与CI结合（2.5小时）

### 教学目标

能将DeepFilterNet与CI处理流程结合，为研究工作做准备。

### 内容与节奏

| 时间 | 内容 | 形式 | 要点 |
|------|------|------|------|
| 0:00-0:15 | 回顾 + 疑问 | 互动 | — |
| 0:15-0:45 | 实验1：修改ERB频带数，观察对不同频率区域的影响 | 边讲边练 | 对接CI的电极数映射 |
| 0:45-1:15 | 实验2：将DeepFilterNet的输出送入ACE策略 | 边讲边练 | 主线项目的关键连接——"增强→编码"pipeline |
| 1:15-1:25 | 休息 | — | — |
| 1:25-1:55 | 实验3：不同SNR条件下的增强效果评估 | 边讲边练 | 制作"SNR→增强效果"的曲线图 |
| 1:55-2:20 | 延迟分析：DeepFilterNet的推理延迟是否满足助听器实时性要求？ | 讲授+实验 | 这是将深度学习模型部署到CI/助听器的核心挑战 |
| 2:20-2:30 | 总结：从"语音增强"到"CI语音增强"的gap在哪里？ | 讨论 | — |

### 配套Notebook

- `notebooks/03-ci-integration.ipynb`

### 课前准备

- 确认DeepFilterNet推理能正常运行
- 运行 `scripts/prepare_test_samples.py` 生成测试样本
- 预习文档：`预习-第3次课.md`

---

## 课后综合任务（分层）

- **基础**：运行DeepFilterNet推理，处理3段不同SNR的语音，计算PESQ/STOI/SI-SDR，提交分析报告
- **进阶**：将DeepFilterNet的增强输出送入ACE策略处理，对比"有增强"和"无增强"两种情况下ACE通道选择的差异

---

## 目录结构

```
module5-deepfilternet/
├── README.md                          # 本文件
├── DATA_REQUIREMENTS.md               # 数据集需求说明
├── exercises.md                       # 编程练习
├── 预习-第1次课.md                     # 课前预习材料
├── 预习-第2次课.md                     # 课前预习材料
├── 预习-第3次课.md                     # 课前预习材料
├── notebooks/
│   ├── 01-se-enhancement-basics.ipynb  # 语音增强基础 + 论文精读
│   ├── 02-code-analysis.ipynb          # 代码解析与运行
│   └── 03-ci-integration.ipynb         # CI结合实验
├── scripts/
│   └── prepare_test_samples.py         # 测试样本生成脚本
├── DeepFilterNet-main/                 # DeepFilterNet源码
│   ├── DeepFilterNet/                  # Python包
│   │   └── df/                         # 核心代码
│   ├── models/                         # 预训练模型权重
│   └── assets/                         # 示例音频文件
└── test_samples/                       # 生成的测试样本（运行脚本后）
```

## 课前准备

- DeepFilterNet论文PDF提前分发
- 预训练权重就绪（解压 `models/DeepFilterNet3.zip`）
- 安装DeepFilterNet（`pip install deep-filter` 或从源码安装）
- 准备不同SNR的带噪语音样本（运行 `scripts/prepare_test_samples.py`）
