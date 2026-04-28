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

## 课前准备

- DeepFilterNet论文PDF提前分发
- 预训练权重就绪
- 准备不同SNR的带噪语音样本
