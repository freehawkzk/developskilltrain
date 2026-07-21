# 模块4：DeepACE模型解析

## 课程安排

**课时：一个上午，3小时**

| 时间 | 内容 | 形式 | 对应 Notebook |
|------|------|------|-------------|
| 0:00-0:40 | ACE 策略回顾、局限与 DeepACE 问题定义 | 讲授+共读 | [01-ace-and-paper](notebooks/01-ace-and-paper.ipynb) |
| 0:40-1:25 | DeepACE 网络架构、训练策略与实验结果 | 共读+讨论 | [01-ace-and-paper](notebooks/01-ace-and-paper.ipynb) |
| 1:25-1:35 | 休息 | — | — |
| 1:35-2:15 | 代码结构、核心模块与数据流解析 | 讲授+代码 | [02-code-analysis](notebooks/02-code-analysis.ipynb) |
| 2:15-2:40 | Mini 数据集训练、推理与通道选择可视化 | 实操 | [02-code-analysis](notebooks/02-code-analysis.ipynb) |
| 2:40-3:00 | 修改假设设计与对比实验讨论 | 讨论+引导练习 | [03-modification-experiments](notebooks/03-modification-experiments.ipynb) |

## 学习目标

完成本模块后，学生应能够：
- 理解ACE策略的工作原理和局限性
- 理解DeepACE的核心创新点和网络架构
- 精读一篇深度学习论文的方法（领读+结构化阅读）
- 运行DeepACE的训练和推理流程
- 理解数据流：音频输入→特征提取→网络处理→通道选择输出
- （进阶）对模型进行修改实验

## 编程练习设计模式

本模块采用**组合式**练习——将已有模块拼在一起。重点是读代码和运行代码，理解数据流。

> **教学策略提醒：** 这个模块的教学方式跟前三个模块有本质区别——前面是"学技能"，这里是"读论文+读代码"。学生需要适应从"跟着教程做"到"独立分析已有工作"的转变。

## 目录结构

```
module4-deepace/
├── ACE/                    # 传统ACE策略的Python实现
│   ├── ace_strategy.py     # ACE策略入口
│   ├── ace/ace_process.py  # ACE处理核心代码
│   └── ...
├── DeepACE_torch/          # DeepACE原始代码
│   ├── config.yaml         # 训练配置
│   ├── model.py            # 主模型
│   ├── netblocks.py        # 网络子模块
│   ├── dataset.py          # 数据集加载
│   ├── train.py            # 训练入口
│   ├── test.py             # 测试入口
│   ├── losses.py           # 损失函数
│   └── data/               # 数据目录（需生成）
├── notebooks/              # 教学notebooks
├── scripts/
│   └── prepare_mini_dataset.py  # mini数据集生成脚本
├── pretrained/             # 预训练权重
├── exercises/              # 课后练习
├── DATA_REQUIREMENTS.md    # 数据集需求说明
└── 预习-第X次课.md          # 上午内容单元的预习文档
```

## 数据集

详见 [DATA_REQUIREMENTS.md](DATA_REQUIREMENTS.md)

mini数据集通过 `scripts/prepare_mini_dataset.py` 自动生成，包含：
- 训练集：20条（mixture WAV + target .mat）
- 验证集：5条
- 测试集：3条（仅mixture）
- 总大小约10MB，训练一个epoch约5-30秒

## 前置条件

- 模块0-3的全部内容
- PyTorch环境已配置，GPU可用（或使用Colab）
- DeepACE论文PDF已分发

## 课后综合任务（分层）

- **基础**：运行DeepACE的完整训练和推理流程，提交一份运行记录和输出分析
- **进阶**：设计并实施一个修改实验，对比修改前后的效果差异，写一份简短实验报告

## 课后GPU资源说明

模块4的全部课后练习可以在 Google Colab 上完成。详见 [colab/module4/](../colab/module4/)。
