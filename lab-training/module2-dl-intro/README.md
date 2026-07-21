# 模块2：深度学习入门

## 课程安排

**课时：一个上午，3小时**

| 时间 | 内容 | 形式 | 对应 Notebook |
|------|------|------|-------------|
| 0:00-0:40 | 线性回归、激活函数、MLP 与反向传播直觉 | 讲授+实操 | [01-linear-regression](notebooks/01-linear-regression.ipynb)、[02-mlp-pytorch](notebooks/02-mlp-pytorch.ipynb) |
| 0:40-1:20 | PyTorch 张量、自动微分与第一个 MLP 训练 | 边讲边练 | [02-mlp-pytorch](notebooks/02-mlp-pytorch.ipynb) |
| 1:20-1:30 | 休息 | — | — |
| 1:30-2:10 | 语谱图、卷积网络与语音/噪声分类 | 讲授+实操 | [03-cnn-audio](notebooks/03-cnn-audio.ipynb) |
| 2:10-2:40 | 评估、调参、数据增强与实验记录 | 讲授+演示 | [04-training-tricks](notebooks/04-training-tricks.ipynb) |
| 2:40-3:00 | 主线项目练习与答疑 | 独立编写 | `03`、`04` 综合 |

## 学习目标

- 理解"学习"就是"调参数"
- 理解反向传播的直觉（不要求推导）
- 理解语谱图：音频可以变成"图像"来处理
- 用PyTorch搭建并训练MLP和CNN
- 知道如何评估模型、调参、做实验记录

## 编程练习设计模式

- 上午前段（notebook 01-02）：**填空式**
- 上午中段（notebook 03）：**修改式**（在已有代码上改参数和结构）
- 上午后段（notebook 04）：**框架注释式**（只给描述，学生自己写代码）

## 前置条件

- 模块0和模块1的全部内容
- PyTorch环境已配置完成，GPU可用

## 课后综合任务（分层）

- **基础**：完成主线项目第一阶段——训练一个能区分语音和噪声的CNN，准确率 > 85%
- **进阶**：尝试不同的网络结构（加深、加宽、加ResNet连接），在TensorBoard中对比结果

## 课后GPU资源说明

模块2的课后训练任务在Google Colab上完全可以完成（免费T4 GPU足够训练小型CNN）。Colab notebook 会预先配置好数据集加载代码，学生只需关注模型定义和训练循环。详见 [colab/module2/](../colab/module2/)。
