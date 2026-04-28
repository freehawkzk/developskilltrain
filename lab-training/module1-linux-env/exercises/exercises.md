# 模块1 课后练习

## 第1次课课后练习：Linux基本操作

### 基础（必做）

完成以下操作并记录每一步的命令和输出：

1. **SSH连接服务器**
   - 用SSH连接到租用的服务器
   - 运行 `hostname` 和 `whoami` 确认身份
   - 运行 `nvidia-smi` 查看GPU信息

2. **文件操作**
   - 在home目录下创建 `~/lab-training/module1/` 目录
   - 用 `echo` 命令创建一个文本文件 `hello.txt`，内容为你的名字
   - 复制 `hello.txt` 为 `hello_backup.txt`
   - 查看 `hello.txt` 的内容
   - 查看 `hello_backup.txt` 的权限

3. **查找与统计**
   - 找到服务器上的示例音频文件目录
   - 统计 `.wav` 文件的数量
   - 查看其中一个 `.wav` 文件的大小

4. **tmux使用**
   - 创建一个名为 `test` 的tmux session
   - 在session中运行 `htop` 或 `nvidia-smi`（持续监控模式）
   - 分离session
   - 重新连接session
   - 退出tmux

**验收标准：** 将操作记录（命令+输出）保存为 `exercise1-log.txt`，提交到Git仓库。

### 进阶（选做）

写一个bash脚本 `batch_convert.sh`，功能如下：
- 接受两个参数：输入目录和输出目录
- 将输入目录下所有 `.wav` 文件转换为16kHz采样率，保存到输出目录
- 打印处理进度（如 "Processing 1/10: file1.wav"）
- 如果输出目录不存在，自动创建

提示：
```bash
#!/bin/bash
# batch_convert.sh - 批量转换wav文件采样率

# TODO: 检查参数数量
# TODO: 创建输出目录
# TODO: 遍历输入目录的wav文件
# TODO: 使用sox转换采样率
# TODO: 打印进度
```

---

## 第2次课课后练习：环境搭建

### 基础（必做）

1. **本地环境**
   - 在自己的笔记本上安装Miniconda
   - 创建 `lab-training` 环境
   - 安装PyTorch（CPU版本即可）
   - 运行以下验证代码：

   ```python
   import torch
   import numpy as np
   import matplotlib.pyplot as plt
   import librosa
   
   print(f'PyTorch: {torch.__version__}')
   print(f'NumPy: {np.__version__}')
   print(f'librosa: {librosa.__version__}')
   print(f'CUDA available: {torch.cuda.is_available()}')
   
   # 创建一个简单的张量运算
   x = torch.randn(3, 3)
   y = torch.randn(3, 3)
   z = torch.mm(x, y)
   print(f'Matrix multiplication result shape: {z.shape}')
   ```

2. **服务器环境**
   - SSH连接到租用的服务器
   - 验证 `torch.cuda.is_available()` 返回 `True`
   - 验证GPU计算功能正常

3. **Colab环境**
   - 打开 [Google Colab](https://colab.research.google.com/)
   - 新建一个notebook
   - 更改运行时类型为T4 GPU
   - 运行相同的验证代码

**验收标准：** 三种环境（本地/服务器/Colab）的验证代码都成功运行，截图或复制输出保存。

### 进阶（选做）

1. **Docker使用**
   - 使用培训提供的Docker镜像启动容器
   - 在容器中运行PyTorch验证代码
   - 对比本地环境、服务器环境、Docker环境、Colab环境四种方式的差异

2. **环境配置文件**
   - 将你当前环境的配置导出为 `environment.yml`
   - 在另一台机器上用这个文件创建相同的环境
   - 验证新环境是否正常工作

3. **Jupyter Lab配置**
   - 配置Jupyter Lab的远程访问
   - 安装有用的扩展（如变量检查器、代码格式化）
   - 配置自动补全和快捷键
