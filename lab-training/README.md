# CI/助听器实验室 - 开发技能培训

面向CI/助听器研究实验室本科生与硕士研究生的深度学习开发技能培训课程。

## 课程概览

| 模块 | 主题 | 课时 | 编程能力目标 |
|------|------|------|-------------|
| 模块0 | Python编程基础 | 3次×2.5h = 7.5h | 能读懂 |
| 模块1 | Linux基本使用与深度学习环境搭建 | 2次×2.5h = 5h | 能读懂 |
| 模块2 | 深度学习入门 | 3次×2.5h = 7.5h | 能修改 |
| 模块3 | 面向声音的分类模型 | 3次×2.5h = 7.5h | 能修改→能组合 |
| 模块4 | DeepACE模型解析 | 2-3次×2.5h = 5-7.5h | 能组合 |
| 模块5 | DeepFilterNet模型解析 | 2-3次×2.5h = 5-7.5h | 能组合 |
| 模块6 | ASR技术简介 | 2次×2.5h = 5h | 能构建 |

**贯穿主线项目：** CI语音增强与识别 Pipeline（带噪语音 → DeepFilterNet增强 → ACE编码策略 → ASR识别评估）

## 目录结构

```
lab-training/
├── README.md                    # 本文件
├── environment.yml              # Conda环境配置
├── docker/
│   └── Dockerfile               # Docker镜像配置（用于租用服务器）
├── colab/                       # Google Colab版本notebook
│   ├── module0/
│   ├── module2/
│   ├── module3/
│   ├── module4/
│   ├── module5/
│   └── module6/
├── module0-python-basics/       # Python编程基础
│   ├── notebooks/               # Jupyter notebook
│   └── exercises/               # 课后练习
├── module1-linux-env/           # Linux与环境搭建
│   ├── cheatsheet-linux.md      # Linux速查表
│   ├── cheatsheet-vim.md        # Vim速查表
│   └── exercises/               # 课后练习
├── module2-dl-intro/            # 深度学习入门
│   ├── notebooks/
│   └── exercises/
├── module3-audio-classification/ # 音频分类模型
│   ├── notebooks/
│   └── data/                    # 示例数据
├── module4-deepace/             # DeepACE模型解析
│   ├── notebooks/
│   └── pretrained/              # 预训练权重
├── module5-deepfilternet/       # DeepFilterNet模型解析
│   ├── notebooks/
│   └── pretrained/
├── module6-asr/                 # ASR技术简介
│   ├── notebooks/
│   └── pretrained/
└── final-project/               # 贯穿主线项目
    ├── pipeline.ipynb
    └── report-template.md
```

## 环境配置

### 方式一：Conda（推荐用于本地开发）

```bash
# 创建环境
conda env create -f environment.yml

# 激活环境
conda activate lab-training
```

### 方式二：Docker（推荐用于租用服务器）

```bash
# 构建镜像
docker build -t lab-training docker/

# 运行容器（带GPU支持）
docker run --gpus all -p 8888:8888 -v $(pwd):/workspace lab-training
```

### 方式三：Google Colab（推荐用于课后练习）

打开 `colab/` 目录下的notebook即可运行，无需任何环境配置。

## 计算资源策略

| 场景 | 推荐方案 | 说明 |
|------|----------|------|
| 课上实操 | 租用GPU服务器 | 统一环境，快速解决问题 |
| 课后练习（无需GPU） | 本地笔记本 | 代码阅读、数据处理、可视化 |
| 课后练习（需要GPU） | Google Colab | 免费T4 GPU，零配置 |
| 长训练实验 | AutoDL/恒源云 | 低成本按小时租用 |

## Git使用规范

培训期间要求使用Git进行版本管理：

```bash
# 首次获取课程代码
git clone <repository-url>
cd lab-training

# 每次课前更新
git pull

# 课后提交练习
git add .
git commit -m "moduleX: 完成XXX练习"
git push
```

## 编程能力渐进路线

```
模块0-1: 填空式 —— 给出框架，填关键代码
    ↓
模块2-3: 修改式 —— 在已有代码上改功能
    ↓
模块4-5: 组合式 —— 用已有模块拼出新功能
    ↓
模块6:   从零式 —— 独立完成小项目
```

## 参考

详细课程设计指引请参阅：`培训课程指引.md`
