# CI/助听器实验室培训 - Docker镜像
# 基于PyTorch官方镜像，预装所有依赖
# 用法: docker build -t lab-training .
# 运行: docker run --gpus all -p 8888:8888 -v $(pwd):/workspace lab-training

FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# 避免交互式安装提示
ENV DEBIAN_FRONTEND=noninteractive

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    ffmpeg \
    sox \
    libsox-dev \
    vim \
    tmux \
    && rm -rf /var/lib/apt/lists/*

# 安装Conda包
RUN conda install -y -c conda-forge \
    numpy \
    scipy \
    matplotlib \
    pandas \
    scikit-learn \
    jupyterlab \
    ipywidgets \
    tqdm \
    librosa \
    soundfile \
    tensorboard \
    && conda clean -afy

# 安装pip包
RUN pip install --no-cache-dir \
    pesq \
    pystoi \
    mir-eval \
    openai-whisper \
    deepfilternet \
    wandb

# 配置Jupyter
RUN jupyter lab --generate-config && \
    echo "c.ServerApp.ip = '0.0.0.0'" >> ~/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.port = 8888" >> ~/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.open_browser = False" >> ~/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.allow_root = True" >> ~/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.token = ''" >> ~/.jupyter/jupyter_lab_config.py

# 创建工作目录
WORKDIR /workspace

# 暴露Jupyter端口
EXPOSE 8888

# 启动Jupyter Lab
CMD ["jupyter", "lab", "--allow-root"]
