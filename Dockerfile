FROM nvidia/cuda:12.1-devel-ubuntu22.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
    python3-pip \
    git \
    wget \
    curl \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 更新 alternatives 以确保 python3 指向 python3.10
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# 安装 Python 包
COPY requirements.txt .
RUN pip3 install --upgrade pip
RUN pip3 install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install -U xformers==0.0.28 --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install -r requirements.txt

# 安装 misaki 和 flash-attn
RUN pip3 install misaki[en]
RUN pip3 install ninja psutil packaging wheel
RUN pip3 install flash_attn==2.7.4.post1

# 安装 librosa 通过 conda
RUN pip3 install conda
RUN conda install -c conda-forge librosa

# 设置工作目录
WORKDIR /app

# 复制代码到容器中
COPY . .

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["python3", "api.py"]