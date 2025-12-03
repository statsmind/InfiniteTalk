FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN sed -i 's/archive.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list
RUN sed -i 's/security.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list
#RUN sed -i 's/http:\/\/archive.ubuntu.com/http:\/\/mirrors.tuna.tsinghua.edu.cn\/ubuntu/g' /etc/apt/sources.list

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

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 更新 alternatives 以确保 python3 指向 python3.10
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# 安装 Python 包
COPY requirements.txt .

RUN --mount=type=cache,target=/root/.cache/pip pip3 install --upgrade pip
#RUN pip3 install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu129
#RUN pip3 install -U xformers==0.0.28 --index-url https://download.pytorch.org/whl/cu121
#RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
#RUN pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu128
RUN --mount=type=cache,target=/root/.cache/pip pip3 install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://mirrors.nju.edu.cn/pytorch/whl/cu121
RUN --mount=type=cache,target=/root/.cache/pip pip3 install -U xformers==0.0.28 --index-url https://mirrors.nju.edu.cn/pytorch/whl/cu121

# 安装 misaki 和 flash-attn
RUN --mount=type=cache,target=/root/.cache/pip pip3 install misaki[en] ninja psutil packaging wheel
RUN wget https://gh-proxy.com/github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
RUN --mount=type=cache,target=/root/.cache/pip pip3 install flash_attn-2.7.4.post1+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
#RUN --mount=type=cache,target=/root/.cache/pip MAX_JOBS=4 CUDA_HOME=/usr/local/cuda pip3 install flash_attn==2.7.4.post1 --no-build-isolation
#RUN --mount=type=cache,target=/root/.cache/pip MAX_JOBS=4 CUDA_HOME=/usr/local/cuda pip3 install torch flash_attn==2.7.4.post1 --no-build-isolation

RUN --mount=type=cache,target=/root/.cache/pip pip3 install -r requirements.txt

COPY requirements-api.txt .
RUN --mount=type=cache,target=/root/.cache/pip pip3 install -r requirements-api.txt
# 安装 librosa 通过 conda
#RUN --mount=type=cache,target=/root/.cache/pip pip3 install auxlib conda==4.3.13
#RUN conda install -c conda-forge librosa
RUN --mount=type=cache,target=/root/.cache/pip pip3 install librosa

RUN apt-get install -y ffmpeg

# 设置工作目录
WORKDIR /app

# 复制代码到容器中
COPY . .

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["bash", "start_api.sh"]
