# CUDA 11.6.2 기반 Ubuntu 20.04 이미지 사용
# FROM nvcr.io/nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04
FROM nvcr.io/nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

# 필수 패키지 설치
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Miniconda 설치
RUN wget --quiet https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    ln -s /opt/conda/bin/conda /usr/local/bin/conda
ENV PATH=/opt/conda/bin:$PATH

COPY . /app/RFdiffusion/
WORKDIR /app/RFdiffusion

# Conda 환경 생성 (패키지 차이 고려)
RUN conda env create -f env/SE3nv.yml

# Conda 환경 활성화
RUN . ~/.bashrc && \
    /opt/conda/bin/conda init bash
RUN echo "source activate SE3nv" >> ~/.bashrc
ENV PATH=/opt/conda/envs/SE3nv/bin:$PATH

RUN pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

# SE3Transformer 및 RFdiffusion 설치
RUN cd env/SE3Transformer && \
    pip install --no-cache-dir -r requirements.txt && \
    python setup.py install && \
    cd ../..
RUN pip install -e .
RUN pip install --no-cache-dir "numpy<2"

# Conda 환경에서 실행되도록 설정
# SHELL ["/bin/bash", "-c"]

# 실행 환경 설정
ENV DGLBACKEND="pytorch"
