# CUDA 12.6 베이스 이미지 (Ubuntu 22.04)
FROM nvidia/cuda:12.6.0-cudnn9-runtime-ubuntu22.04

# 메타데이터
LABEL maintainer="your-email@example.com"
LABEL description="Semiconductor RAG System with Faiss GPU"

# 환경 변수 설정
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda-12.6
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV PATH=${CUDA_HOME}/bin:${PATH}

# 작업 디렉토리
WORKDIR /app

# 시스템 패키지 업데이트 및 Python 설치
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Python 심볼릭 링크
RUN ln -s /usr/bin/python3.11 /usr/bin/python

# pip 업그레이드
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# PyTorch with CUDA 12.6 지원
RUN pip install --no-cache-dir torch==2.5.1+cu126 --index-url https://download.pytorch.org/whl/cu126

# 나머지 Python 패키지 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# faiss-gpu 설치 (conda 사용)
RUN pip install --no-cache-dir conda && \
    conda install -y -c conda-forge faiss-gpu=1.8.0

# 애플리케이션 코드 복사
COPY . .

# MongoDB 연결을 위한 대기 스크립트
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# 포트 노출
EXPOSE 8001

# 헬스체크
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# 엔트리포인트
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]

# 기본 명령어
CMD ["python", "rag_main.py"]




