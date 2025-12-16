# VirtualFab RAG System - Backend Dockerfile
FROM python:3.12-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 의존성 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY app/ ./app/
COPY main.py .
COPY config.py .
COPY mcp_config.json .

# 모델 디렉토리 생성
RUN mkdir -p models data/vectorstore logs

# Non-root 사용자 생성
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# 환경 변수
ENV PYTHONUNBUFFERED=1
ENV API_HOST=0.0.0.0
ENV API_PORT=8000

# 헬스체크
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# 포트 노출
EXPOSE 8000

# 서버 실행
CMD ["python", "main.py"]
