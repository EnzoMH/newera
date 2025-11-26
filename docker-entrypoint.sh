#!/bin/bash
set -e

echo "=================================="
echo "Semiconductor RAG System Starting"
echo "=================================="

# MongoDB 연결 대기
echo "Waiting for MongoDB..."
until python -c "import pymongo; pymongo.MongoClient('${MONGODB_URI}', serverSelectionTimeoutMS=2000).admin.command('ping')" 2>/dev/null; do
    echo "MongoDB is unavailable - sleeping"
    sleep 2
done

echo "MongoDB is up - proceeding"

# GPU 확인
echo "Checking GPU availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Faiss GPU 확인
echo "Checking Faiss GPU support..."
python -c "import faiss; print(f'Faiss GPU: {hasattr(faiss, \"StandardGpuResources\")}')"

echo "=================================="
echo "Starting RAG Application..."
echo "=================================="

# 전달된 명령어 실행
exec "$@"




