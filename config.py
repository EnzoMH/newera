"""
프로젝트 설정 파일
환경 변수에서 값을 로드하며, 기본값을 제공합니다.
"""
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# Ollama 설정
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "exaone-1.2b:latest")
