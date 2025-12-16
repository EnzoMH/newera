"""
프로젝트 설정 파일
환경 변수에서 값을 로드하며, 기본값을 제공합니다.
"""
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# LLM 설정 (LlamaCpp 사용)
MODEL_NAME = os.getenv("MODEL_NAME", "LGAI-EXAONE/EXAONE-4.0-1.2B-GGUF")

# 양자화 옵션 (파일 크기 선택):
# - Q2_K: ~500MB (가장 작음, 품질 낮음)
# - Q3_K_M: ~550MB (작음, 품질 중간)
# - Q4_K_M: ~700MB (권장, 품질 좋음)
# - Q5_K_M: ~850MB (큼, 품질 매우 좋음)
LLAMA_CPP_FILENAME = os.getenv("LLAMA_CPP_FILENAME", "EXAONE-4.0-1.2B-Q4_K_M.gguf")
LLAMA_CPP_N_GPU_LAYERS = os.getenv("LLAMA_CPP_N_GPU_LAYERS", "35")
LLAMA_CPP_N_CTX = os.getenv("LLAMA_CPP_N_CTX", "4096")

# 호환성을 위한 Ollama 설정 (더 이상 사용하지 않음)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
