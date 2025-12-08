"""
LLM Providers Package
"""
# llm.py는 상위 디렉토리에 있으므로 직접 import
# 순환 import 방지를 위해 importlib 사용
import importlib.util
from pathlib import Path

llm_file = Path(__file__).parent.parent / "llm.py"
spec = importlib.util.spec_from_file_location("llm_module", llm_file)
llm_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(llm_module)

OllamaLLMProvider = llm_module.OllamaLLMProvider

from .dto import OllamaRequest, OllamaResponse, GeminiRequest, GeminiResponse

__all__ = [
    "OllamaLLMProvider",
    "OllamaRequest",
    "OllamaResponse",
    "GeminiRequest",
    "GeminiResponse"
]