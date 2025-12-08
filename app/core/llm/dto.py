"""
LLM Provider DTOs (Data Transfer Objects)
"""
from pydantic import BaseModel
from typing import Optional


class OllamaRequest(BaseModel):
    """Ollama LLM 요청"""
    prompt: str
    system_prompt: Optional[str] = None
    context: Optional[str] = None
    temperature: Optional[float] = 0.1


class OllamaResponse(BaseModel):
    """Ollama LLM 응답"""
    response: str
    model_name: str
    tokens_used: Optional[int] = None


class GeminiRequest(BaseModel):
    """Gemini LLM 요청"""
    prompt: str
    system_prompt: Optional[str] = None
    context: Optional[str] = None
    temperature: Optional[float] = 0.1


class GeminiResponse(BaseModel):
    """Gemini LLM 응답"""
    response: str
    model_name: str
    tokens_used: Optional[int] = None
