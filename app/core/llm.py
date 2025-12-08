"""
Ollama LLM Provider
단일 책임: Ollama LLM과의 상호작용
"""
import logging
from typing import Optional
from langchain_ollama import OllamaLLM
import os

from app.core.llm.dto import OllamaRequest, OllamaResponse

logger = logging.getLogger(__name__)


class OllamaLLMProvider:
    """
    Ollama LLM Provider
    - 단일 책임: Ollama LLM과의 상호작용 및 응답 생성
    """

    def __init__(self, model_name: str = "qwen2.5-3b-instruct:latest"):
        self.model_name = model_name
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        # Ollama LLM 초기화
        self.llm = OllamaLLM(
            model=self.model_name,
            base_url=self.base_url,
            temperature=0.1
        )

        logger.info(f"✅ Ollama LLM Provider 초기화: {self.model_name}")

    def generate_response(self, request: OllamaRequest) -> OllamaResponse:
        """
        기본 응답 생성

        Args:
            request: Ollama 요청 객체

        Returns:
            Ollama 응답 객체
        """
        try:
            # 기본 프롬프트 생성
            full_prompt = request.prompt

            # 시스템 프롬프트가 있는 경우 추가
            if request.system_prompt:
                full_prompt = f"[시스템]\n{request.system_prompt}\n\n[질문]\n{request.prompt}"

            # 컨텍스트가 있는 경우 추가
            if request.context:
                full_prompt = f"[컨텍스트]\n{request.context}\n\n{full_prompt}"

            # LLM 호출
            response_text = self.llm.invoke(full_prompt)

            return OllamaResponse(
                response=response_text,
                model_name=self.model_name
            )

        except Exception as e:
            logger.error(f"Ollama 응답 생성 실패: {e}")
            return OllamaResponse(
                response=f"오류가 발생했습니다: {str(e)}",
                model_name=self.model_name
            )

    def generate_simple_response(self, prompt: str, temperature: float = 0.1) -> str:
        """
        간단한 텍스트 응답 생성

        Args:
            prompt: 프롬프트 텍스트
            temperature: 온도 설정

        Returns:
            응답 텍스트
        """
        try:
            # 온도 설정 업데이트
            self.llm.temperature = temperature

            response = self.llm.invoke(prompt)
            return str(response) if response else ""

        except Exception as e:
            logger.error(f"Ollama 간단 응답 생성 실패: {e}")
            return f"오류가 발생했습니다: {str(e)}"

    def is_available(self) -> bool:
        """
        Ollama 서비스 사용 가능 여부 확인

        Returns:
            사용 가능 여부
        """
        try:
            # 간단한 테스트 호출
            test_response = self.llm.invoke("test")
            return bool(test_response)
        except Exception:
            return False