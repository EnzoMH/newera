"""
Memory 베이스 인터페이스
LangChain Memory 시스템의 기본 인터페이스
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseMemory(ABC):
    """
    메모리 시스템 추상 베이스 클래스
    모든 메모리 구현체가 상속받아야 함
    """

    def __init__(self, memory_key: str = "default"):
        self.memory_key = memory_key
        self.logger = None  # 서브클래스에서 설정

    @abstractmethod
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """
        대화 컨텍스트 저장

        Args:
            inputs: 입력 데이터
            outputs: 출력 데이터
        """
        pass

    @abstractmethod
    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        메모리 변수 로드

        Args:
            inputs: 입력 데이터

        Returns:
            메모리 변수 딕셔너리
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """
        메모리 클리어
        """
        pass

    def get_memory_variables(self) -> List[str]:
        """
        메모리 변수 이름 목록

        Returns:
            변수 이름 리스트
        """
        return ["history"]

    def validate_data(self, data: Any) -> bool:
        """
        데이터 유효성 검증

        Args:
            data: 검증할 데이터

        Returns:
            유효성 여부
        """
        return data is not None

    def format_context(self, context_list: List[Dict[str, Any]]) -> str:
        """
        컨텍스트 리스트를 문자열로 포맷팅

        Args:
            context_list: 컨텍스트 딕셔너리 리스트

        Returns:
            포맷팅된 문자열
        """
        if not context_list:
            return ""

        formatted = []
        for ctx in context_list:
            human = ctx.get("human", "")
            ai = ctx.get("ai", "")
            if human and ai:
                formatted.append(f"Human: {human}\nAI: {ai}")

        return "\n\n".join(formatted)

