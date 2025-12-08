"""
Conversation Memory 구현
LangChain ConversationBufferMemory 기반 구현
"""
import logging
from typing import Any, Dict, List, Optional
from functools import lru_cache

from langchain.memory import ConversationBufferMemory

from .base import BaseMemory

logger = logging.getLogger(__name__)


class ConversationBufferMemory(BaseMemory):
    """
    대화 버퍼 메모리
    최근 대화들을 버퍼에 저장하고 유지
    """

    def __init__(self, memory_key: str = "default", max_token_limit: int = 2000):
        super().__init__(memory_key)
        self.logger = logger
        self.max_token_limit = max_token_limit

        # LangChain ConversationBufferMemory 초기화
        self.langchain_memory = ConversationBufferMemory(
            memory_key=self.memory_key,
            max_token_limit=self.max_token_limit
        )

        logger.info(f"💬 Conversation Memory 초기화: {memory_key}")

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """
        대화 컨텍스트 저장

        Args:
            inputs: 입력 데이터 ({"human": "질문"})
            outputs: 출력 데이터 ({"ai": "답변"})
        """
        try:
            human_input = inputs.get("human", "")
            ai_output = outputs.get("ai", "")

            if not human_input or not ai_output:
                logger.warning("저장할 컨텍스트가 불완전합니다")
                return

            # LangChain 메모리에 저장
            self.langchain_memory.save_context(inputs, outputs)

            logger.debug(f"💾 메모리 저장: {len(human_input)}자 입력")

        except Exception as e:
            logger.error(f"메모리 저장 실패: {e}", exc_info=True)

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        메모리 변수 로드

        Args:
            inputs: 입력 데이터

        Returns:
            메모리 변수 딕셔너리
        """
        try:
            # LangChain 메모리에서 로드
            memory_vars = self.langchain_memory.load_memory_variables(inputs)

            # 버퍼 크기 확인
            buffer_size = len(self.langchain_memory.buffer)
            logger.debug(f"📖 메모리 로드: {buffer_size}개 대화")

            return memory_vars

        except Exception as e:
            logger.error(f"메모리 로드 실패: {e}", exc_info=True)
            return {self.memory_key: ""}

    def clear(self) -> None:
        """
        메모리 클리어
        """
        try:
            self.langchain_memory.clear()
            logger.info("🧹 메모리 클리어됨")

        except Exception as e:
            logger.error(f"메모리 클리어 실패: {e}", exc_info=True)

    def get_buffer_size(self) -> int:
        """
        현재 버퍼 크기 반환

        Returns:
            버퍼에 저장된 대화 수
        """
        return len(self.langchain_memory.buffer)

    def get_memory_variables(self) -> List[str]:
        """
        메모리 변수 이름 목록

        Returns:
            변수 이름 리스트
        """
        return [self.memory_key]


# 간단한 메모리 인스턴스 관리
_memory_instances = {}


def get_conversation_memory(memory_key: str = "default") -> ConversationBufferMemory:
    """
    대화 메모리 인스턴스

    Args:
        memory_key: 메모리 키

    Returns:
        ConversationBufferMemory 인스턴스
    """
    if memory_key not in _memory_instances:
        _memory_instances[memory_key] = ConversationBufferMemory(memory_key=memory_key)

    return _memory_instances[memory_key]


def clear_all_memories():
    """
    모든 메모리 인스턴스 클리어
    """
    global _memory_instances
    for memory in _memory_instances.values():
        memory.clear()
    _memory_instances.clear()
    logger.info("🧹 모든 메모리 인스턴스 클리어됨")
