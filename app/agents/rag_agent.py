"""
RAG Agent 구현
LangGraph와 LangChain을 통합한 RAG 전문 Agent
"""
import logging
from typing import Dict, Any, Optional
from functools import lru_cache

from .base import BaseAgent
from .graph.state import create_initial_state, RAGAgentState, AgentStatus
from .graph.workflow import get_rag_workflow

logger = logging.getLogger(__name__)


class RAGAgent(BaseAgent):
    """
    RAG 전문 Agent
    LangGraph 워크플로우를 사용하여 RAG 작업을 수행
    """

    def __init__(self):
        super().__init__(
            name="RAGAgent",
            description="LangGraph 기반 RAG 전문 Agent"
        )
        self.logger = logger
        self.workflow = None
        self.is_initialized = False

        logger.info("🎯 RAG Agent 초기화 중...")

    def initialize(self) -> bool:
        """
        Agent 초기화
        워크플로우 및 의존성 설정

        Returns:
            초기화 성공 여부
        """
        try:
            # LangGraph 워크플로우 초기화
            self.workflow = get_rag_workflow()

            # RAG 시스템 준비 (나중에 별도 초기화)
            # TODO: RAGSystem 통합 시 여기서 초기화

            self.is_initialized = True
            logger.info("✅ RAG Agent 초기화 완료")
            return True

        except Exception as e:
            logger.error(f"❌ RAG Agent 초기화 실패: {e}", exc_info=True)
            return False

    def process_query(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        쿼리 처리
        LangGraph 워크플로우를 통해 RAG 작업 수행

        Args:
            query: 사용자 쿼리
            **kwargs: 추가 파라미터들 (conversation_id 등)

        Returns:
            처리 결과 딕셔너리
        """
        if not self.is_initialized:
            return self.handle_error(
                Exception("Agent가 초기화되지 않았습니다"),
                "초기화 확인"
            )

        # 쿼리 유효성 검증
        if not self.validate_query(query):
            return self.handle_error(
                ValueError("유효하지 않은 쿼리입니다"),
                "쿼리 검증"
            )

        try:
            logger.info(f"📥 RAG Agent 쿼리 처리 시작: {query[:50]}...")

            # 초기 State 생성
            conversation_id = kwargs.get("conversation_id")
            initial_state = create_initial_state(query, conversation_id)

            # 워크플로우 실행
            result_state = self.workflow.invoke(initial_state)

            # 응답 포맷팅
            response = self.format_response(result_state)

            logger.info(f"✅ RAG Agent 쿼리 처리 완료: {result_state.get('status')}")
            return response

        except Exception as e:
            return self.handle_error(e, "쿼리 처리")

    def get_status(self) -> Dict[str, Any]:
        """
        Agent 상태 조회

        Returns:
            상태 정보 딕셔너리
        """
        return {
            "name": self.name,
            "description": self.description,
            "initialized": self.is_initialized,
            "workflow_available": self.workflow is not None,
            "type": "langgraph_rag_agent"
        }

    def process_query_sync(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        동기 쿼리 처리 (비동기 워크플로우를 동기로 실행)
        FastAPI에서 사용하기 위한 래퍼

        Args:
            query: 사용자 쿼리
            **kwargs: 추가 파라미터들

        Returns:
            처리 결과
        """
        # 현재는 동기로 실행 (나중에 async 지원 시 변경)
        return self.process_query(query, **kwargs)


# 싱글톤 패턴
@lru_cache()
def get_rag_agent() -> RAGAgent:
    """
    RAG Agent 싱글톤 인스턴스

    Returns:
        RAGAgent 인스턴스
    """
    agent = RAGAgent()
    agent.initialize()  # 초기화 보장
    return agent
