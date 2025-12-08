"""
Base Agent 추상 클래스
모든 Agent의 기본 인터페이스 정의
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from app.agents.graph.state import RAGAgentState


class BaseAgent(ABC):
    """
    Agent 추상 베이스 클래스
    모든 Agent는 이 클래스를 상속받아야 함
    """

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.logger = None  # 서브클래스에서 설정

    @abstractmethod
    def initialize(self) -> bool:
        """
        Agent 초기화

        Returns:
            초기화 성공 여부
        """
        pass

    @abstractmethod
    def process_query(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        쿼리 처리 (메인 인터페이스)

        Args:
            query: 사용자 쿼리
            **kwargs: 추가 파라미터들

        Returns:
            처리 결과 딕셔너리
        """
        pass

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """
        Agent 상태 조회

        Returns:
            상태 정보 딕셔너리
        """
        pass

    def validate_query(self, query: str) -> bool:
        """
        쿼리 유효성 검증

        Args:
            query: 검증할 쿼리

        Returns:
            유효성 여부
        """
        if not query or not isinstance(query, str):
            return False

        query = query.strip()
        return len(query) > 0 and len(query) <= 1000

    def format_response(self, state: RAGAgentState) -> Dict[str, Any]:
        """
        Agent State를 API 응답 형식으로 변환

        Args:
            state: 최종 Agent State

        Returns:
            API 응답 형식 딕셔너리
        """
        # 검색 결과를 SourceDocument 형식으로 변환
        retrieved_docs = state.get("retrieved_docs", [])
        sources = []
        for doc in retrieved_docs:
            if isinstance(doc, dict):
                sources.append({
                    "filename": doc.get("source"),  # filename에 source 값 사용
                    "source": doc.get("source"),
                    "content": doc.get("content", ""),
                    "score": doc.get("score"),
                    "domain": doc.get("topic"),
                    "metadata": {
                        "chunk_id": doc.get("chunk_id"),
                        "topic": doc.get("topic")
                    }
                })

        return {
            "answer": state.get("answer", ""),
            "sources": sources,
            "metadata": {
                "agent": self.name,
                "status": state.get("status", "unknown"),
                "progress": state.get("progress", 0),
                "conversation_id": state.get("conversation_id"),
                **state.get("metadata", {})
            },
            "conversation_history": state.get("conversation_history", [])
        }

    def handle_error(self, error: Exception, context: str = "") -> Dict[str, Any]:
        """
        에러 처리 및 표준화된 에러 응답 생성

        Args:
            error: 발생한 예외
            context: 에러 발생 컨텍스트

        Returns:
            표준화된 에러 응답
        """
        error_msg = f"{context}: {str(error)}" if context else str(error)

        if self.logger:
            self.logger.error(f"❌ Agent 에러: {error_msg}", exc_info=True)

        return {
            "answer": f"오류가 발생했습니다: {error_msg}",
            "sources": [],
            "metadata": {
                "agent": self.name,
                "status": "error",
                "error": error_msg
            },
            "conversation_history": []
        }
