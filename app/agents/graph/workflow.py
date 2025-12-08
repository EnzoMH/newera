"""
LangGraph Workflow 정의
RAG Agent 워크플로우 생성 및 컴파일
"""
import logging
from langgraph.graph import StateGraph, END
from typing import Literal

from .state import RAGAgentState, AgentStatus
from .nodes import (
    initialize_agent,
    retrieve_documents,
    generate_answer,
    finalize_agent,
    handle_error
)

logger = logging.getLogger(__name__)


def create_rag_workflow():
    """
    RAG Agent 워크플로우 생성

    Returns:
        컴파일된 LangGraph 워크플로우
    """
    logger.info("🔄 RAG Agent 워크플로우 생성 중...")

    # StateGraph 생성
    workflow = StateGraph(RAGAgentState)

    # 노드 추가
    workflow.add_node("initialize", initialize_agent)
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("generate", generate_answer)
    workflow.add_node("finalize", finalize_agent)
    workflow.add_node("error_handler", handle_error)

    # 엣지 정의 (순차 실행)
    workflow.set_entry_point("initialize")
    workflow.add_edge("initialize", "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "finalize")
    workflow.add_edge("finalize", END)

    # 조건부 엣지 (에러 처리)
    def check_error(state: RAGAgentState) -> Literal["error_handler", "continue"]:
        """에러 발생 여부 확인"""
        if state.get("error") or state.get("status") == AgentStatus.FAILED:
            return "error_handler"
        return "continue"

    # 모든 노드에서 에러 체크 (실제로는 필요한 노드에만 적용)
    # workflow.add_conditional_edges("initialize", check_error)
    # workflow.add_conditional_edges("retrieve", check_error)
    # workflow.add_conditional_edges("generate", check_error)

    logger.info("✅ RAG Agent 워크플로우 생성 완료")
    return workflow.compile()


# 전역 워크플로우 인스턴스
_rag_workflow = None


def get_rag_workflow():
    """
    RAG 워크플로우 싱글톤
    성능을 위해 한 번만 생성

    Returns:
        컴파일된 워크플로우 인스턴스
    """
    global _rag_workflow
    if _rag_workflow is None:
        _rag_workflow = create_rag_workflow()
    return _rag_workflow
