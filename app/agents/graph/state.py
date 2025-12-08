"""
LangGraph State 정의
RAG Agent를 위한 State 구조
"""
from typing import TypedDict, List, Dict, Optional
from enum import Enum


class AgentStatus(str, Enum):
    """Agent 처리 상태"""
    PENDING = "pending"
    INITIALIZING = "initializing"
    RETRIEVING = "retrieving"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"


class RAGAgentState(TypedDict):
    """
    RAG Agent State
    LangGraph 워크플로우에서 사용되는 상태 구조
    """

    # Input
    question: str
    conversation_id: Optional[str]

    # Processing Status
    status: AgentStatus
    current_step: str
    progress: int  # 0-100

    # RAG Components
    retrieved_docs: List[Dict]
    context: str

    # Generation
    answer: str
    sources: List[Dict]

    # Memory
    memory_key: str  # LangChain Memory key
    conversation_history: List[Dict]  # 대화 히스토리

    # Metadata
    metadata: Dict[str, any]

    # Error
    error: Optional[str]


# State 초기화 헬퍼 함수
def create_initial_state(question: str, conversation_id: Optional[str] = None) -> RAGAgentState:
    """
    초기 RAG Agent State 생성

    Args:
        question: 사용자 질문
        conversation_id: 대화 ID (없으면 자동 생성)

    Returns:
        초기화된 RAGAgentState
    """
    return RAGAgentState(
        question=question,
        conversation_id=conversation_id or f"conv_{hash(question) % 10000:04d}",
        status=AgentStatus.PENDING,
        current_step="초기화 중",
        progress=0,
        retrieved_docs=[],
        context="",
        answer="",
        sources=[],
        memory_key=conversation_id or f"conv_{hash(question) % 10000:04d}",
        conversation_history=[],
        metadata={},
        error=None
    )
