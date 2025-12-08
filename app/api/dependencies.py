"""
API Dependencies
단일 책임: FastAPI 의존성 주입 관리
"""
import logging
from typing import Optional
from fastapi import HTTPException

from ..core.rag import RAGSystem

logger = logging.getLogger(__name__)

# 전역 RAG 시스템 인스턴스
_rag_system: Optional[RAGSystem] = None


def set_rag_system(rag_system: RAGSystem) -> None:
    """
    RAG 시스템 의존성 주입
    
    Args:
        rag_system: 초기화된 RAGSystem 인스턴스
    """
    global _rag_system
    _rag_system = rag_system
    logger.info("✅ RAG 시스템 의존성 주입 완료")


def get_rag_system() -> RAGSystem:
    """
    RAG 시스템 의존성 제공자
    FastAPI 엔드포인트에서 Depends로 사용
    
    Returns:
        RAGSystem 인스턴스
        
    Raises:
        HTTPException: RAG 시스템이 초기화되지 않은 경우
    """
    if _rag_system is None:
        logger.error("❌ RAG 시스템이 초기화되지 않았습니다")
        raise HTTPException(
            status_code=503,
            detail="RAG 시스템이 초기화되지 않았습니다. 서버 시작 시 초기화가 필요합니다."
        )
    return _rag_system


def get_rag_agent_dependency():
    """
    RAG Agent 의존성 제공자
    FastAPI 엔드포인트에서 Depends로 사용

    Returns:
        RAGAgent 인스턴스
    """
    try:
        from ..agents import get_rag_agent
        return get_rag_agent()
    except Exception as e:
        logger.error(f"❌ RAG Agent 의존성 주입 실패: {e}")
        raise HTTPException(
            status_code=503,
            detail="RAG Agent를 초기화할 수 없습니다."
        )


def check_rag_initialized() -> bool:
    """
    RAG 시스템 초기화 여부 확인

    Returns:
        초기화 여부
    """
    return _rag_system is not None and _rag_system.is_initialized


def check_agent_available() -> bool:
    """
    RAG Agent 사용 가능 여부 확인

    Returns:
        사용 가능 여부
    """
    try:
        agent = get_rag_agent_dependency()
        return agent.is_initialized
    except Exception:
        return False

