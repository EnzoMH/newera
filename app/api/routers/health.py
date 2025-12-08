"""
Health Check API Router
단일 책임: 시스템 헬스체크 엔드포인트
"""
import logging
from fastapi import APIRouter
from datetime import datetime

from ..schemas import HealthResponse
from ..dependencies import check_rag_initialized

logger = logging.getLogger(__name__)

# 라우터 생성
router = APIRouter(
    prefix="/health",
    tags=["Health"],
    responses={
        200: {"description": "헬스체크 성공"}
    }
)


@router.get("", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    시스템 헬스체크
    
    RAG 시스템 없이도 동작하는 기본 헬스체크 엔드포인트입니다.
    
    Returns:
        HealthResponse: 시스템 상태 정보
    """
    rag_initialized = check_rag_initialized()
    
    return HealthResponse(
        status="healthy",
        initialized=rag_initialized,
        llm_available=rag_initialized,
        vector_store_available=False,
        domain="VirtualFab/Digital Twin",
        version="2.1.0"
    )

