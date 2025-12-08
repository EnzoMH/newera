"""
API Router 통합
단일 책임: 모든 API 라우터를 통합하여 FastAPI 앱에 등록
"""
from fastapi import APIRouter

from .routers import rag_router, health_router, system_router, agent_router

# 메인 API 라우터 생성
router = APIRouter()

# 하위 라우터 등록
router.include_router(rag_router)
router.include_router(health_router)
router.include_router(system_router)
router.include_router(agent_router)

# 하위 호환성을 위한 export
from .dependencies import set_rag_system

__all__ = ["router", "set_rag_system"]
