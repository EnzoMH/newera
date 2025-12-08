"""
System API Router
단일 책임: 시스템 상태 및 정보 조회 엔드포인트
"""
import logging
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from ..schemas import SystemStatusResponse
from ..dependencies import get_rag_system, check_rag_initialized
from ...core.rag import RAGSystem

logger = logging.getLogger(__name__)

# 라우터 생성
router = APIRouter(
    prefix="/system",
    tags=["System"],
    responses={
        200: {"description": "요청 성공"},
        503: {"description": "서비스 사용 불가"}
    }
)


@router.get("/status", response_model=SystemStatusResponse)
async def system_status(
    rag_system: RAGSystem = Depends(get_rag_system)
) -> SystemStatusResponse:
    """
    상세 시스템 상태 조회
    
    RAG 시스템의 모든 컴포넌트 상태를 상세히 반환합니다.
    
    Returns:
        SystemStatusResponse: 상세 시스템 상태 정보
    """
    try:
        status_info = rag_system.get_status()
        
        response = SystemStatusResponse(
            initialized=status_info["initialized"],
            llm_available=status_info["llm_available"],
            vector_store_available=status_info["vector_store_available"],
            crawler_available=status_info.get("crawler_available", False),
            retriever_available=status_info.get("retriever_available", False),
            domain=status_info["domain"],
            components={
                "llm": {
                    "provider": "ollama",
                    "available": status_info["llm_available"]
                },
                "vecdb": {
                    "available": status_info["vector_store_available"]
                },
                "crawler": {
                    "available": status_info.get("crawler_available", False)
                },
                "retriever": {
                    "available": status_info.get("retriever_available", False)
                }
            }
        )
        
        return response
        
    except Exception as e:
        logger.error(f"❌ 시스템 상태 조회 실패: {e}", exc_info=True)
        raise HTTPException(
            status_code=503,
            detail=f"시스템 상태 조회 중 오류가 발생했습니다: {str(e)}"
        )


@router.get("/info")
async def system_info() -> JSONResponse:
    """
    시스템 정보 조회
    
    RAG 시스템 없이도 동작하는 기본 시스템 정보를 반환합니다.
    
    Returns:
        JSONResponse: 시스템 정보
    """
    rag_initialized = check_rag_initialized()
    
    info = {
        "system": "VirtualFab RAG System",
        "version": "2.1.0",
        "domain": "VirtualFab/Digital Twin",
        "rag_initialized": rag_initialized,
        "components": {
            "api": "available",
            "rag": "available" if rag_initialized else "not_initialized",
            "mcp": "available"
        }
    }
    
    return JSONResponse(content=info)

