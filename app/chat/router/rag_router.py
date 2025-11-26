"""
RAG 라우터
"""
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.chat.dto.dto_rq import ChatRequest
from app.chat.dto.dto_rp import ChatResponse


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rag", tags=["RAG"])


class RAGQueryRequest(BaseModel):
    """RAG 쿼리 요청"""
    query: str = Field(..., description="검색 질의")
    top_k: int = Field(5, description="검색할 문서 수", ge=1, le=20)
    use_gemini: bool = Field(False, description="Gemini 사용 여부")
    filter_domain: str | None = Field(None, description="도메인 필터")


class RAGQueryResponse(BaseModel):
    """RAG 쿼리 응답"""
    answer: str
    sources: list[dict]
    metadata: dict


rag_service = None


def set_rag_service(service):
    """RAG 서비스 설정 (의존성 주입)"""
    global rag_service
    rag_service = service


@router.post("/query", response_model=RAGQueryResponse)
async def rag_query(request: RAGQueryRequest) -> RAGQueryResponse:
    """
    RAG 쿼리 엔드포인트
    """
    if rag_service is None:
        raise HTTPException(status_code=503, detail="RAG 서비스가 초기화되지 않았습니다")
    
    try:
        result = rag_service.query(
            question=request.query,
            top_k=request.top_k,
            use_gemini=request.use_gemini,
            filter_domain=request.filter_domain
        )
        
        return RAGQueryResponse(**result)
    
    except Exception as e:
        logger.error(f"RAG 쿼리 실패: {e}")
        raise HTTPException(status_code=500, detail=f"RAG 쿼리 실패: {str(e)}")


@router.get("/health")
async def rag_health() -> dict:
    """RAG 시스템 헬스 체크"""
    if rag_service is None:
        return {"status": "unhealthy", "message": "RAG 서비스 미초기화"}
    
    try:
        stats = rag_service.retriever.get_stats()
        return {
            "status": "healthy",
            "stats": stats
        }
    except Exception as e:
        logger.error(f"헬스 체크 실패: {e}")
        return {"status": "unhealthy", "error": str(e)}




