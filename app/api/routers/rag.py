"""
RAG API Router
단일 책임: RAG 질의 처리 엔드포인트
"""
import logging
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse

from ..schemas import QueryRequest, QueryResponse, HealthResponse, ErrorResponse
from ..dependencies import get_rag_system
from ...core.rag import RAGSystem

logger = logging.getLogger(__name__)

# 라우터 생성
router = APIRouter(
    prefix="/rag",
    tags=["RAG"],
    responses={
        404: {"description": "엔드포인트를 찾을 수 없습니다"},
        500: {"description": "서버 내부 오류"}
    }
)


@router.post("/query", response_model=QueryResponse)
async def query_rag(
    request: QueryRequest,
    rag_system: RAGSystem = Depends(get_rag_system)
) -> QueryResponse:
    """
    RAG 질의 처리 엔드포인트
    
    사용자 질문을 받아 RAG 시스템을 통해 답변을 생성합니다.
    
    - **question**: 사용자 질문 (필수, 1-1000자)
    - **temperature**: 응답 다양성 (0.0-1.0, 기본값: 0.1)
    - **max_tokens**: 최대 토큰 수 (선택)
    
    Returns:
        QueryResponse: AI 답변, 참고 문서, 메타정보 포함
    """
    try:
        logger.info(f"📥 RAG 질의 요청: {request.question[:50]}...")

        # RAG 시스템에 질의
        result = rag_system.query(
            question=request.question,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )

        # 응답 변환
        response = QueryResponse(**result)
        logger.info("✅ RAG 질의 처리 완료")

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ RAG 질의 처리 실패: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"RAG 질의 처리 중 오류가 발생했습니다: {str(e)}"
        )


@router.get("/health", response_model=HealthResponse)
async def rag_health_check(
    rag_system: RAGSystem = Depends(get_rag_system)
) -> HealthResponse:
    """
    RAG 시스템 헬스체크
    
    RAG 시스템의 상태 및 컴포넌트 가용성을 확인합니다.
    
    Returns:
        HealthResponse: 시스템 상태 정보
    """
    try:
        status_info = rag_system.get_status()

        response = HealthResponse(
            status="healthy" if status_info["initialized"] else "unhealthy",
            initialized=status_info["initialized"],
            llm_available=status_info["llm_available"],
            vector_store_available=status_info["vector_store_available"],
            domain=status_info["domain"],
            version="2.1.0"
        )

        return response

    except Exception as e:
        logger.error(f"❌ RAG 헬스체크 실패: {e}", exc_info=True)
        raise HTTPException(
            status_code=503,
            detail=f"헬스체크 중 오류가 발생했습니다: {str(e)}"
        )

