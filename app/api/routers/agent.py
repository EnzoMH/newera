"""
Agent API Router
LangGraph Agent 실행을 위한 REST API
"""
import logging
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse

from ..schemas import AgentQueryRequest, AgentQueryResponse
from ..dependencies import get_rag_agent_dependency
from ...agents import get_rag_agent

logger = logging.getLogger(__name__)

# 라우터 생성
router = APIRouter(
    prefix="/agent",
    tags=["Agent"],
    responses={
        404: {"description": "엔드포인트를 찾을 수 없습니다"},
        500: {"description": "서버 내부 오류"}
    }
)


@router.post("/query", response_model=AgentQueryResponse)
async def query_agent(
    request: AgentQueryRequest,
    agent = Depends(get_rag_agent_dependency)
) -> AgentQueryResponse:
    """
    LangGraph Agent 질의 처리

    Agent가 LangGraph 워크플로우를 통해 질의를 처리합니다.
    RAG 검색, 메모리 활용, Tool 사용 등이 포함됩니다.

    - **question**: 사용자 질문 (필수, 1-1000자)
    - **conversation_id**: 대화 ID (선택, 없으면 자동 생성)
    - **use_memory**: 메모리 사용 여부 (기본값: true)
    - **temperature**: 응답 다양성 (0.0-1.0, 기본값: 0.1)

    Returns:
        AgentQueryResponse: Agent 처리 결과
    """
    try:
        logger.info(f"🤖 Agent 질의 요청: {request.question[:50]}...")

        # Agent 실행 (이미 의존성 주입으로 agent 파라미터로 전달됨)
        result = agent.process_query(
            query=request.question,
            conversation_id=request.conversation_id,
            use_memory=request.use_memory,
            temperature=request.temperature
        )

        # 응답 변환
        response = AgentQueryResponse(**result)
        logger.info("✅ Agent 질의 처리 완료")

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Agent 질의 처리 실패: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Agent 질의 처리 중 오류가 발생했습니다: {str(e)}"
        )


@router.get("/status")
async def agent_status(
    agent = Depends(get_rag_agent_dependency)
) -> JSONResponse:
    """
    Agent 상태 조회

    LangGraph Agent의 현재 상태를 반환합니다.

    Returns:
        JSONResponse: Agent 상태 정보
    """
    try:
        status_info = agent.get_status()
        return JSONResponse(content=status_info)

    except Exception as e:
        logger.error(f"❌ Agent 상태 조회 실패: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Agent 상태 조회 중 오류가 발생했습니다: {str(e)}"
        )


@router.post("/memory/clear")
async def clear_agent_memory() -> JSONResponse:
    """
    Agent 메모리 클리어

    Agent의 대화 메모리를 초기화합니다.

    Returns:
        JSONResponse: 클리어 결과
    """
    try:
        from ...memory.conversation import clear_all_memories

        clear_all_memories()

        return JSONResponse(content={
            "status": "success",
            "message": "Agent 메모리가 클리어되었습니다."
        })

    except Exception as e:
        logger.error(f"❌ 메모리 클리어 실패: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"메모리 클리어 중 오류가 발생했습니다: {str(e)}"
        )


@router.get("/tools")
async def list_agent_tools() -> JSONResponse:
    """
    사용 가능한 Agent Tools 목록

    Agent가 사용할 수 있는 Tool들을 반환합니다.

    Returns:
        JSONResponse: Tool 목록
    """
    try:
        from ...tools.registry import get_tool_registry

        registry = get_tool_registry()
        tools_info = registry.get_registry_stats()

        return JSONResponse(content=tools_info)

    except Exception as e:
        logger.error(f"❌ Tool 목록 조회 실패: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Tool 목록 조회 중 오류가 발생했습니다: {str(e)}"
        )
