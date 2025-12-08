"""
VirtualFab RAG System - Application Factory

FastAPI 애플리케이션 생성 및 구성
단일 책임: 앱 생성, 의존성 설정, 컴포넌트 통합
"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .core.rag import RAGSystem
from .api.router import router as api_router, set_rag_system

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """
    VirtualFab RAG System FastAPI 애플리케이션 생성

    Returns:
        구성된 FastAPI 애플리케이션
    """

    logger.info("[Initializing...] VirtualFab RAG System 초기화 중...")

    # RAG 시스템 초기화 (코어 비즈니스 로직)
    rag_system = RAGSystem()
    initialization_success = rag_system.initialize()

    if not initialization_success:
        logger.warning("[Failed] RAG 시스템 초기화에 실패했습니다. 기본 기능만 사용할 수 있습니다.")

    # FastAPI 앱 생성
    app = FastAPI(
        title="VirtualFab RAG System",
        description="Virtual Fab RAG System",
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # CORS 미들웨어 추가
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # RAG 시스템을 API 라우터에 주입
    set_rag_system(rag_system)

    # REST API 라우터 등록
    app.include_router(api_router, prefix="/api", tags=["API"])

    # 헬스체크 엔드포인트 (API 라우터 외 추가)
    @app.get("/health", tags=["Health"])
    async def health_check():
        """시스템 헬스체크"""
        return {
            "status": "healthy",
            "system": "VirtualFab RAG",
            "version": "2.0.0",
            "rag_initialized": rag_system.is_initialized
        }

    # Gradio 웹 UI 마운트 (lazy import로 순환 참조 방지)
    import gradio as gr
    from .web.gradio_ui import create_gradio_app
    gradio_app = create_gradio_app(rag_system)
    app = gr.mount_gradio_app(app, gradio_app, path="/")

    logger.info("[Success] VirtualFab RAG System 초기화 완료!")
    logger.info("[API Docs] /docs")
    logger.info("[Gradio UI] /")

    return app