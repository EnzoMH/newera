"""
VirtualFab RAG System - Main Entry Point
FastAPI 서버 실행 및 포트 자동 할당
"""
import os
import sys
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from dotenv import load_dotenv

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent))

from app.core.rag import RAGSystem
from app.api.dependencies import set_rag_system
from app.core.utils.port import get_port_from_env

# .env 파일 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    애플리케이션 생명주기 관리
    startup과 shutdown 이벤트 처리
    """
    # Startup
    logger.info("=" * 60)
    logger.info("VirtualFab RAG System 시작 중...")
    logger.info("=" * 60)
    
    try:
        # RAG 시스템 초기화
        rag_system = RAGSystem()
        initialization_success = rag_system.initialize()
        
        if initialization_success:
            logger.info("✅ RAG 시스템 초기화 완료")
        else:
            logger.warning("⚠️ RAG 시스템 초기화 실패. 기본 기능만 사용 가능합니다.")
        
        # 의존성 주입
        set_rag_system(rag_system)
        logger.info("✅ 의존성 주입 완료")
        
    except Exception as e:
        logger.error(f"❌ 시스템 초기화 중 오류 발생: {e}", exc_info=True)
        raise
    
    logger.info("=" * 60)
    
    yield
    
    # Shutdown
    logger.info("=" * 60)
    logger.info("VirtualFab RAG System 종료 중...")
    logger.info("=" * 60)


def create_app() -> FastAPI:
    """
    FastAPI 애플리케이션 생성
    
    Returns:
        구성된 FastAPI 애플리케이션
    """
    # CORS 설정 (환경변수 기반)
    cors_origins = os.getenv("CORS_ORIGINS", "*")
    if cors_origins != "*":
        cors_origins = [origin.strip() for origin in cors_origins.split(",")]
    
    # FastAPI 앱 생성
    app = FastAPI(
        title="VirtualFab RAG System",
        description="LangGraph + LangChain + RAG 기반 반도체 제조 도메인 AI 시스템",
        version="2.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan
    )
    
    # CORS 미들웨어 추가
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # 로깅 미들웨어 추가
    from app.api.middleware import LoggingMiddleware
    app.add_middleware(LoggingMiddleware)
    
    # 에러 핸들러 미들웨어 추가
    from app.api.middleware import ErrorHandlerMiddleware
    app.add_middleware(ErrorHandlerMiddleware)
    
    # API 라우터 등록
    from app.api.router import router as api_router
    app.include_router(api_router, prefix="/api")
    
    logger.info("✅ API 라우터 등록 완료: /api/rag, /api/health, /api/system")
    
    # 루트 엔드포인트
    @app.get("/", tags=["Root"])
    async def root():
        """루트 엔드포인트"""
        return {
            "status": "healthy",
            "message": "VirtualFab RAG System API",
            "version": "2.1.0",
            "docs": "/docs",
            "health": "/health"
        }
    
    # 헬스체크 엔드포인트 (RAG 시스템 없이도 동작)
    @app.get("/health", tags=["Health"])
    async def health_check():
        """시스템 헬스체크"""
        from app.api.dependencies import check_rag_initialized
        
        return {
            "status": "healthy",
            "system": "VirtualFab RAG System",
            "version": "2.1.0",
            "rag_initialized": check_rag_initialized()
        }
    
    return app


def main():
    """메인 실행 함수"""
    try:
        # 포트 자동 할당
        port = get_port_from_env(default=8000)
        host = os.getenv("API_HOST", "0.0.0.0")
        
        logger.info(f"🌐 서버 시작: http://{host}:{port}")
        logger.info(f"📚 API 문서: http://{host}:{port}/docs")
        logger.info(f"💚 헬스체크: http://{host}:{port}/health")
        
        # FastAPI 앱 생성
        app = create_app()
        
        # Uvicorn 서버 실행
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level="info",
            reload=os.getenv("RELOAD", "false").lower() == "true"
        )
        
    except KeyboardInterrupt:
        logger.info("서버가 사용자에 의해 중지되었습니다.")
    except Exception as e:
        logger.error(f"서버 실행 중 오류 발생: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
