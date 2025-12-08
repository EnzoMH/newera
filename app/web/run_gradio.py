"""
Gradio 독립 실행 스크립트
FastAPI 없이 Gradio만 실행할 때 사용
"""
import logging
import sys
from pathlib import Path

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.core.rag import RAGSystem
from app.web.gradio_ui import create_gradio_app

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Gradio 앱 독립 실행"""
    logger.info("VirtualFab RAG System - Gradio UI 시작 중...")
    
    # RAG 시스템 초기화
    rag_system = RAGSystem()
    initialization_success = rag_system.initialize()
    
    if not initialization_success:
        logger.warning("RAG 시스템 초기화 실패. 빈 문서로 시작합니다.")
    
    # Gradio 앱 생성
    demo = create_gradio_app(rag_system)
    
    # 실행
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )


if __name__ == "__main__":
    main()