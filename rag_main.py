"""
RAG FastAPI + Gradio Application
"""
import os
import sys
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import gradio as gr
from fastapi import FastAPI

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent))

from app.vecdb.embedding_service import EmbeddingService
from app.vecdb.faiss_manager import FaissManager
from app.vecdb.mongodb_client import MongoDBClient
from app.vecdb.local_storage import LocalJSONStorage
from app.vecdb.retriever import RAGRetriever
from app.chat.services.gemini_service import GeminiService
from app.chat.router.rag_router import router as rag_router

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)



logger = logging.getLogger(__name__)




# 전역 RAG 컴포넌트
rag_components = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 생명주기 관리"""
    logger.info("="*60)
    logger.info("RAG 시스템 초기화 중...")
    logger.info("="*60)
    
    try:
        # 1. Embedding Service
        logger.info("1. Embedding Service 초기화...")
        embedding_model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
        embedding_service = EmbeddingService(model_name=embedding_model)
        rag_components['embedding_service'] = embedding_service
        
        # 2. Faiss Manager
        logger.info("2. Faiss 인덱스 로드...")
        
        # 로컬 인덱스 우선 확인
        local_index_path = "data/local_vecdb/faiss.index"
        default_index_path = os.getenv("FAISS_INDEX_PATH", "data/vectordb/faiss.index")
        
        index_path = local_index_path if Path(local_index_path).exists() else default_index_path
        
        if not Path(index_path).exists():
            logger.warning(f"Faiss 인덱스가 없습니다: {index_path}")
            logger.warning("scripts/build_vectordb_local.py를 먼저 실행하세요")
            faiss_manager = FaissManager(dimension=embedding_service.get_dimension())
        else:
            faiss_manager = FaissManager(dimension=embedding_service.get_dimension())
            faiss_manager.load(index_path)
            logger.info(f"✓ Faiss 인덱스 로드 완료: {index_path}")
        
        rag_components['faiss_manager'] = faiss_manager
        
        # 3. Metadata Store (MongoDB or LocalJSON)
        logger.info("3. Metadata Store 초기화...")
        
        use_local = os.getenv("USE_LOCAL_STORAGE", "true").lower() == "true"
        
        if use_local or not Path(local_index_path).exists():
            logger.info("   → Local JSON Storage 사용")
            metadata_store = LocalJSONStorage(storage_dir="data/local_vecdb")
        else:
            logger.info("   → MongoDB 사용")
            mongodb_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
            mongodb_database = os.getenv("MONGODB_DATABASE", "semiconductor_rag")
            mongodb_collection = os.getenv("MONGODB_COLLECTION", "documents")
            metadata_store = MongoDBClient(mongodb_uri, mongodb_database, mongodb_collection)
        
        rag_components['metadata_store'] = metadata_store
        
        # 4. RAG Retriever
        logger.info("4. RAG Retriever 초기화...")
        retriever = RAGRetriever(
            faiss_manager=faiss_manager,
            metadata_store=metadata_store,
            embedding_service=embedding_service
        )
        rag_components['retriever'] = retriever
        
        # 5. LLM Services
        logger.info("5. LLM 서비스 초기화...")
        
        # Gemini (선택사항)
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if google_api_key:
            gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
            gemini_service = GeminiService(api_key=google_api_key, model_name=gemini_model)
            rag_components['gemini_service'] = gemini_service
            logger.info(f"   ✓ Gemini: {gemini_model}")
        else:
            logger.info("   ⓘ GOOGLE_API_KEY 없음 - Gemini 미사용")
        
        # Ollama (기본)
        try:
            from langchain_ollama import OllamaLLM
            ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            ollama_model = os.getenv("MODEL_NAME", "exaone-1.2b:latest")
            
            ollama_service = OllamaLLM(
                model=ollama_model,
                base_url=ollama_base_url,
                temperature=0.3,  # 더 일관된 답변을 위해 낮춤
                num_ctx=4096,  # Context 길이 증가
            )
            rag_components['ollama_service'] = ollama_service
            logger.info(f"   ✓ Ollama: {ollama_model}")
        except Exception as e:
            logger.warning(f"   ⚠ Ollama 초기화 실패: {e}")
        
        logger.info("="*60)
        logger.info("✓ RAG 시스템 초기화 완료!")
        
        stats = retriever.get_stats()
        logger.info(f"통계:")
        logger.info(f"  - 벡터: {stats['faiss']['total_vectors']}개")
        logger.info(f"  - 문서: {stats['metadata_store']['total_documents']}개")
        logger.info(f"  - GPU: {'✓' if stats['faiss']['gpu_enabled'] else '✗'}")
        logger.info("="*60)
        
        yield
        
    except Exception as e:
        logger.error(f"초기화 실패: {e}", exc_info=True)
        raise
    
    finally:
        logger.info("RAG 시스템 종료 중...")
        if 'metadata_store' in rag_components:
            rag_components['metadata_store'].close()
        logger.info("✓ RAG 시스템 종료 완료")


# FastAPI 앱
app = FastAPI(
    title="Semiconductor RAG System",
    version="1.0.0",
    lifespan=lifespan
)

# RAG Router 등록
app.include_router(rag_router, tags=["RAG"])


# Gradio UI
def create_gradio_interface():
    """Gradio 인터페이스 생성"""
    
    def chat_with_rag(message: str, history: list) -> str:
        """RAG 기반 채팅"""
        try:
            if not message or message.strip() == "":
                return "질문을 입력해주세요."
            
            retriever = rag_components.get('retriever')
            gemini_service = rag_components.get('gemini_service')
            ollama_service = rag_components.get('ollama_service')
            
            if not retriever:
                return "❌ RAG 시스템이 초기화되지 않았습니다."
            
            # LLM 우선순위: Ollama(EXAONE) > Gemini
            llm_service = ollama_service if ollama_service else gemini_service
            llm_name = "Ollama (EXAONE)" if ollama_service else "Gemini"
            
            if not llm_service:
                return "❌ LLM 서비스가 없습니다. Ollama를 실행하거나 GOOGLE_API_KEY를 설정하세요."
            
            # 1. 관련 문서 검색
            results = retriever.retrieve(message, top_k=3)
            
            if not results:
                return "관련된 문서를 찾을 수 없습니다. 다른 질문을 시도해보세요."
            
            # 2. Context 구성
            context_parts = []
            for i, result in enumerate(results, 1):
                context_parts.append(f"### 문서 {i}")
                context_parts.append(f"- 출처: {result['metadata']['paper_filename']}")
                context_parts.append(f"- 도메인: {result['metadata']['domain']}")
                context_parts.append(f"- 유사도: {result['score']:.2f}")
                context_parts.append(f"\n**내용:**")
                # 더 많은 context 제공 (500 -> 800자)
                context_parts.append(f"{result['content'][:800]}")
                if len(result['content']) > 800:
                    context_parts.append("...")
                context_parts.append("")
            
            context = "\n".join(context_parts)
            
            # 3. Prompt 구성
            prompt = f"""당신은 반도체 제조 및 설비 분야의 전문 연구원입니다. 
주어진 참고 문서를 기반으로 질문에 대해 **한국어로 명확하고 상세하게** 답변하세요.

## 답변 원칙
1. 반드시 **한국어**로 답변할 것
2. 참고 문서의 내용을 기반으로 답변할 것
3. 기술적 용어는 한글로 설명하되, 필요시 영문 병기 (예: 수율(Yield))
4. 구체적인 방법론이나 기술이 있다면 명확히 설명할 것
5. 문서에 없는 내용은 추측하지 말고 "문서에서 확인할 수 없습니다"라고 답변할 것

## 참고 문서
{context}

## 질문
{message}

## 답변 (한국어로 작성):
"""
            
            # 4. LLM 호출
            if ollama_service:
                # Ollama(EXAONE) 우선 사용
                response = ollama_service.invoke(prompt)
            else:
                # Gemini는 generate 메서드 사용
                response = gemini_service.generate(prompt)
            
            # 5. 출처 추가
            sources = f"\n\n---\n**LLM:** {llm_name}\n**참고 문서:**\n"
            for i, result in enumerate(results, 1):
                sources += f"{i}. {result['metadata']['paper_filename']} (유사도: {result['score']:.2f})\n"
            
            return response + sources
            
        except Exception as e:
            logger.error(f"채팅 오류: {e}", exc_info=True)
            return f"❌ 오류가 발생했습니다: {str(e)}"
    
    # Gradio ChatInterface
    demo = gr.ChatInterface(
        fn=chat_with_rag,
        title="🔬 반도체 RAG 시스템",
        description="""
        반도체 제조 및 설비에 관한 질문을 입력하세요.
        ArXiv 논문과 기술 블로그를 기반으로 답변합니다.
        
        **예시 질문:**
        - What is Virtual Metrology in semiconductor manufacturing?
        - Explain HNSW algorithm for vector search
        - How does defect inspection work in fab?
        """,
        examples=[
            "What is Virtual Metrology?",
            "Explain fab scheduling optimization",
            "How does defect inspection work?",
            "What are the challenges in semiconductor manufacturing?",
        ],
        theme=gr.themes.Soft(),
    )
    
    return demo


# Gradio를 FastAPI에 마운트
gradio_app = create_gradio_interface()
app = gr.mount_gradio_app(app, gradio_app, path="/")


@app.get("/health")
async def health():
    """헬스 체크"""
    return {
        "status": "healthy",
        "rag_initialized": bool(rag_components.get('retriever')),
        "gemini_available": bool(rag_components.get('gemini_service')),
        "ollama_available": bool(rag_components.get('ollama_service')),
        "llm_available": bool(rag_components.get('gemini_service') or rag_components.get('ollama_service')),
    }


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", "8001"))
    
    logger.info(f"🚀 RAG 시스템 시작 중... (Port: {port})")
    logger.info(f"   - API: http://localhost:{port}/docs")
    logger.info(f"   - Gradio UI: http://localhost:{port}/")
    
    uvicorn.run(
        "rag_main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )
