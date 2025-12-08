"""
RAG System Core
단일 책임: VirtualFab RAG 시스템의 전체 오케스트레이션
"""
import logging
from typing import Dict, List, Optional, Any
import os
import sys
from pathlib import Path

# 프로젝트 루트에서 config import
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
try:
    from config import MODEL_NAME
except ImportError:
    MODEL_NAME = None

from .llm import OllamaLLMProvider, OllamaRequest
from .llm.dto import OllamaResponse
from .vector_db import get_vector_db, FAISSVectorDB, initialize_sample_data

logger = logging.getLogger(__name__)


class RAGSystem:
    """
    VirtualFab RAG System
    - 단일 책임: 전체 RAG 파이프라인 오케스트레이션
    - 반도체 제조(VirtualFab/Digital Twin) 도메인 특화
    """

    def __init__(self):
        self.llm_provider: Optional[OllamaLLMProvider] = None
        self.vector_store: Optional[FAISSVectorDB] = None
        self.crawler = None      # 추후 구현
        self.retriever = None    # 추후 구현

        self.is_initialized = False

        logger.info("🎯 RAG System 초기화 중...")

    def initialize(self) -> bool:
        """
        RAG 시스템 초기화

        Returns:
            초기화 성공 여부
        """
        try:
            # LLM Provider 초기화
            # 환경변수 우선순위: OLLAMA_MODEL > MODEL_NAME (config.py) > 기본값
            ollama_model = (
                os.getenv("OLLAMA_MODEL") or 
                (MODEL_NAME if MODEL_NAME else None) or 
                "exaone-1.2b:latest"
            )
            logger.info(f"🤖 사용할 모델: {ollama_model}")
            self.llm_provider = OllamaLLMProvider(model_name=ollama_model)

            # LLM 사용 가능 여부 확인
            if not self.llm_provider.is_available():
                logger.warning(f"⚠️ Ollama LLM 모델 '{ollama_model}'이 사용 불가능합니다")
                logger.warning("⚠️ 사용 가능한 모델을 확인하세요: ollama list")
                return False

            # VectorDB 초기화
            self.vector_store = get_vector_db()
            if not self.vector_store.initialize():
                logger.warning("⚠️ VectorDB 초기화 실패. 검색 기능이 제한될 수 있습니다.")
                # return False  # VectorDB 없어도 기본 기능은 동작 가능

            # 샘플 데이터 초기화 (첫 실행시)
            try:
                initialize_sample_data()
                logger.info("✅ 샘플 데이터 초기화 완료")
            except Exception as e:
                logger.warning(f"⚠️ 샘플 데이터 초기화 실패: {e}")

            self.is_initialized = True
            logger.info("✅ RAG System 초기화 완료")
            return True

        except Exception as e:
            logger.error(f"❌ RAG System 초기화 실패: {e}")
            return False

    def query(self, question: str, **kwargs) -> Dict[str, Any]:
        """
        RAG 질의 처리

        Args:
            question: 사용자 질문
            **kwargs: 추가 파라미터들

        Returns:
            응답 딕셔너리
        """
        if not self.is_initialized:
            return {
                "answer": "시스템이 초기화되지 않았습니다.",
                "sources": [],
                "metadata": {"error": "not_initialized"}
            }

        try:
            logger.info(f"📥 RAG 질의: {question}")

            # 1. 벡터 검색 수행
            search_results = []
            if self.vector_store:
                search_results = self.vector_store.similarity_search(
                    query=question,
                    k=kwargs.get('top_k', 3),
                    score_threshold=kwargs.get('score_threshold', 0.0)
                )
                logger.info(f"🔍 검색 결과: {len(search_results)}개 문서")

            # 2. 검색 컨텍스트 구성
            context = ""
            sources = []
            if search_results:
                context_parts = []
                for doc, score in search_results:
                    context_parts.append(doc.page_content)
                    sources.append({
                        "content": doc.page_content,
                        "source": doc.metadata.get("source", "unknown"),
                        "score": float(score),
                        "chunk_id": doc.metadata.get("chunk_id", 0)
                    })
                context = "\n\n".join(context_parts)
            else:
                logger.warning("⚠️ 검색 결과가 없습니다. 일반 LLM 응답을 생성합니다.")

            # 3. RAG 프롬프트 구성
            rag_prompt = self._get_rag_prompt(question, context)

            # 4. LLM 호출
            request = OllamaRequest(
                prompt=rag_prompt,
                system_prompt=self._get_system_prompt(),
                temperature=kwargs.get('temperature', 0.1)
            )

            response = self.llm_provider.generate_response(request)

            return {
                "answer": response.response,
                "sources": sources,
                "metadata": {
                    "llm_provider": "ollama",
                    "model": response.model_name,
                    "rag_enabled": len(sources) > 0,
                    "search_results_count": len(sources),
                    "context_length": len(context)
                }
            }

        except Exception as e:
            logger.error(f"❌ RAG 질의 처리 실패: {e}")
            return {
                "answer": f"오류가 발생했습니다: {str(e)}",
                "sources": [],
                "metadata": {"error": str(e)}
            }

    def _get_system_prompt(self) -> str:
        """
        시스템 프롬프트 생성
        VirtualFab 도메인 특화

        Returns:
            시스템 프롬프트
        """
        return """당신은 반도체 제조(VirtualFab/Digital Twin) 도메인의 전문 AI 어시스턴트입니다.

전문 분야:
- 반도체 공정 (8대 공정, Lithography, Etching 등)
- Virtual Metrology
- Digital Twin
- Predictive Maintenance
- Process Optimization
- Yield Management

응답 원칙:
1. 정확하고 전문적인 답변 제공
2. 한국어로 자연스럽게 설명
3. 필요한 경우 예시나 추가 설명 포함
4. 모르는 내용은 솔직히 밝히고 추정하지 않음

질문에 성실하고 도움이 되는 답변을 제공하세요."""

    def _get_rag_prompt(self, question: str, context: str) -> str:
        """
        RAG 프롬프트 생성
        검색된 컨텍스트를 활용한 질문 답변

        Args:
            question: 사용자 질문
            context: 검색된 컨텍스트

        Returns:
            RAG 프롬프트
        """
        if context:
            return f"""다음은 검색된 관련 정보입니다:

{context}

위 정보를 참고하여 다음 질문에 답변해주세요:

질문: {question}

답변:"""
        else:
            return f"질문: {question}\n\n답변:"

    def get_status(self) -> Dict[str, Any]:
        """
        시스템 상태 조회

        Returns:
            시스템 상태 정보
        """
        return {
            "initialized": self.is_initialized,
            "llm_available": self.llm_provider.is_available() if self.llm_provider else False,
            "vector_store_available": self.vector_store is not None,
            "crawler_available": self.crawler is not None,
            "retriever_available": self.retriever is not None,
            "domain": "VirtualFab/Digital Twin"
        }