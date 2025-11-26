"""
RAG 오케스트레이션 서비스
"""
import logging
from typing import Any

from app.vecdb.retriever import RAGRetriever
from app.chat.services.gemini_service import GeminiService


logger = logging.getLogger(__name__)


class RAGService:
    """
    RAG 시스템 오케스트레이션
    - Retriever: 문서 검색
    - LLM: 답변 생성 (Gemini or Ollama)
    """
    
    def __init__(
        self,
        retriever: RAGRetriever,
        gemini_service: GeminiService | None = None,
        ollama_service: Any | None = None
    ):
        """
        Args:
            retriever: RAG Retriever
            gemini_service: Gemini 서비스 (옵션)
            ollama_service: Ollama 서비스 (옵션)
        """
        self.retriever = retriever
        self.gemini_service = gemini_service
        self.ollama_service = ollama_service
        
        logger.info("✓ RAG Service 초기화 완료")
    
    def query(
        self,
        question: str,
        top_k: int = 5,
        use_gemini: bool = False,
        filter_domain: str | None = None
    ) -> dict[str, Any]:
        """
        RAG 쿼리 실행
        
        Args:
            question: 사용자 질문
            top_k: 검색할 문서 수
            use_gemini: Gemini 사용 여부 (False면 Ollama)
            filter_domain: 도메인 필터
            
        Returns:
            answer: 답변
            sources: 참조 문서
            metadata: 메타정보
        """
        logger.info(f"RAG 쿼리: '{question}'")
        
        search_results = self.retriever.retrieve(question, top_k, filter_domain)
        
        if not search_results:
            return {
                "answer": "관련 문서를 찾을 수 없습니다. 다른 질문을 시도해주세요.",
                "sources": [],
                "metadata": {"retrieved_docs": 0}
            }
        
        context = self._build_context(search_results)
        
        system_prompt = """당신은 반도체 제조 공정 및 Virtual Fab/Digital Twin/Virtual Metrology 전문가입니다.
주어진 문서를 바탕으로 정확하고 전문적인 답변을 제공하세요.
답변은 한국어로 작성해주세요."""
        
        if use_gemini and self.gemini_service:
            answer = self.gemini_service.generate(question, context, system_prompt)
        elif self.ollama_service:
            answer = self._generate_with_ollama(question, context)
        else:
            answer = "LLM 서비스가 설정되지 않았습니다."
        
        return {
            "answer": answer,
            "sources": [
                {
                    "content": result['content'][:200] + "...",
                    "filename": result['metadata']['paper_filename'],
                    "domain": result['metadata']['domain'],
                    "score": result['score'],
                }
                for result in search_results
            ],
            "metadata": {
                "retrieved_docs": len(search_results),
                "filter_domain": filter_domain,
                "llm_provider": "gemini" if use_gemini else "ollama",
            }
        }
    
    def _build_context(self, search_results: list[dict[str, Any]]) -> str:
        """검색 결과로 컨텍스트 구성"""
        context_parts = []
        
        for i, result in enumerate(search_results, 1):
            context_parts.append(f"""[문서 {i}]
출처: {result['metadata']['paper_filename']}
도메인: {result['metadata']['domain']}
내용:
{result['content']}

""")
        
        return "\n".join(context_parts)
    
    def _generate_with_ollama(self, question: str, context: str) -> str:
        """Ollama로 답변 생성"""
        if not self.ollama_service:
            return "Ollama 서비스가 설정되지 않았습니다."
        
        prompt = f"""다음은 반도체 제조 관련 문서 정보입니다:

{context}

질문: {question}

위 정보를 바탕으로 답변해주세요. 답변은 한국어로 작성해주세요."""
        
        try:
            response = self.ollama_service.invoke(prompt)
            return response
        except Exception as e:
            logger.error(f"Ollama 생성 실패: {e}")
            return f"오류가 발생했습니다: {str(e)}"

