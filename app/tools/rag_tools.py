"""
RAG 관련 Tools
벡터 검색, 문서 처리 등 RAG 전용 Tools
"""
import logging
from typing import Any, Dict, Optional
from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun

logger = logging.getLogger(__name__)


class VectorSearchTool(BaseTool):
    """
    벡터 검색 Tool
    FAISS 또는 Chroma를 통한 유사도 검색
    """

    name = "vector_search"
    description = "벡터 데이터베이스에서 유사한 문서를 검색합니다. 질문이나 키워드를 입력하세요."

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """
        동기 검색 실행
        """
        try:
            logger.info(f"🔍 벡터 검색: {query}")

            # TODO: 실제 벡터 검색 구현
            # 현재는 Dummy 결과 반환

            dummy_results = [
                {"id": "doc_001", "content": "반도체 제조 공정에 대한 문서", "score": 0.95},
                {"id": "doc_002", "content": "Digital Twin 기술 설명", "score": 0.89},
                {"id": "doc_003", "content": "Virtual Metrology 적용 사례", "score": 0.87}
            ]

            result_text = f"벡터 검색 결과 ({len(dummy_results)}개):\n"
            for i, result in enumerate(dummy_results, 1):
                result_text += f"{i}. {result['content']} (유사도: {result['score']})\n"

            return result_text

        except Exception as e:
            logger.error(f"벡터 검색 실패: {e}")
            return f"검색 중 오류 발생: {str(e)}"

    async def _arun(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """
        비동기 검색 실행
        """
        # 현재는 동기와 동일한 로직
        return self._run(query, run_manager)


class DocumentChunkerTool(BaseTool):
    """
    문서 청킹 Tool
    긴 텍스트를 적절한 크기로 분할
    """

    name = "document_chunker"
    description = "긴 문서를 지정된 크기로 청킹합니다. 문서 내용과 청크 크기를 입력하세요."

    def _run(self, input_text: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """
        동기 청킹 실행
        """
        try:
            logger.info("📄 문서 청킹 시작")

            # 기본 청크 크기
            chunk_size = 512
            overlap = 50

            # 간단한 텍스트 청킹 (실제로는 더 정교한 알고리즘)
            words = input_text.split()
            chunks = []

            i = 0
            while i < len(words):
                chunk_words = words[i:i + chunk_size]
                chunk_text = " ".join(chunk_words)
                chunks.append(chunk_text)

                # 오버랩만큼 이동
                i += chunk_size - overlap
                if i <= 0:
                    break

            result = f"문서 청킹 완료: {len(chunks)}개 청크 생성\n\n"
            for i, chunk in enumerate(chunks[:3], 1):  # 처음 3개만 표시
                result += f"청크 {i}: {chunk[:100]}...\n\n"

            if len(chunks) > 3:
                result += f"... 외 {len(chunks) - 3}개 청크"

            return result

        except Exception as e:
            logger.error(f"문서 청킹 실패: {e}")
            return f"청킹 중 오류 발생: {str(e)}"

    async def _arun(self, input_text: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """
        비동기 청킹 실행
        """
        return self._run(input_text, run_manager)


class ContextRetrieverTool(BaseTool):
    """
    컨텍스트 검색 Tool
    여러 소스에서 관련 컨텍스트를 수집
    """

    name = "context_retriever"
    description = "질문과 관련된 컨텍스트를 다양한 소스에서 검색합니다."

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """
        동기 컨텍스트 검색
        """
        try:
            logger.info(f"🔍 컨텍스트 검색: {query}")

            # TODO: 실제 컨텍스트 검색 구현
            # 벡터 검색 + 메모리 검색 등 통합

            dummy_contexts = [
                "반도체 제조 공정은 8개의 주요 단계로 구성됩니다.",
                "Digital Twin은 물리적 시스템의 가상 복제본입니다.",
                "Virtual Metrology는 측정 데이터를 예측하는 기술입니다."
            ]

            result = f"관련 컨텍스트 검색 결과:\n\n"
            for i, context in enumerate(dummy_contexts, 1):
                result += f"{i}. {context}\n"

            return result

        except Exception as e:
            logger.error(f"컨텍스트 검색 실패: {e}")
            return f"컨텍스트 검색 중 오류 발생: {str(e)}"

    async def _arun(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """
        비동기 컨텍스트 검색
        """
        return self._run(query, run_manager)


class MemoryAccessTool(BaseTool):
    """
    메모리 접근 Tool
    LangChain Memory에서 대화 히스토리 조회
    """

    name = "memory_access"
    description = "대화 메모리에서 이전 대화 내용을 검색합니다."

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """
        동기 메모리 접근
        """
        try:
            logger.info("🧠 메모리 접근")

            from ..memory import get_conversation_memory

            # 기본 메모리 인스턴스
            memory = get_conversation_memory()

            # 메모리 변수 로드
            memory_vars = memory.load_memory_variables({})

            history = memory_vars.get("history", "")

            if history:
                return f"대화 히스토리:\n{history}"
            else:
                return "저장된 대화 히스토리가 없습니다."

        except Exception as e:
            logger.error(f"메모리 접근 실패: {e}")
            return f"메모리 접근 중 오류 발생: {str(e)}"

    async def _arun(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """
        비동기 메모리 접근
        """
        return self._run(query, run_manager)


# RAG Tool 팩토리 함수들
def create_vector_search_tool() -> VectorSearchTool:
    """벡터 검색 Tool 생성"""
    return VectorSearchTool()


def create_document_chunker_tool() -> DocumentChunkerTool:
    """문서 청킹 Tool 생성"""
    return DocumentChunkerTool()


def create_context_retriever_tool() -> ContextRetrieverTool:
    """컨텍스트 검색 Tool 생성"""
    return ContextRetrieverTool()


def create_memory_access_tool() -> MemoryAccessTool:
    """메모리 접근 Tool 생성"""
    return MemoryAccessTool()


# 모든 RAG Tool 생성 함수
RAG_TOOL_FACTORIES = {
    "vector_search": create_vector_search_tool,
    "document_chunker": create_document_chunker_tool,
    "context_retriever": create_context_retriever_tool,
    "memory_access": create_memory_access_tool
}


def get_all_rag_tools() -> Dict[str, BaseTool]:
    """
    모든 RAG Tools 생성

    Returns:
        Tool 이름 -> Tool 인스턴스 매핑
    """
    tools = {}
    for name, factory in RAG_TOOL_FACTORIES.items():
        try:
            tools[name] = factory()
            logger.info(f"✅ RAG Tool 생성: {name}")
        except Exception as e:
            logger.error(f"❌ RAG Tool 생성 실패 ({name}): {e}")

    return tools


def register_rag_tools_to_registry():
    """
    RAG Tools를 Tool Registry에 등록
    """
    from .registry import get_tool_registry

    registry = get_tool_registry()
    tools = get_all_rag_tools()

    for name, tool in tools.items():
        registry.register_tool(name, type(tool), instantiate=False)
        registry._tools[name] = tool  # 직접 인스턴스 등록

    logger.info(f"📋 RAG Tools 등록 완료: {len(tools)}개")
