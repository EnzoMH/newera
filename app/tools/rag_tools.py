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
            logger.info(f"🔍 실제 벡터 검색: {query}")

            # 실제 VectorDB 사용
            from ..core.vector_db import get_vector_db
            
            vector_db = get_vector_db()
            results = vector_db.similarity_search(query, k=5)

            if not results:
                return "검색 결과가 없습니다. VectorDB에 문서가 추가되지 않았을 수 있습니다."

            result_text = f"벡터 검색 결과 ({len(results)}개):\n"
            for i, result in enumerate(results, 1):
                content = result.get('content', result.get('page_content', ''))[:150] + "..."
                score = result.get('score', 'N/A')
                source = result.get('metadata', {}).get('source', 'Unknown')
                result_text += f"{i}. [{source}] {content} (유사도: {score})\n"

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

            # LangChain RecursiveCharacterTextSplitter 사용
            from langchain.text_splitter import RecursiveCharacterTextSplitter

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ". ", " ", ""],
                length_function=len,
            )

            # 파일 경로인지 확인 후 읽기
            content = input_text
            if input_text.endswith(('.txt', '.md', '.py', '.js', '.json', '.pdf')):
                if input_text.endswith('.pdf'):
                    # PDF는 별도 처리 필요
                    return "PDF 파일은 PDF Tool을 사용하세요."
                try:
                    with open(input_text, 'r', encoding='utf-8') as f:
                        content = f.read()
                except Exception as e:
                    return f"파일 읽기 실패: {e}"

            chunks = text_splitter.split_text(content)

            result = f"LangChain 문서 청킹 완료: {len(chunks)}개 청크 생성\n"
            result += f"- 원본 길이: {len(content):,} 문자\n"
            result += f"- 청크 크기: 최대 1000자 (오버랩 200자)\n\n"

            for i, chunk in enumerate(chunks[:3], 1):  # 처음 3개만 표시
                preview = chunk[:100] + "..." if len(chunk) > 100 else chunk
                result += f"청크 {i}: {preview}\n\n"

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


class PDFProcessorTool(BaseTool):
    """
    PDF 처리 도구 (LangChain document parser 기반)
    PDF 파일을 텍스트로 변환하고 청킹
    """

    name = "pdf_processor"
    description = "PDF 파일을 처리하여 텍스트를 추출하고 청킹합니다. 파일 경로를 입력하세요."

    def _run(self, file_path: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """
        PDF 처리 실행
        """
        try:
            logger.info(f"📄 PDF 처리 시작: {file_path}")

            # LangChain PDF 로더들
            from langchain_community.document_loaders import PyPDFLoader
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            from pathlib import Path

            # 파일 존재 확인
            pdf_path = Path(file_path)
            if not pdf_path.exists():
                return f"파일이 존재하지 않습니다: {file_path}"

            if not pdf_path.suffix.lower() == '.pdf':
                return "PDF 파일만 처리 가능합니다."

            # PDF 로더 사용
            loader = PyPDFLoader(str(pdf_path))
            documents = loader.load()

            if not documents:
                return "PDF에서 텍스트를 추출할 수 없습니다."

            # 텍스트 청킹
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            chunks = text_splitter.split_documents(documents)

            # 결과 생성
            total_pages = len(documents)
            total_chunks = len(chunks)
            total_chars = sum(len(doc.page_content) for doc in documents)

            result = f"PDF 처리 완료: {pdf_path.name}\n"
            result += f"- 총 페이지: {total_pages}페이지\n"
            result += f"- 추출 텍스트: {total_chars:,} 문자\n"
            result += f"- 생성 청크: {total_chunks}개\n\n"

            # 샘플 청크
            if chunks:
                sample = chunks[0].page_content[:200] + "..." if len(chunks[0].page_content) > 200 else chunks[0].page_content
                result += f"샘플 청크:\n{sample}\n"

            return result

        except Exception as e:
            logger.error(f"PDF 처리 실패: {e}")
            return f"PDF 처리 중 오류 발생: {str(e)}"

    async def _arun(self, file_path: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """
        PDF 처리 비동기 실행
        """
        return self._run(file_path, run_manager)


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

