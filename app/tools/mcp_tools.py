"""
MCP Tools LangChain 래핑
기존 MCP Tools를 LangChain Tool로 변환
"""
import logging
from typing import Any, Dict, Optional, Type
from functools import lru_cache

from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun

from ..mcp.tools.crawler_tool import WebCrawlerTool as MCPWebCrawlerTool
from ..mcp.tools.pdf_tool import PDFParserTool as MCPPdfParserTool
from ..mcp.tools.vecdb_tool import VectorDBTool as MCPVectorDBTool
from ..mcp.tools.mongodb_tool import MongoDBTool as MPMongoDBTool
from ..mcp.config import MCPConfig

logger = logging.getLogger(__name__)


class MCPToolWrapper(BaseTool):
    """
    MCP Tool을 LangChain Tool로 래핑하는 베이스 클래스
    """

    def __init__(self, mcp_tool_instance, tool_name: str, description: str):
        super().__init__(
            name=tool_name,
            description=description,
            callback_manager=None
        )
        self.mcp_tool = mcp_tool_instance

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """
        동기 실행 (LangChain 요구사항)
        """
        try:
            # MCP Tool 실행을 위한 인자 변환
            arguments = self._parse_query_to_args(query)

            # 비동기 함수를 동기로 실행 (임시)
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(self.mcp_tool.execute(arguments))
                return result
            finally:
                loop.close()

        except Exception as e:
            logger.error(f"MCP Tool 실행 실패 ({self.name}): {e}")
            return f"Tool 실행 중 오류 발생: {str(e)}"

    async def _arun(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """
        비동기 실행
        """
        try:
            arguments = self._parse_query_to_args(query)
            result = await self.mcp_tool.execute(arguments)
            return result

        except Exception as e:
            logger.error(f"MCP Tool 비동기 실행 실패 ({self.name}): {e}")
            return f"Tool 실행 중 오류 발생: {str(e)}"

    def _parse_query_to_args(self, query: str) -> Dict[str, Any]:
        """
        쿼리 문자열을 MCP Tool 인자로 변환
        각 서브클래스에서 구현 필요

        Args:
            query: 쿼리 문자열

        Returns:
            MCP Tool 인자 딕셔너리
        """
        return {"query": query}


class WebCrawlerTool(MCPToolWrapper):
    """MCP Web Crawler Tool 래핑"""

    def __init__(self, config: MCPConfig):
        mcp_tool = MCPWebCrawlerTool(config)
        super().__init__(
            mcp_tool,
            "web_crawler",
            "ArXiv 논문 웹 크롤러. 연구 논문을 검색하고 다운로드합니다."
        )

    def _parse_query_to_args(self, query: str) -> Dict[str, Any]:
        """ArXiv 검색 쿼리 파싱"""
        return {
            "query": query,
            "max_results": 10,
            "sort_by": "relevance"
        }


class PDFParserTool(MCPToolWrapper):
    """MCP PDF Parser Tool 래핑"""

    def __init__(self, config: MCPConfig):
        mcp_tool = MCPPdfParserTool(config)
        super().__init__(
            mcp_tool,
            "pdf_parser",
            "PDF 문서 파서. PDF 파일을 텍스트로 변환하고 청킹합니다."
        )

    def _parse_query_to_args(self, query: str) -> Dict[str, Any]:
        """PDF 파싱 파라미터 파싱"""
        # 파일 경로 추출 시도
        if ".pdf" in query:
            file_path = query.split(".pdf")[0] + ".pdf"
        else:
            file_path = query

        return {
            "file_path": file_path,
            "chunk_size": 512,
            "chunk_overlap": 50
        }


class VectorDBTool(MCPToolWrapper):
    """MCP VectorDB Tool 래핑"""

    def __init__(self, config: MCPConfig):
        mcp_tool = MCPVectorDBTool(config)
        super().__init__(
            mcp_tool,
            "vector_db",
            "벡터 데이터베이스 관리. 검색, 저장, 삭제 작업을 수행합니다."
        )

    def _parse_query_to_args(self, query: str) -> Dict[str, Any]:
        """벡터 DB 작업 파싱"""
        # 간단한 파싱 로직 (실제로는 더 정교하게)
        if "search" in query.lower():
            action = "search"
            search_query = query.replace("search", "").strip()
            return {
                "action": action,
                "query": search_query,
                "top_k": 5
            }
        elif "create" in query.lower():
            return {"action": "create"}
        elif "delete" in query.lower():
            return {"action": "delete"}
        else:
            return {"action": "stats"}


class MongoDBTool(MCPToolWrapper):
    """MCP MongoDB Tool 래핑"""

    def __init__(self, config: MCPConfig):
        mcp_tool = MPMongoDBTool(config)
        super().__init__(
            mcp_tool,
            "mongodb",
            "MongoDB 문서 저장소. 문서 저장, 검색, 관리를 수행합니다."
        )

    def _parse_query_to_args(self, query: str) -> Dict[str, Any]:
        """MongoDB 작업 파싱"""
        if "find" in query.lower() or "search" in query.lower():
            return {
                "action": "find",
                "collection": "documents",
                "limit": 10
            }
        elif "insert" in query.lower():
            return {
                "action": "insert",
                "collection": "documents",
                "data": {"content": query}
            }
        else:
            return {
                "action": "stats"
            }


# Tool 팩토리 함수들
def create_web_crawler_tool() -> WebCrawlerTool:
    """Web Crawler Tool 생성"""
    config = MCPConfig()
    return WebCrawlerTool(config)


def create_pdf_parser_tool() -> PDFParserTool:
    """PDF Parser Tool 생성"""
    config = MCPConfig()
    return PDFParserTool(config)


def create_vector_db_tool() -> VectorDBTool:
    """VectorDB Tool 생성"""
    config = MCPConfig()
    return VectorDBTool(config)


def create_mongodb_tool() -> MongoDBTool:
    """MongoDB Tool 생성"""
    config = MCPConfig()
    return MongoDBTool(config)


# 모든 MCP Tool 생성 함수
MCP_TOOL_FACTORIES = {
    "web_crawler": create_web_crawler_tool,
    "pdf_parser": create_pdf_parser_tool,
    "vector_db": create_vector_db_tool,
    "mongodb": create_mongodb_tool
}


def get_all_mcp_tools() -> Dict[str, BaseTool]:
    """
    모든 MCP Tools 생성

    Returns:
        Tool 이름 -> Tool 인스턴스 매핑
    """
    tools = {}
    for name, factory in MCP_TOOL_FACTORIES.items():
        try:
            tools[name] = factory()
            logger.info(f"✅ MCP Tool 생성: {name}")
        except Exception as e:
            logger.error(f"❌ MCP Tool 생성 실패 ({name}): {e}")

    return tools


def register_mcp_tools_to_registry():
    """
    MCP Tools를 Tool Registry에 등록
    """
    from .registry import get_tool_registry

    registry = get_tool_registry()
    tools = get_all_mcp_tools()

    for name, tool in tools.items():
        registry.register_tool(name, type(tool), instantiate=False)
        registry._tools[name] = tool  # 직접 인스턴스 등록

    logger.info(f"📋 MCP Tools 등록 완료: {len(tools)}개")
