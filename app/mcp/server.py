"""
VirtualFab RAG System MCP Server
MCP 프로토콜을 통해 AI Tool들을 노출
"""
import asyncio
import logging
from typing import Any, Dict, List, Optional
from pathlib import Path

from mcp.server import Server
from mcp.types import (
    Resource,
    Tool,
    TextContent,
    LoggingLevel
)
from mcp.server.stdio import stdio_server

from .config import MCPConfig
from .tools.crawler_tool import WebCrawlerTool
from .tools.pdf_tool import PDFParserTool
from .tools.vecdb_tool import VectorDBTool
from .tools.mongodb_tool import MongoDBTool

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NewEraMCPServer:
    """VirtualFab RAG System MCP Server"""

    def __init__(self):
        self.config = MCPConfig()
        self.server_config = self.config.get_server_config()

        # Tool 인스턴스들
        self.tools = {
            "web_crawler": WebCrawlerTool(self.config),
            "pdf_parser": PDFParserTool(self.config),
            "vector_db": VectorDBTool(self.config),
            "mongodb": MongoDBTool(self.config)
        }

        logger.info("🎯 NewEra MCP Server 초기화 중...")

    async def list_tools(self) -> List[Tool]:
        """사용 가능한 MCP Tool들 목록 반환"""
        tools = []

        # 각 Tool이 활성화되어 있으면 추가
        for tool_name, tool_instance in self.tools.items():
            if self.server_config["tools"][tool_name]["enabled"]:
                tool_schema = tool_instance.get_tool_schema()
                tools.append(Tool(**tool_schema))

        logger.info(f"✅ {len(tools)}개 MCP Tool 로드됨")
        return tools

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> List[TextContent]:
        """MCP Tool 호출 처리"""
        try:
            logger.info(f"🔧 MCP Tool 호출: {name}")

            if name not in self.tools:
                return [TextContent(
                    type="text",
                    text=f"❌ Tool '{name}'을 찾을 수 없습니다."
                )]

            tool = self.tools[name]
            result = await tool.execute(arguments)

            return [TextContent(
                type="text",
                text=result
            )]

        except Exception as e:
            logger.error(f"❌ Tool 실행 실패: {e}")
            return [TextContent(
                type="text",
                text=f"❌ Tool 실행 중 오류 발생: {str(e)}"
            )]

    async def list_resources(self) -> List[Resource]:
        """MCP 리소스 목록 반환"""
        resources = []

        # 데이터 디렉토리 구조를 리소스로 노출
        data_dir = self.config.data_dir
        if data_dir.exists():
            for item in data_dir.rglob("*"):
                if item.is_file():
                    relative_path = item.relative_to(self.config.base_dir)
                    resources.append(Resource(
                        uri=f"file://{relative_path}",
                        name=str(relative_path),
                        description=f"Data file: {relative_path}",
                        mimeType="application/octet-stream"
                    ))

        return resources

    async def read_resource(self, uri: str) -> str:
        """리소스 내용 읽기"""
        try:
            if uri.startswith("file://"):
                file_path = self.config.base_dir / uri[7:]  # "file://" 제거
                if file_path.exists():
                    return file_path.read_text()
                else:
                    return f"파일을 찾을 수 없습니다: {file_path}"
            else:
                return f"지원하지 않는 URI 형식: {uri}"
        except Exception as e:
            return f"리소스 읽기 실패: {str(e)}"


async def serve():
    """MCP 서버 실행"""
    server = NewEraMCPServer()

    async with stdio_server() as (read_stream, write_stream):
        await Server(
            {
                "list_tools": server.list_tools,
                "call_tool": server.call_tool,
                "list_resources": server.list_resources,
                "read_resource": server.read_resource,
            },
            {
                "server_info": {
                    "name": server.server_config["name"],
                    "version": server.server_config["version"],
                }
            }
        ).run(
            read_stream,
            write_stream,
            None  # options
        )


if __name__ == "__main__":
    asyncio.run(serve())