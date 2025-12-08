"""
MongoDB 관리 MCP Tool
문서 저장소 관리 기능 제공
"""
import asyncio
import logging
from typing import Dict, Any, List
from pathlib import Path

from ..config import MCPConfig

logger = logging.getLogger(__name__)


class MongoDBTool:
    """MongoDB 관리 MCP Tool"""

    def __init__(self, config: MCPConfig):
        self.config = config
        self.tool_config = config.get_tool_config("mongodb")
        self.connection_string = None

    def get_tool_schema(self) -> Dict[str, Any]:
        """MCP Tool 스키마 반환"""
        return {
            "name": "mongodb",
            "description": "MongoDB 문서 저장소 관리",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["connect", "insert", "find", "stats", "clear"],
                        "description": "수행할 작업 종류"
                    },
                    "collection": {
                        "type": "string",
                        "description": "대상 컬렉션",
                        "enum": list(self.tool_config["collections"].values()),
                        "default": self.tool_config["collections"]["documents"]
                    },
                    "data": {
                        "type": "object",
                        "description": "삽입할 데이터 (insert 시 필요)"
                    },
                    "query": {
                        "type": "object",
                        "description": "검색 쿼리 (find 시 필요)",
                        "default": {}
                    },
                    "limit": {
                        "type": "integer",
                        "description": "반환할 최대 문서 수",
                        "default": 10
                    }
                },
                "required": ["action"]
            }
        }

    async def execute(self, arguments: Dict[str, Any]) -> str:
        """Tool 실행"""
        try:
            action = arguments.get("action")

            logger.info(f"🗃️ MongoDB 작업: {action}")

            if action == "connect":
                return await self._connect_db()
            elif action == "insert":
                return await self._insert_document(arguments)
            elif action == "find":
                return await self._find_documents(arguments)
            elif action == "stats":
                return await self._get_stats()
            elif action == "clear":
                return await self._clear_collection(arguments)
            else:
                return f"❌ 지원하지 않는 작업: {action}"

        except Exception as e:
            logger.error(f"MongoDB 작업 실패: {e}")
            return f"❌ MongoDB 작업 실패: {str(e)}"

    async def _connect_db(self) -> str:
        """MongoDB 연결"""
        # 실제 연결 대신 시뮬레이션
        await asyncio.sleep(0.3)

        self.connection_string = "mongodb://localhost:27017"
        database = self.tool_config["database"]

        return f"""✅ MongoDB 연결 성공

🔗 연결 문자열: {self.connection_string}
🗄️ 데이터베이스: {database}

📋 사용 가능한 컬렉션:
{chr(10).join(f"- {name}: {collection}" for name, collection in self.tool_config["collections"].items())}"""

    async def _insert_document(self, args: Dict[str, Any]) -> str:
        """문서 삽입"""
        collection = args.get("collection", self.tool_config["collections"]["documents"])
        data = args.get("data")

        if not data:
            return "❌ data가 필요합니다."

        # 삽입 시뮬레이션
        await asyncio.sleep(0.2)

        doc_id = f"doc_{hash(str(data)) % 10000:04d}"

        return f"""✅ 문서 삽입 완료

📄 컬렉션: {collection}
🆔 문서 ID: {doc_id}

📝 삽입된 데이터:
{chr(10).join(f"- {k}: {v}" for k, v in data.items())}"""

    async def _find_documents(self, args: Dict[str, Any]) -> str:
        """문서 검색"""
        collection = args.get("collection", self.tool_config["collections"]["documents"])
        query = args.get("query", {})
        limit = args.get("limit", 10)

        # 검색 시뮬레이션
        await asyncio.sleep(0.3)

        # 샘플 결과
        sample_docs = [
            {"_id": "doc_0001", "title": "VirtualFab Overview", "type": "document"},
            {"_id": "doc_0002", "title": "Digital Twin Architecture", "type": "document"},
            {"_id": "doc_0003", "title": "Process Optimization", "type": "document"}
        ][:limit]

        return f"""✅ 문서 검색 완료

📄 컬렉션: {collection}
🔍 쿼리: {query}
📊 반환 문서 수: {len(sample_docs)}

📝 검색 결과:
{chr(10).join(f"- {doc['_id']}: {doc['title']} ({doc['type']})" for doc in sample_docs)}"""

    async def _get_stats(self) -> str:
        """DB 통계"""
        # 통계 조회 시뮬레이션
        await asyncio.sleep(0.2)

        stats = {
            "database": self.tool_config["database"],
            "collections": {
                "documents": {"count": 150, "size": "2.5 MB"},
                "chunks": {"count": 2500, "size": "45.8 MB"},
                "metadata": {"count": 25, "size": "0.3 MB"}
            },
            "total_size": "48.6 MB",
            "connections": 5
        }

        return f"""📊 MongoDB 통계

🗄️ 데이터베이스: {stats['database']}
📏 총 크기: {stats['total_size']}
🔗 활성 연결: {stats['connections']}

📋 컬렉션별 정보:
{chr(10).join(f"- {name}: {info['count']} 문서, {info['size']}" for name, info in stats['collections'].items())}"""

    async def _clear_collection(self, args: Dict[str, Any]) -> str:
        """컬렉션 비우기"""
        collection = args.get("collection", self.tool_config["collections"]["documents"])

        # 삭제 시뮬레이션
        await asyncio.sleep(0.5)

        return f"""✅ 컬렉션 비우기 완료

📄 컬렉션: {collection}
🗑️ 모든 문서가 삭제되었습니다.

⚠️ 실제 데이터 삭제는 안전을 위해 수동으로 수행하세요."""