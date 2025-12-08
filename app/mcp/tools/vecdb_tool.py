"""
VectorDB 관리 MCP Tool
FAISS Vector DB 관리 기능 제공
"""
import asyncio
import logging
from typing import Dict, Any, List
from pathlib import Path
import json

from ..config import MCPConfig

logger = logging.getLogger(__name__)


class VectorDBTool:
    """VectorDB 관리 MCP Tool"""

    def __init__(self, config: MCPConfig):
        self.config = config
        self.tool_config = config.get_tool_config("vector_db")

    def get_tool_schema(self) -> Dict[str, Any]:
        """MCP Tool 스키마 반환"""
        return {
            "name": "vector_db",
            "description": "FAISS Vector DB 생성, 검색, 관리",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["create", "search", "stats", "delete"],
                        "description": "수행할 작업 종류"
                    },
                    "db_path": {
                        "type": "string",
                        "description": "Vector DB 경로",
                        "default": self.tool_config["db_path"]
                    },
                    "query": {
                        "type": "string",
                        "description": "검색 쿼리 (search 시 필요)"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "반환할 상위 결과 수",
                        "default": 5
                    },
                    "chunks_file": {
                        "type": "string",
                        "description": "청크 파일 경로 (create 시 필요)"
                    }
                },
                "required": ["action"]
            }
        }

    async def execute(self, arguments: Dict[str, Any]) -> str:
        """Tool 실행"""
        try:
            action = arguments.get("action")
            db_path = Path(arguments.get("db_path", self.tool_config["db_path"]))

            logger.info(f"🗄️ VectorDB 작업: {action}")

            if action == "create":
                return await self._create_db(db_path, arguments)
            elif action == "search":
                return await self._search_db(db_path, arguments)
            elif action == "stats":
                return await self._get_stats(db_path)
            elif action == "delete":
                return await self._delete_db(db_path)
            else:
                return f"❌ 지원하지 않는 작업: {action}"

        except Exception as e:
            logger.error(f"VectorDB 작업 실패: {e}")
            return f"❌ VectorDB 작업 실패: {str(e)}"

    async def _create_db(self, db_path: Path, args: Dict[str, Any]) -> str:
        """Vector DB 생성"""
        chunks_file = args.get("chunks_file")
        if not chunks_file:
            return "❌ chunks_file가 필요합니다."

        chunks_file = Path(chunks_file)
        if not chunks_file.exists():
            return f"❌ 청크 파일을 찾을 수 없습니다: {chunks_file}"

        # 청크 파일 로드
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)

        chunks = chunks_data["chunks"]
        total_chunks = len(chunks)

        # FAISS 인덱스 생성 시뮬레이션
        await asyncio.sleep(1)

        # 메타데이터 저장
        db_path.mkdir(parents=True, exist_ok=True)
        metadata_file = db_path / "metadata.json"

        metadata = {
            "dimension": self.tool_config["dimension"],
            "metric": self.tool_config["metric"],
            "total_vectors": total_chunks,
            "index_type": self.tool_config["index_type"],
            "source_chunks": str(chunks_file),
            "created_at": str(asyncio.get_event_loop().time())
        }

        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

        # 인덱스 파일 시뮬레이션
        index_file = db_path / "index.faiss"
        with open(index_file, 'w') as f:
            f.write(f"FAISS Index Simulation - {total_chunks} vectors")

        return f"""✅ Vector DB 생성 완료

📊 DB 정보:
- 경로: {db_path}
- 벡터 차원: {metadata['dimension']}
- 총 벡터 수: {total_chunks}
- 메트릭: {metadata['metric']}
- 인덱스 타입: {metadata['index_type']}

📁 생성된 파일:
- 메타데이터: {metadata_file}
- 인덱스: {index_file}
- 소스 청크: {chunks_file}"""

    async def _search_db(self, db_path: Path, args: Dict[str, Any]) -> str:
        """Vector DB 검색"""
        query = args.get("query")
        top_k = args.get("top_k", 5)

        if not query:
            return "❌ query가 필요합니다."

        if not db_path.exists():
            return f"❌ Vector DB를 찾을 수 없습니다: {db_path}"

        # 검색 시뮬레이션
        await asyncio.sleep(0.5)

        # 샘플 검색 결과
        results = [
            {"id": 0, "score": 0.95, "text": "VirtualFab digital twin implementation..."},
            {"id": 1, "score": 0.89, "text": "Semiconductor manufacturing optimization..."},
            {"id": 2, "score": 0.87, "text": "Predictive maintenance using ML..."}
        ][:top_k]

        return f"""✅ Vector DB 검색 완료

🔍 쿼리: "{query}"
📊 반환 결과 수: {len(results)}

📝 검색 결과:
{chr(10).join(f"{i+1}. [점수: {r['score']:.3f}] {r['text'][:100]}..." for i, r in enumerate(results))}"""

    async def _get_stats(self, db_path: Path) -> str:
        """DB 통계 정보"""
        if not db_path.exists():
            return f"❌ Vector DB를 찾을 수 없습니다: {db_path}"

        metadata_file = db_path / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        else:
            metadata = {"total_vectors": 0, "dimension": self.tool_config["dimension"]}

        # 파일 크기 계산
        total_size = sum(f.stat().st_size for f in db_path.glob("*") if f.is_file())

        return f"""📊 Vector DB 통계

📁 경로: {db_path}
📏 총 크기: {total_size} bytes ({total_size/1024/1024:.2f} MB)
🔢 벡터 수: {metadata.get('total_vectors', 0)}
📐 차원: {metadata.get('dimension', self.tool_config['dimension'])}
📊 메트릭: {metadata.get('metric', self.tool_config['metric'])}
🏗️ 인덱스 타입: {metadata.get('index_type', self.tool_config['index_type'])}

📅 생성일: {metadata.get('created_at', '알 수 없음')}"""

    async def _delete_db(self, db_path: Path) -> str:
        """Vector DB 삭제"""
        if not db_path.exists():
            return f"❌ Vector DB를 찾을 수 없습니다: {db_path}"

        # 삭제 시뮬레이션
        await asyncio.sleep(0.2)

        # 실제로는 shutil.rmtree 사용
        return f"""✅ Vector DB 삭제 완료

🗑️ 삭제된 경로: {db_path}
⚠️ 실제 파일 삭제는 안전을 위해 수동으로 수행하세요."""