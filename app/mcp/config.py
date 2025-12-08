"""
MCP 서버 설정
"""
import os
from typing import Dict, Any
from pathlib import Path


class MCPConfig:
    """MCP 서버 설정 관리"""

    def __init__(self):
        self.base_dir = Path(__file__).parent.parent.parent
        self.data_dir = self.base_dir / "data"
        self.crawled_dir = self.data_dir / "crawled"
        self.chunks_dir = self.data_dir / "chunks"
        self.vecdb_dir = self.data_dir / "vecdb"

    def get_server_config(self) -> Dict[str, Any]:
        """MCP 서버 설정 반환"""
        return {
            "name": "newera-mcp-server",
            "version": "1.0.0",
            "description": "VirtualFab RAG System MCP Server",
            "tools": {
                "web_crawler": {
                    "enabled": True,
                    "description": "ArXiv 논문 웹 크롤러"
                },
                "pdf_parser": {
                    "enabled": True,
                    "description": "PDF 문서 파싱 및 청킹"
                },
                "vector_db": {
                    "enabled": True,
                    "description": "FAISS Vector DB 관리"
                },
                "mongodb": {
                    "enabled": True,
                    "description": "MongoDB 문서 저장소 관리"
                }
            },
            "environment": {
                "OLLAMA_MODEL": os.getenv("OLLAMA_MODEL", "qwen2.5-3b-instruct:latest"),
                "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
                "MONGODB_URL": os.getenv("MONGODB_URL", "mongodb://localhost:27017")
            }
        }

    def get_tool_config(self, tool_name: str) -> Dict[str, Any]:
        """특정 Tool 설정 반환"""
        configs = {
            "web_crawler": {
                "arxiv_categories": ["cs.AI", "cs.LG", "cs.CV"],
                "max_papers": 100,
                "output_dir": str(self.crawled_dir / "ArXiv")
            },
            "pdf_parser": {
                "chunk_size": 512,
                "chunk_overlap": 50,
                "supported_formats": [".pdf", ".txt"],
                "output_dir": str(self.chunks_dir)
            },
            "vector_db": {
                "index_type": "HNSW",
                "dimension": 1024,
                "metric": "cosine",
                "db_path": str(self.vecdb_dir / "new_rag")
            },
            "mongodb": {
                "database": "newera",
                "collections": {
                    "documents": "documents",
                    "chunks": "chunks",
                    "metadata": "metadata"
                }
            }
        }
        return configs.get(tool_name, {})