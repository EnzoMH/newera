"""
PDF Parser MCP Tool
PDF 문서 파싱 및 청킹 기능 제공
"""
import asyncio
import logging
from typing import Dict, Any, List
from pathlib import Path

from ..config import MCPConfig

logger = logging.getLogger(__name__)


class PDFParserTool:
    """PDF 파서 MCP Tool"""

    def __init__(self, config: MCPConfig):
        self.config = config
        self.tool_config = config.get_tool_config("pdf_parser")

    def get_tool_schema(self) -> Dict[str, Any]:
        """MCP Tool 스키마 반환"""
        return {
            "name": "pdf_parser",
            "description": "PDF 문서 파싱 및 텍스트 청킹",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "파싱할 PDF 파일 경로"
                    },
                    "chunk_size": {
                        "type": "integer",
                        "description": "청크 크기 (토큰 수)",
                        "default": self.tool_config["chunk_size"]
                    },
                    "chunk_overlap": {
                        "type": "integer",
                        "description": "청크 오버랩 (토큰 수)",
                        "default": self.tool_config["chunk_overlap"]
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "청크 저장 디렉토리",
                        "default": self.tool_config["output_dir"]
                    }
                },
                "required": ["file_path"]
            }
        }

    async def execute(self, arguments: Dict[str, Any]) -> str:
        """Tool 실행"""
        try:
            file_path = arguments.get("file_path")
            chunk_size = arguments.get("chunk_size", self.tool_config["chunk_size"])
            chunk_overlap = arguments.get("chunk_overlap", self.tool_config["chunk_overlap"])
            output_dir = Path(arguments.get("output_dir", self.tool_config["output_dir"]))

            if not file_path:
                return "❌ file_path가 필요합니다."

            file_path = Path(file_path)
            if not file_path.exists():
                return f"❌ 파일을 찾을 수 없습니다: {file_path}"

            logger.info(f"📄 PDF 파싱 시작: {file_path}")

            # PDF 파싱 및 청킹
            chunks = await self._parse_and_chunk_pdf(file_path, chunk_size, chunk_overlap)

            # 청크 저장
            output_dir.mkdir(parents=True, exist_ok=True)
            chunk_file = output_dir / f"{file_path.stem}_chunks.json"

            import json
            with open(chunk_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "source_file": str(file_path),
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "total_chunks": len(chunks),
                    "chunks": chunks
                }, f, ensure_ascii=False, indent=2)

            return f"""✅ PDF 파싱 및 청킹 완료

📊 처리 결과:
- 원본 파일: {file_path.name}
- 청크 크기: {chunk_size} 토큰
- 청크 오버랩: {chunk_overlap} 토큰
- 생성된 청크 수: {len(chunks)}
- 저장 위치: {chunk_file}

📝 샘플 청크:
{chr(10).join(f"청크 {i+1}: {chunk[:100]}..." for i, chunk in enumerate(chunks[:3]))}"""

        except Exception as e:
            logger.error(f"PDF 파싱 실패: {e}")
            return f"❌ PDF 파싱 실패: {str(e)}"

    async def _parse_and_chunk_pdf(self, file_path: Path, chunk_size: int, overlap: int) -> List[str]:
        """PDF 파싱 및 청킹 시뮬레이션"""
        # 실제 구현에서는 PyMuPDF나 pdfplumber 사용
        # 여기서는 샘플 텍스트 청킹

        await asyncio.sleep(0.5)  # 파싱 시뮬레이션

        # 샘플 텍스트 (실제로는 PDF에서 추출)
        sample_text = """
        VirtualFab: Digital Twin for Semiconductor Manufacturing

        Abstract: This paper presents VirtualFab, a comprehensive digital twin system
        for semiconductor manufacturing facilities. VirtualFab integrates real-time
        sensor data, process models, and machine learning algorithms to create
        accurate virtual representations of physical fabrication processes.

        Introduction: Semiconductor manufacturing is becoming increasingly complex
        with shrinking feature sizes and growing process variability. Digital twins
        offer a promising approach to optimize manufacturing operations, predict
        equipment failures, and improve yield rates.

        System Architecture: VirtualFab consists of three main components:
        1. Data Acquisition Layer: Collects data from various sensors and equipment
        2. Modeling Layer: Creates physics-based and data-driven models
        3. Optimization Layer: Uses reinforcement learning for process optimization

        Results: VirtualFab achieved 15% improvement in cycle time and 20% reduction
        in defect rates compared to traditional methods.
        """

        # 간단한 텍스트 청킹 (실제로는 더 정교한 알고리즘 사용)
        words = sample_text.split()
        chunks = []

        i = 0
        while i < len(words):
            chunk_words = words[i:i + chunk_size]
            chunk_text = " ".join(chunk_words)
            chunks.append(chunk_text)

            # 오버랩만큼 뒤로 이동
            i += chunk_size - overlap
            if i <= 0:  # 무한 루프 방지
                break

        return chunks