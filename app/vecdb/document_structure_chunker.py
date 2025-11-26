"""
Document Structure-based Chunking
- ArXiv 논문의 섹션 구조 활용
- Abstract, Introduction, Methodology 등 자동 분할
"""
import re
import logging
from typing import Any

logger = logging.getLogger(__name__)


class DocumentStructureChunker:
    """
    문서 구조 기반 청킹
    - ArXiv 논문에 최적화
    - 섹션, 서브섹션 단위 분할
    """
    
    def __init__(self, min_chunk_size: int = 200, max_chunk_size: int = 2000):
        """
        Args:
            min_chunk_size: 최소 청크 크기 (너무 작은 섹션 병합)
            max_chunk_size: 최대 청크 크기 (큰 섹션 분할)
        """
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        
        # ArXiv 논문 섹션 패턴
        self.section_patterns = [
            r'^ABSTRACT\s*$',
            r'^INTRODUCTION\s*$',
            r'^BACKGROUND\s*$',
            r'^RELATED WORK\s*$',
            r'^METHODOLOGY\s*$',
            r'^METHOD\s*$',
            r'^APPROACH\s*$',
            r'^EXPERIMENTS?\s*$',
            r'^RESULTS?\s*$',
            r'^DISCUSSION\s*$',
            r'^CONCLUSION\s*$',
            r'^REFERENCES\s*$',
            r'^APPENDIX\s*$',
            # 번호 있는 섹션
            r'^\d+\.\s+[A-Z][A-Za-z\s]+$',
            r'^[IVX]+\.\s+[A-Z][A-Za-z\s]+$',
        ]
        
        self.subsection_patterns = [
            # 서브섹션 (제목만 대문자)
            r'^[A-Z][a-z].*[a-z]$',
            # 번호 있는 서브섹션
            r'^\d+\.\d+\s+[A-Z][A-Za-z\s]+$',
        ]
        
        logger.info(f"[Document Structure] 청킹 서비스 초기화")
        logger.info(f"  - min_chunk_size: {min_chunk_size}")
        logger.info(f"  - max_chunk_size: {max_chunk_size}")
    
    def is_section_header(self, line: str) -> bool:
        """섹션 헤더 감지"""
        line = line.strip()
        for pattern in self.section_patterns:
            if re.match(pattern, line, re.IGNORECASE):
                return True
        return False
    
    def is_subsection_header(self, line: str) -> bool:
        """서브섹션 헤더 감지"""
        line = line.strip()
        for pattern in self.subsection_patterns:
            if re.match(pattern, line):
                return True
        return False
    
    def chunk_text(self, text: str) -> list[dict[str, Any]]:
        """
        문서 구조 기반 청킹
        
        Args:
            text: 입력 텍스트 (ArXiv 논문)
            
        Returns:
            청크 리스트 (섹션 정보 포함)
        """
        if not text.strip():
            return []
        
        logger.info(f"[Document Structure] 청킹 시작: {len(text)} chars")
        
        lines = text.split('\n')
        chunks = []
        
        current_section = None
        current_subsection = None
        current_content = []
        chunk_id = 1
        
        for line in lines:
            stripped = line.strip()
            
            # 페이지 구분자 무시
            if stripped.startswith('=== Page'):
                continue
            
            # 섹션 헤더 감지
            if self.is_section_header(stripped):
                # 이전 청크 저장
                if current_content:
                    chunk = self._create_chunk(
                        current_content, 
                        chunk_id, 
                        current_section, 
                        current_subsection
                    )
                    if chunk:
                        chunks.append(chunk)
                        chunk_id += 1
                
                current_section = stripped
                current_subsection = None
                current_content = [line]
                continue
            
            # 서브섹션 헤더 감지
            if self.is_subsection_header(stripped) and len(stripped) < 100:
                # 이전 서브섹션 저장
                if current_content and len(current_content) > 1:
                    chunk = self._create_chunk(
                        current_content, 
                        chunk_id, 
                        current_section, 
                        current_subsection
                    )
                    if chunk:
                        chunks.append(chunk)
                        chunk_id += 1
                
                current_subsection = stripped
                current_content = [line]
                continue
            
            # 일반 내용 추가
            current_content.append(line)
            
            # 최대 크기 초과 시 분할
            content_str = '\n'.join(current_content)
            if len(content_str) > self.max_chunk_size:
                chunk = self._create_chunk(
                    current_content, 
                    chunk_id, 
                    current_section, 
                    current_subsection
                )
                if chunk:
                    chunks.append(chunk)
                    chunk_id += 1
                current_content = []
        
        # 마지막 청크 저장
        if current_content:
            chunk = self._create_chunk(
                current_content, 
                chunk_id, 
                current_section, 
                current_subsection
            )
            if chunk:
                chunks.append(chunk)
        
        logger.info(f"[Document Structure] 청킹 완료: {len(chunks)}개 청크")
        logger.info(f"  - 섹션 기반 분할")
        logger.info(f"  - 평균 크기: {sum(c['size'] for c in chunks) / len(chunks):.0f} chars")
        
        return chunks
    
    def _create_chunk(
        self, 
        content_lines: list[str], 
        chunk_id: int,
        section: str | None,
        subsection: str | None
    ) -> dict[str, Any] | None:
        """청크 생성"""
        content = '\n'.join(content_lines).strip()
        
        # 너무 작은 청크 무시
        if len(content) < self.min_chunk_size:
            return None
        
        chunk = {
            'id': chunk_id,
            'content': content,
            'size': len(content),
            'section': section or 'Unknown',
            'subsection': subsection,
            'structure_based': True,
        }
        
        return chunk


if __name__ == "__main__":
    # 테스트
    logging.basicConfig(level=logging.INFO)
    
    sample_text = """
=== Page 1 ===
Physics-Constrained Adaptive Neural Networks
Abstract
This paper presents a novel approach...

INTRODUCTION
Semiconductor Manufacturing Challenges
The semiconductor industry faces...

METHODOLOGY
Our approach consists of...

RESULTS
We achieved significant improvements...

CONCLUSION
In conclusion, we demonstrated...
"""
    
    chunker = DocumentStructureChunker()
    chunks = chunker.chunk_text(sample_text)
    
    print("\n생성된 청크:")
    for chunk in chunks:
        print(f"\n[{chunk['id']}] {chunk['section']}")
        if chunk['subsection']:
            print(f"  └─ {chunk['subsection']}")
        print(f"  크기: {chunk['size']} chars")
        print(f"  내용: {chunk['content'][:100]}...")

