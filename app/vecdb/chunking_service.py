"""
New-RAG 청킹 서비스 (개선 버전)
- LangChain RecursiveCharacterTextSplitter 사용
- 512 토큰 기준 (8192까지 지원)
- 구조 보존 (Recursive)
"""
import logging
from typing import Any
from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class ChunkingService:
    """
    New-RAG 청킹 서비스 (개선 버전)
    - LangChain 기반
    - 512 토큰 기준 (bge-m3는 8192 지원)
    - Recursive 전략으로 구조 보존
    """
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        """
        Args:
            chunk_size: 청크 크기 (토큰 수에 근사)
            chunk_overlap: 오버랩 크기
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
            is_separator_regex=False,
        )
        
        logger.info(f"[New-RAG] 청킹 서비스 초기화")
        logger.info(f"  - chunk_size: {chunk_size} (토큰 근사)")
        logger.info(f"  - chunk_overlap: {chunk_overlap}")
        logger.info(f"  - separators: ['\\n\\n', '\\n', '. ', ' ', '']")
    
    def chunk_text(self, text: str) -> list[dict[str, Any]]:
        """
        텍스트를 청크로 분할
        
        Args:
            text: 입력 텍스트
            
        Returns:
            청크 리스트 (id, content, size 포함)
        """
        if not text.strip():
            return []
        
        logger.info(f"[New-RAG] 청킹 시작: {len(text)} chars")
        
        chunks_text = self.splitter.split_text(text)
        
        chunks = []
        for idx, chunk_text in enumerate(chunks_text, 1):
            chunk_info = {
                'id': idx,
                'content': chunk_text,
                'size': len(chunk_text),
            }
            chunks.append(chunk_info)
        
        logger.info(f"[New-RAG] 청킹 완료: {len(chunks)}개 청크")
        return chunks
    
    def chunk_documents(self, documents: list[str]) -> list[dict[str, Any]]:
        """
        여러 문서를 청크로 분할
        
        Args:
            documents: 문서 리스트
            
        Returns:
            청크 리스트
        """
        all_chunks = []
        
        for doc_idx, doc in enumerate(documents):
            chunks = self.chunk_text(doc)
            
            for chunk in chunks:
                chunk['doc_index'] = doc_idx
            
            all_chunks.extend(chunks)
        
        logger.info(f"[New-RAG] 총 {len(all_chunks)}개 청크 생성 ({len(documents)}개 문서)")
        return all_chunks

