"""
Old-RAG 청킹 서비스
- 고정 크기: 500 문자
- 오버랩: 100 문자
- 문장 기반 수동 구현
"""
import re
import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    import nltk
    from nltk.tokenize import sent_tokenize
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    NLTK_AVAILABLE = True
except Exception:
    NLTK_AVAILABLE = False
    logger.warning("[Old-RAG] NLTK 미설치 - 기본 문장 분리 사용")


class ChunkingServiceOld:
    """
    Old-RAG 청킹 서비스 (Baseline)
    - 500 문자 고정
    - 100 문자 오버랩
    - 문장 기반 수동 구현
    """
    
    def __init__(self, chunk_size: int = 500, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        logger.info(f"[Old-RAG] 청킹 서비스 초기화")
        logger.info(f"  - chunk_size: {chunk_size} 문자")
        logger.info(f"  - overlap: {overlap} 문자")
    
    def chunk_text(self, text: str) -> list[dict[str, Any]]:
        """문장 기반 수동 청킹"""
        if not text.strip():
            return []
        
        logger.info(f"[Old-RAG] 청킹 시작: {len(text)} chars")
        
        if NLTK_AVAILABLE:
            sentences = sent_tokenize(text)
        else:
            sentences = re.split(r'[.!?]+\s+', text)
        
        chunks: list[dict[str, Any]] = []
        current_chunk = ""
        current_size = 0
        chunk_id = 1
        
        for sentence in sentences:
            sentence_len = len(sentence)
            
            if current_size + sentence_len <= self.chunk_size:
                current_chunk += sentence + " "
                current_size += sentence_len + 1
            else:
                if current_chunk.strip():
                    chunk_info = {
                        'id': chunk_id,
                        'content': current_chunk.strip(),
                        'size': len(current_chunk.strip()),
                        'sentences': len(sent_tokenize(current_chunk)) if NLTK_AVAILABLE else len(re.findall(r'[.!?]+', current_chunk)),
                    }
                    chunks.append(chunk_info)
                    chunk_id += 1
                
                if self.overlap > 0 and current_chunk:
                    overlap_text = current_chunk[-self.overlap:] if len(current_chunk) > self.overlap else current_chunk
                    current_chunk = overlap_text + sentence + " "
                    current_size = len(current_chunk)
                else:
                    current_chunk = sentence + " "
                    current_size = sentence_len + 1
        
        if current_chunk.strip():
            chunk_info = {
                'id': chunk_id,
                'content': current_chunk.strip(),
                'size': len(current_chunk.strip()),
                'sentences': len(sent_tokenize(current_chunk)) if NLTK_AVAILABLE else len(re.findall(r'[.!?]+', current_chunk)),
            }
            chunks.append(chunk_info)
        
        logger.info(f"[Old-RAG] 청킹 완료: {len(chunks)}개 청크")
        return chunks

