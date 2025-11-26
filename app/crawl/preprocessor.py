"""
문서 전처리기: PDF 텍스트 추출, 청킹, 필터링
"""
import re
import json
import logging
from pathlib import Path
from typing import Any

from app.crawl.interfaces import DocumentProcessor

logger = logging.getLogger(__name__)

try:
    import PyPDF2
    import fitz
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logger.warning("PyMuPDF 미설치 - PDF 처리 기능 제한")

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    import nltk
    from nltk.tokenize import sent_tokenize
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    NLTK_AVAILABLE = True
except Exception:
    NLTK_AVAILABLE = False
    logger.warning("NLTK 미설치 - 기본 문장 분리 사용")


class DocumentPreprocessor(DocumentProcessor):
    """
    문서 전처리기
    - PDF 텍스트 추출 (PyMuPDF → pdfplumber → PyPDF2 fallback)
    - 문장 기반 스마트 청킹
    - 도메인 관련도 기반 필터링
    """
    
    def __init__(self, chunk_size: int = 500, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        self.semiconductor_keywords = [
            'wafer', 'fab', 'lithography', 'EUV', 'etch', 'deposition',
            'CVD', 'ALD', 'CMP', 'photoresist', 'patterning',
            'yield', 'defect', 'scrap', 'metrology', 'inspection',
            'WIP', 'cycle time', 'queue time', 'throughput',
            'virtual fab', 'virtual metrology', 'digital twin',
            'simulation', 'reinforcement learning', 'deep learning',
            'machine learning', 'optimization', 'process control',
        ]
        
        logger.info(f"Preprocessor 초기화: chunk_size={chunk_size}, overlap={overlap}")
    
    def extract_text(self, file_path: Path) -> str:
        """PDF 텍스트 추출 (fallback 전략)"""
        logger.info(f"PDF 추출 시작: {file_path.name}")
        
        if PYMUPDF_AVAILABLE:
            text = self._extract_with_pymupdf(file_path)
            if text.strip():
                logger.info(f"PyMuPDF 추출 완료: {len(text)} chars")
                return text
        
        if PDFPLUMBER_AVAILABLE:
            text = self._extract_with_pdfplumber(file_path)
            if text.strip():
                logger.info(f"pdfplumber 추출 완료: {len(text)} chars")
                return text
        
        text = self._extract_with_pypdf2(file_path)
        if text.strip():
            logger.info(f"PyPDF2 추출 완료: {len(text)} chars")
            return text
        
        logger.error(f"텍스트 추출 실패: {file_path.name}")
        return ""
    
    def _extract_with_pymupdf(self, pdf_path: Path) -> str:
        """PyMuPDF로 추출"""
        try:
            doc = fitz.open(str(pdf_path))
            text_content = ""
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                if page_text.strip():
                    text_content += f"\n=== Page {page_num + 1} ===\n{page_text}"
            
            doc.close()
            return text_content
        except Exception as e:
            logger.warning(f"PyMuPDF 실패: {e}")
            return ""
    
    def _extract_with_pdfplumber(self, pdf_path: Path) -> str:
        """pdfplumber로 추출"""
        try:
            import pdfplumber
            text_content = ""
            
            with pdfplumber.open(str(pdf_path)) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        text_content += f"\n=== Page {page_num + 1} ===\n{page_text}"
            
            return text_content
        except Exception as e:
            logger.warning(f"pdfplumber 실패: {e}")
            return ""
    
    def _extract_with_pypdf2(self, pdf_path: Path) -> str:
        """PyPDF2로 추출"""
        try:
            text_content = ""
            
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        text_content += f"\n=== Page {page_num + 1} ===\n{page_text}"
            
            return text_content
        except Exception as e:
            logger.warning(f"PyPDF2 실패: {e}")
            return ""
    
    def chunk_text(self, text: str, chunk_size: int | None = None, overlap: int | None = None) -> list[dict[str, Any]]:
        """문장 기반 스마트 청킹"""
        if not text.strip():
            return []
        
        chunk_size = chunk_size or self.chunk_size
        overlap = overlap or self.overlap
        
        logger.info(f"청킹 시작: {len(text)} chars → {chunk_size} chars/chunk")
        
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
            
            if current_size + sentence_len <= chunk_size:
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
                
                if overlap > 0 and current_chunk:
                    overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
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
        
        logger.info(f"청킹 완료: {len(chunks)}개 청크 생성")
        return chunks
    
    def filter_chunks(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """도메인 관련도 기반 청크 필터링"""
        filtered_chunks: list[dict[str, Any]] = []
        
        for chunk in chunks:
            content = chunk['content'].lower()
            
            if len(chunk['content']) < 100:
                logger.debug(f"청크 #{chunk['id']}: 너무 짧음 (스킵)")
                continue
            
            keyword_count = sum(1 for kw in self.semiconductor_keywords if kw.lower() in content)
            
            if keyword_count >= 2:
                chunk['keyword_count'] = keyword_count
                chunk['is_relevant'] = True
                filtered_chunks.append(chunk)
            else:
                logger.debug(f"청크 #{chunk['id']}: 키워드 부족 ({keyword_count}개)")
        
        logger.info(f"필터링 완료: {len(filtered_chunks)}/{len(chunks)}개 청크 유지")
        return filtered_chunks
    
    def process_pdf(self, pdf_path: Path, output_dir: Path) -> dict[str, Any]:
        """PDF 처리 파이프라인"""
        logger.info(f"PDF 처리 시작: {pdf_path.name}")
        
        text = self.extract_text(pdf_path)
        if not text:
            return {}
        
        chunks = self.chunk_text(text)
        filtered_chunks = self.filter_chunks(chunks)
        
        domain = 'General'
        parts = pdf_path.stem.split('_', 2)
        if len(parts) >= 2:
            domain = parts[1]
        
        result = {
            'source': 'ArXiv',
            'filename': pdf_path.name,
            'domain': domain,
            'total_chars': len(text),
            'total_chunks': len(filtered_chunks),
            'chunks': filtered_chunks,
        }
        
        output_file = output_dir / f"chunks_{pdf_path.stem}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"처리 완료: {output_file.name} ({len(filtered_chunks)}개 청크)")
        return result

