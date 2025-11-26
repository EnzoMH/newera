"""
Agentic Chunking (하이브리드 LLM)
- Main: 로컬 LLM (llama.cpp + Qwen2.5-3B-Korean)
- Fallback: Gemini 2.0 Flash
- 비용 최적화 + 안정성
"""
import os
import logging
from typing import Any
from enum import Enum

logger = logging.getLogger(__name__)

# 선택적 import (없어도 동작)
try:
    from llama_cpp import Llama
    LLAMACPP_AVAILABLE = True
except ImportError:
    LLAMACPP_AVAILABLE = False
    logger.warning("llama-cpp-python 미설치 - Gemini만 사용")

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("google-generativeai 미설치 - 로컬 LLM만 사용")


class LLMBackend(str, Enum):
    """LLM 백엔드 타입"""
    LOCAL = "local"      # llama.cpp
    GEMINI = "gemini"    # Google Gemini
    AUTO = "auto"        # 로컬 우선 → Gemini fallback


class AgenticChunker:
    """
    LLM 기반 지능형 청킹
    - Gemini가 문서를 읽고 의미 있는 단위로 분할
    - 가장 정확하지만 비용/속도 트레이드오프
    """
    
    def __init__(self, api_key: str | None = None, model: str = "gemini-2.0-flash-lite-001"):
        """
        Args:
            api_key: Google API 키
            model: Gemini 모델 (flash = 빠름/저렴, pro = 정확/비쌈)
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.model_name = model
        
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY 환경변수 또는 api_key 인자 필요")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model)
        
        logger.info(f"[Agentic] 청킹 서비스 초기화")
        logger.info(f"  - 모델: {model}")
        logger.info(f"  - LLM 기반 지능형 청킹")
    
    def chunk_text(self, text: str, max_chunks: int = 20) -> list[dict[str, Any]]:
        """
        LLM 기반 청킹
        
        Args:
            text: 입력 텍스트
            max_chunks: 최대 청크 수
            
        Returns:
            청크 리스트 (LLM이 결정한 의미 단위)
        """
        if not text.strip():
            return []
        
        logger.info(f"[Agentic] 청킹 시작: {len(text)} chars")
        logger.info("  ⚠️  LLM 호출 중 (비용 발생)...")
        
        prompt = self._create_chunking_prompt(text, max_chunks)
        
        try:
            response = self.model.generate_content(prompt)
            chunks = self._parse_llm_response(response.text)
            
            logger.info(f"[Agentic] 청킹 완료: {len(chunks)}개 청크")
            logger.info(f"  - LLM이 의미 단위로 분할")
            
            return chunks
        
        except Exception as e:
            logger.error(f"[Agentic] LLM 호출 실패: {e}")
            logger.info("  → Fallback: 간단한 문단 분할")
            return self._fallback_chunking(text)
    
    def _create_chunking_prompt(self, text: str, max_chunks: int) -> str:
        """청킹 프롬프트 생성"""
        prompt = f"""You are an expert document analyzer. 
Split this scientific paper into meaningful semantic chunks.

**Instructions:**
1. Each chunk should contain ONE coherent topic or idea
2. Preserve section boundaries (Abstract, Introduction, etc.)
3. Keep related information together
4. Aim for {max_chunks} chunks maximum
5. Output format:
```
CHUNK_1:
[content]

CHUNK_2:
[content]
```

**Document:**
{text[:10000]}  # 10K chars limit for API

**Output the chunks:**"""
        
        return prompt
    
    def _parse_llm_response(self, response_text: str) -> list[dict[str, Any]]:
        """LLM 응답 파싱"""
        chunks = []
        
        # "CHUNK_N:" 패턴으로 분할
        import re
        chunk_pattern = r'CHUNK_(\d+):\s*(.*?)(?=CHUNK_\d+:|$)'
        matches = re.findall(chunk_pattern, response_text, re.DOTALL)
        
        for chunk_id, content in matches:
            content = content.strip()
            if content and len(content) > 50:  # 최소 길이
                chunks.append({
                    'id': int(chunk_id),
                    'content': content,
                    'size': len(content),
                    'method': 'llm_based',
                    'agentic': True,
                })
        
        # 파싱 실패 시 fallback
        if not chunks:
            logger.warning("[Agentic] LLM 응답 파싱 실패, fallback 사용")
            return self._fallback_chunking(response_text)
        
        return chunks
    
    def _fallback_chunking(self, text: str) -> list[dict[str, Any]]:
        """Fallback: 간단한 문단 분할"""
        paragraphs = text.split('\n\n')
        chunks = []
        
        for idx, para in enumerate(paragraphs, 1):
            para = para.strip()
            if para and len(para) > 100:
                chunks.append({
                    'id': idx,
                    'content': para,
                    'size': len(para),
                    'method': 'fallback',
                    'agentic': False,
                })
        
        return chunks
    
    def analyze_document_structure(self, text: str) -> dict[str, Any]:
        """
        문서 구조 분석 (보너스 기능)
        - LLM이 문서의 주요 섹션, 주제 등을 요약
        """
        prompt = f"""Analyze this scientific paper and provide:
1. Main sections (e.g., Abstract, Introduction, Methodology...)
2. Key topics covered
3. Recommended chunking strategy

Paper (first 5000 chars):
{text[:5000]}

Provide a JSON response."""
        
        try:
            response = self.model.generate_content(prompt)
            logger.info(f"[Agentic] 문서 구조 분석 완료")
            return {'analysis': response.text}
        except Exception as e:
            logger.error(f"[Agentic] 구조 분석 실패: {e}")
            return {'error': str(e)}


class HybridChunker:
    """
    하이브리드 청킹 전략
    - Document Structure (빠름, 저렴) 먼저 시도
    - 복잡한 경우 Agentic (느림, 비쌈) 사용
    """
    
    def __init__(self, gemini_api_key: str | None = None):
        from app.vecdb.document_structure_chunker import DocumentStructureChunker
        
        self.structure_chunker = DocumentStructureChunker()
        self.agentic_chunker = AgenticChunker(api_key=gemini_api_key) if gemini_api_key else None
        
        logger.info("[Hybrid] 하이브리드 청킹 초기화")
        logger.info("  - Primary: Document Structure")
        logger.info(f"  - Fallback: Agentic (활성화: {self.agentic_chunker is not None})")
    
    def chunk_text(self, text: str, use_agentic: bool = False) -> list[dict[str, Any]]:
        """
        하이브리드 청킹
        
        Args:
            text: 입력 텍스트
            use_agentic: True면 Agentic 우선, False면 Structure 우선
        """
        if use_agentic and self.agentic_chunker:
            logger.info("[Hybrid] Agentic 청킹 사용")
            return self.agentic_chunker.chunk_text(text)
        else:
            logger.info("[Hybrid] Document Structure 청킹 사용")
            chunks = self.structure_chunker.chunk_text(text)
            
            # 구조 청킹 실패 시 Agentic fallback
            if not chunks and self.agentic_chunker:
                logger.warning("[Hybrid] Structure 실패, Agentic fallback")
                return self.agentic_chunker.chunk_text(text)
            
            return chunks


if __name__ == "__main__":
    # 테스트 (API 키 필요)
    logging.basicConfig(level=logging.INFO)
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("⚠️  GOOGLE_API_KEY 환경변수 필요")
        print("테스트 스킵")
    else:
        sample_text = """
Physics-Constrained Adaptive Neural Networks

ABSTRACT
This paper presents a novel approach to semiconductor manufacturing optimization...

INTRODUCTION
The semiconductor industry faces unprecedented challenges...

METHODOLOGY
Our method combines physics-informed neural networks with adaptive learning...
"""
        
        # Agentic 청킹 테스트
        chunker = AgenticChunker(api_key=api_key, model="gemini-1.5-flash")
        chunks = chunker.chunk_text(sample_text, max_chunks=5)
        
        print("\n=== Agentic 청킹 결과 ===")
        for chunk in chunks:
            print(f"\n[{chunk['id']}] {chunk.get('method', 'unknown')}")
            print(f"  크기: {chunk['size']} chars")
            print(f"  내용: {chunk['content'][:100]}...")

