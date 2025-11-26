"""
하이브리드 Agentic Chunker
- Main: 로컬 LLM (llama.cpp + Qwen2.5-3B-Korean)
- Fallback: Gemini 2.0 Flash
- 비용 최적화 + 안정성 보장
"""
import os
import logging
from typing import Any
from enum import Enum
import time

logger = logging.getLogger(__name__)

# 선택적 import
try:
    from llama_cpp import Llama
    LLAMACPP_AVAILABLE = True
except ImportError:
    LLAMACPP_AVAILABLE = False
    logger.warning("[Hybrid] llama-cpp-python 미설치 - Gemini만 사용")

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("[Hybrid] google-generativeai 미설치 - 로컬 LLM만 사용")


class LLMBackend(str, Enum):
    """LLM 백엔드 타입"""
    LOCAL = "local"      # llama.cpp (무료, 빠름)
    GEMINI = "gemini"    # Gemini 2.0 Flash (유료, 안정적)
    AUTO = "auto"        # 로컬 → Gemini fallback


class HybridAgenticChunker:
    """
    하이브리드 Agentic Chunker
    
    전략:
    1. 로컬 LLM 우선 시도 (무료)
    2. 실패 시 Gemini fallback (안정성)
    3. 비용/성능 최적화
    
    비용 비교:
    - 로컬 LLM: $0 (GPU 전기료만)
    - Gemini Flash: ~$0.01/문서
    
    사용 예:
        chunker = HybridAgenticChunker(
            local_model="MyeongHo0621/Qwen2.5-3B-Korean",
            gemini_api_key=GOOGLE_API_KEY
        )
        chunks = chunker.chunk_text(text)
    """
    
    def __init__(
        self,
        local_model: str = "MyeongHo0621/Qwen2.5-3B-Korean",
        local_model_file: str = "gguf/qwen25-3b-korean-Q4_K_M.gguf",
        gemini_api_key: str | None = None,
        gemini_model: str = "gemini-2.0-flash-001",
        backend: LLMBackend = LLMBackend.AUTO,
        use_gpu: bool = True
    ):
        """
        Args:
            local_model: Hugging Face repo ID
            local_model_file: GGUF 파일명
            gemini_api_key: Google API 키
            gemini_model: Gemini 모델 (2.0-flash-001 추천)
            backend: LLM 백엔드 선택
            use_gpu: 로컬 LLM GPU 사용 여부
        """
        self.backend = backend
        self.local_llm = None
        self.gemini_model = None
        
        # 통계
        self.stats = {
            'local_success': 0,
            'local_failure': 0,
            'gemini_success': 0,
            'gemini_failure': 0,
            'total_cost': 0.0,  # USD
        }
        
        logger.info(f"[Hybrid Agentic] 초기화")
        logger.info(f"  - 백엔드: {backend}")
        
        # 1. 로컬 LLM 초기화
        if backend in [LLMBackend.LOCAL, LLMBackend.AUTO]:
            if LLAMACPP_AVAILABLE:
                try:
                    logger.info(f"  - 로컬 모델 로드 중: {local_model}")
                    self.local_llm = Llama.from_pretrained(
                        repo_id=local_model,
                        filename=local_model_file,
                        n_ctx=4096,  # 컨텍스트 크기
                        n_gpu_layers=-1 if use_gpu else 0,  # GPU 레이어 (-1 = 전체)
                        verbose=False
                    )
                    logger.info(f"  ✓ 로컬 LLM 로드 완료")
                    logger.info(f"    - GPU: {use_gpu}")
                    logger.info(f"    - 비용: $0 (무료!)")
                except Exception as e:
                    logger.error(f"  ✗ 로컬 LLM 로드 실패: {e}")
                    if backend == LLMBackend.LOCAL:
                        raise
            else:
                logger.warning("  ⚠️  llama-cpp-python 미설치")
                if backend == LLMBackend.LOCAL:
                    raise ImportError("llama-cpp-python 설치 필요: pip install llama-cpp-python")
        
        # 2. Gemini 초기화
        if backend in [LLMBackend.GEMINI, LLMBackend.AUTO]:
            if GEMINI_AVAILABLE:
                api_key = gemini_api_key or os.getenv("GOOGLE_API_KEY")
                if api_key:
                    try:
                        genai.configure(api_key=api_key)
                        self.gemini_model = genai.GenerativeModel(gemini_model)
                        logger.info(f"  ✓ Gemini 초기화 완료")
                        logger.info(f"    - 모델: {gemini_model}")
                        logger.info(f"    - 비용: ~$0.01/문서")
                    except Exception as e:
                        logger.error(f"  ✗ Gemini 초기화 실패: {e}")
                        if backend == LLMBackend.GEMINI:
                            raise
                else:
                    logger.warning("  ⚠️  GOOGLE_API_KEY 없음")
                    if backend == LLMBackend.GEMINI:
                        raise ValueError("GOOGLE_API_KEY 환경변수 필요")
            else:
                logger.warning("  ⚠️  google-generativeai 미설치")
                if backend == LLMBackend.GEMINI:
                    raise ImportError("google-generativeai 설치 필요")
        
        # 최종 상태 확인
        if not self.local_llm and not self.gemini_model:
            raise RuntimeError("사용 가능한 LLM 백엔드가 없습니다")
        
        logger.info("[Hybrid Agentic] 초기화 완료")
    
    def chunk_text(self, text: str, max_chunks: int = 20) -> list[dict[str, Any]]:
        """
        하이브리드 청킹
        
        Args:
            text: 입력 텍스트
            max_chunks: 최대 청크 수
            
        Returns:
            청크 리스트
        """
        if not text.strip():
            return []
        
        logger.info(f"[Hybrid] 청킹 시작: {len(text)} chars")
        
        # AUTO 모드: 로컬 → Gemini fallback
        if self.backend == LLMBackend.AUTO:
            if self.local_llm:
                logger.info("[Hybrid] 1차 시도: 로컬 LLM")
                chunks = self._chunk_with_local(text, max_chunks)
                if chunks:
                    self.stats['local_success'] += 1
                    logger.info(f"[Hybrid] ✓ 로컬 LLM 성공: {len(chunks)}개 청크")
                    logger.info(f"  - 비용: $0")
                    return chunks
                else:
                    self.stats['local_failure'] += 1
                    logger.warning("[Hybrid] ✗ 로컬 LLM 실패, Gemini fallback")
            
            if self.gemini_model:
                logger.info("[Hybrid] 2차 시도: Gemini")
                chunks = self._chunk_with_gemini(text, max_chunks)
                if chunks:
                    self.stats['gemini_success'] += 1
                    cost = self._estimate_cost(text)
                    self.stats['total_cost'] += cost
                    logger.info(f"[Hybrid] ✓ Gemini 성공: {len(chunks)}개 청크")
                    logger.info(f"  - 비용: ${cost:.4f}")
                    return chunks
                else:
                    self.stats['gemini_failure'] += 1
            
            # 둘 다 실패
            logger.error("[Hybrid] ✗ 모든 백엔드 실패, fallback 청킹 사용")
            return self._fallback_chunking(text)
        
        # LOCAL 모드
        elif self.backend == LLMBackend.LOCAL:
            chunks = self._chunk_with_local(text, max_chunks)
            if chunks:
                self.stats['local_success'] += 1
            else:
                self.stats['local_failure'] += 1
                chunks = self._fallback_chunking(text)
            return chunks
        
        # GEMINI 모드
        elif self.backend == LLMBackend.GEMINI:
            chunks = self._chunk_with_gemini(text, max_chunks)
            if chunks:
                self.stats['gemini_success'] += 1
                cost = self._estimate_cost(text)
                self.stats['total_cost'] += cost
            else:
                self.stats['gemini_failure'] += 1
                chunks = self._fallback_chunking(text)
            return chunks
    
    def _chunk_with_local(self, text: str, max_chunks: int) -> list[dict[str, Any]]:
        """로컬 LLM으로 청킹"""
        if not self.local_llm:
            return []
        
        try:
            start_time = time.time()
            
            prompt = self._create_chunking_prompt(text, max_chunks)
            
            # llama.cpp 호출
            response = self.local_llm(
                prompt,
                max_tokens=4096,
                temperature=0.3,
                stop=["User:", "\n\n\n"]
            )
            
            elapsed = time.time() - start_time
            response_text = response['choices'][0]['text']
            
            logger.info(f"[Local LLM] 응답 생성: {elapsed:.2f}초")
            
            chunks = self._parse_llm_response(response_text)
            
            if chunks:
                for chunk in chunks:
                    chunk['llm_backend'] = 'local'
                    chunk['llm_model'] = 'Qwen2.5-3B-Korean'
            
            return chunks
        
        except Exception as e:
            logger.error(f"[Local LLM] 오류: {e}")
            return []
    
    def _chunk_with_gemini(self, text: str, max_chunks: int) -> list[dict[str, Any]]:
        """Gemini로 청킹"""
        if not self.gemini_model:
            return []
        
        try:
            start_time = time.time()
            
            prompt = self._create_chunking_prompt(text, max_chunks)
            
            # Gemini 호출
            response = self.gemini_model.generate_content(prompt)
            
            elapsed = time.time() - start_time
            
            logger.info(f"[Gemini] 응답 생성: {elapsed:.2f}초")
            
            chunks = self._parse_llm_response(response.text)
            
            if chunks:
                for chunk in chunks:
                    chunk['llm_backend'] = 'gemini'
                    chunk['llm_model'] = 'gemini-2.0-flash-001'
            
            return chunks
        
        except Exception as e:
            logger.error(f"[Gemini] 오류: {e}")
            return []
    
    def _create_chunking_prompt(self, text: str, max_chunks: int) -> str:
        """청킹 프롬프트 생성"""
        # 한국어/영어 혼합 프롬프트 (Qwen2.5-3B-Korean 최적화)
        prompt = f"""You are an expert document analyzer. Split this scientific paper into meaningful semantic chunks.

**Instructions:**
1. Each chunk should contain ONE coherent topic
2. Preserve section boundaries (Abstract, Introduction, Methodology, etc.)
3. Keep related information together
4. Create {max_chunks} chunks maximum
5. Output format:
```
CHUNK_1:
[content]

CHUNK_2:
[content]
```

**Document (first 8000 chars):**
{text[:8000]}

**Output the chunks:**"""
        
        return prompt
    
    def _parse_llm_response(self, response_text: str) -> list[dict[str, Any]]:
        """LLM 응답 파싱"""
        chunks = []
        
        # "CHUNK_N:" 패턴으로 분할
        import re
        chunk_pattern = r'CHUNK[_\s]*(\d+):\s*(.*?)(?=CHUNK[_\s]*\d+:|$)'
        matches = re.findall(chunk_pattern, response_text, re.DOTALL | re.IGNORECASE)
        
        for chunk_id, content in matches:
            content = content.strip()
            if content and len(content) > 50:
                chunks.append({
                    'id': int(chunk_id),
                    'content': content,
                    'size': len(content),
                    'method': 'agentic',
                })
        
        return chunks
    
    def _fallback_chunking(self, text: str) -> list[dict[str, Any]]:
        """Fallback: 간단한 문단 분할"""
        logger.info("[Hybrid] Fallback 청킹 사용")
        
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
                    'llm_backend': 'none',
                })
        
        return chunks
    
    def _estimate_cost(self, text: str) -> float:
        """비용 추정 (Gemini 2.0 Flash)"""
        # Gemini 2.0 Flash 가격 (2025년 기준)
        # https://ai.google.dev/pricing
        input_price = 0.075 / 1_000_000  # $0.075 per 1M tokens
        output_price = 0.30 / 1_000_000   # $0.30 per 1M tokens
        
        # 간단한 토큰 추정 (1 token ≈ 4 chars)
        input_tokens = len(text) / 4
        output_tokens = 2000  # 평균 출력 (청크 설명 등)
        
        cost = (input_tokens * input_price) + (output_tokens * output_price)
        return cost
    
    def get_stats(self) -> dict[str, Any]:
        """통계 반환"""
        total_attempts = (
            self.stats['local_success'] + 
            self.stats['local_failure'] + 
            self.stats['gemini_success'] + 
            self.stats['gemini_failure']
        )
        
        local_rate = 0
        if self.stats['local_success'] + self.stats['local_failure'] > 0:
            local_rate = self.stats['local_success'] / (
                self.stats['local_success'] + self.stats['local_failure']
            )
        
        return {
            'backend': self.backend,
            'total_attempts': total_attempts,
            'local_success': self.stats['local_success'],
            'local_failure': self.stats['local_failure'],
            'local_success_rate': local_rate,
            'gemini_success': self.stats['gemini_success'],
            'gemini_failure': self.stats['gemini_failure'],
            'total_cost_usd': self.stats['total_cost'],
            'avg_cost_per_doc': self.stats['total_cost'] / max(total_attempts, 1),
        }
    
    def print_stats(self):
        """통계 출력"""
        stats = self.get_stats()
        
        print("\n" + "="*60)
        print("하이브리드 Agentic Chunker 통계")
        print("="*60)
        print(f"백엔드: {stats['backend']}")
        print(f"총 시도: {stats['total_attempts']}회")
        print(f"\n로컬 LLM:")
        print(f"  - 성공: {stats['local_success']}회")
        print(f"  - 실패: {stats['local_failure']}회")
        print(f"  - 성공률: {stats['local_success_rate']:.1%}")
        print(f"  - 비용: $0 (무료!)")
        print(f"\nGemini:")
        print(f"  - 성공: {stats['gemini_success']}회")
        print(f"  - 실패: {stats['gemini_failure']}회")
        print(f"\n비용:")
        print(f"  - 총 비용: ${stats['total_cost_usd']:.4f}")
        print(f"  - 문서당 평균: ${stats['avg_cost_per_doc']:.4f}")
        print("="*60)


if __name__ == "__main__":
    # 테스트
    logging.basicConfig(level=logging.INFO)
    
    sample_text = """
=== Page 1 ===
Physics-Constrained Adaptive Neural Networks Enable Real-Time
Semiconductor Manufacturing Optimization

ABSTRACT
This paper presents a novel approach to semiconductor manufacturing optimization...

INTRODUCTION
The semiconductor industry faces unprecedented computational challenges...

METHODOLOGY
Our method combines physics-informed neural networks with adaptive learning...

RESULTS
We achieved significant improvements in EUV lithography optimization...
"""
    
    print("\n" + "="*60)
    print("하이브리드 Agentic Chunker 테스트")
    print("="*60)
    
    # AUTO 모드 (로컬 → Gemini fallback)
    chunker = HybridAgenticChunker(
        backend=LLMBackend.AUTO,
        gemini_api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    chunks = chunker.chunk_text(sample_text, max_chunks=5)
    
    print(f"\n생성된 청크: {len(chunks)}개")
    for chunk in chunks:
        print(f"\n[{chunk['id']}] {chunk.get('llm_backend', 'unknown')}")
        print(f"  모델: {chunk.get('llm_model', 'unknown')}")
        print(f"  크기: {chunk['size']} chars")
        print(f"  내용: {chunk['content'][:100]}...")
    
    # 통계 출력
    chunker.print_stats()

