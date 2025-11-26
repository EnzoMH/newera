"""
하이브리드 Agentic Chunking 테스트
- 로컬 LLM (Qwen2.5-3B-Korean) vs Gemini 2.0 Flash
- 비용/성능 비교
"""
import sys
from pathlib import Path
import logging
import os
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.vecdb.hybrid_agentic_chunker import HybridAgenticChunker, LLMBackend

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_hybrid_chunking():
    """하이브리드 청킹 테스트"""
    
    # 샘플 ArXiv 논문 텍스트 로드
    chunks_dir = Path("data/chunks")
    chunk_files = list(chunks_dir.glob("chunks_*.json"))
    
    if not chunk_files:
        logger.error("청크 파일이 없습니다")
        logger.info("먼저 scripts/preprocess_pdfs.py를 실행하세요")
        return
    
    # 첫 번째 논문 사용
    with open(chunk_files[0], 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 청크 내용 합쳐서 원본 텍스트 재구성 (일부만)
    chunks = data.get('chunks', [])[:10]  # 처음 10개만
    sample_text = '\n\n'.join([c['content'] for c in chunks])
    
    print("="*80)
    print("하이브리드 Agentic Chunking 테스트")
    print("="*80)
    print(f"\n테스트 문서: {data['filename']}")
    print(f"도메인: {data['domain']}")
    print(f"텍스트 길이: {len(sample_text):,} chars")
    print()
    
    # 하이브리드 Chunker 생성
    try:
        chunker = HybridAgenticChunker(
            local_model="MyeongHo0621/Qwen2.5-3B-Korean",
            local_model_file="gguf/qwen25-3b-korean-Q4_K_M.gguf",
            gemini_api_key=os.getenv("GOOGLE_API_KEY"),
            backend=LLMBackend.AUTO,  # 로컬 → Gemini fallback
            use_gpu=True
        )
    except Exception as e:
        logger.error(f"Chunker 초기화 실패: {e}")
        logger.info("\n설치 가이드:")
        logger.info("1. llama-cpp-python: pip install llama-cpp-python")
        logger.info("2. GOOGLE_API_KEY 설정: export GOOGLE_API_KEY=your_key")
        return
    
    # 청킹 실행
    print("\n" + "="*80)
    print("청킹 실행 중...")
    print("="*80)
    
    result_chunks = chunker.chunk_text(sample_text, max_chunks=10)
    
    # 결과 출력
    print(f"\n생성된 청크: {len(result_chunks)}개")
    print()
    
    for chunk in result_chunks:
        print(f"\n[청크 {chunk['id']}]")
        print(f"  LLM 백엔드: {chunk.get('llm_backend', 'unknown')}")
        print(f"  LLM 모델: {chunk.get('llm_model', 'unknown')}")
        print(f"  크기: {chunk['size']} chars")
        print(f"  내용 (처음 200자):")
        print(f"  {chunk['content'][:200]}...")
    
    # 통계 출력
    print()
    chunker.print_stats()
    
    # 비용 분석
    stats = chunker.get_stats()
    print("\n" + "="*80)
    print("비용 분석")
    print("="*80)
    
    if stats['local_success'] > 0:
        print(f"✅ 로컬 LLM 성공률: {stats['local_success_rate']:.1%}")
        print(f"  - 절감 비용: ${stats['local_success'] * 0.01:.4f}")
    
    if stats['gemini_success'] > 0:
        print(f"💰 Gemini 사용: {stats['gemini_success']}회")
        print(f"  - 실제 비용: ${stats['total_cost_usd']:.4f}")
    
    total_docs = stats['total_attempts']
    if total_docs > 0:
        full_gemini_cost = total_docs * 0.01
        saved = full_gemini_cost - stats['total_cost_usd']
        saved_percent = (saved / full_gemini_cost * 100) if full_gemini_cost > 0 else 0
        
        print(f"\n하이브리드 효과:")
        print(f"  - Gemini만 사용 시: ${full_gemini_cost:.4f}")
        print(f"  - 하이브리드 사용 시: ${stats['total_cost_usd']:.4f}")
        print(f"  - 절감: ${saved:.4f} ({saved_percent:.1f}%)")
    
    print("\n" + "="*80)
    print("✓ 테스트 완료")
    print("="*80)


if __name__ == "__main__":
    test_hybrid_chunking()

