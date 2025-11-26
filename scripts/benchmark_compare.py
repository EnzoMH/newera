"""
Old-RAG vs New-RAG 벤치마크 비교
- 검색 성능 비교
- 검색 속도 비교
- 결과 품질 비교
"""
import sys
from pathlib import Path
import json
import time
import logging
from typing import Any
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.vecdb.old.embedding_old import EmbeddingServiceOld
from app.vecdb.embedding_service import EmbeddingService
from app.vecdb.faiss_manager import FaissManager
from app.vecdb.local_storage import LocalJSONStorage

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class RAGBenchmark:
    """RAG 시스템 벤치마크"""
    
    def __init__(self, vecdb_path: str, embedding_service: Any, name: str):
        self.name = name
        self.embedding_service = embedding_service
        
        logger.info(f"\n[{name}] 초기화 중...")
        
        self.storage = LocalJSONStorage(storage_dir=vecdb_path)
        
        self.faiss_manager = FaissManager(dimension=embedding_service.get_dimension())
        index_path = Path(vecdb_path) / "faiss.index"
        self.faiss_manager.load(str(index_path))
        
        logger.info(f"✓ [{name}] 로드 완료")
        logger.info(f"  - 총 벡터: {self.faiss_manager.total_vectors}")
        logger.info(f"  - 차원: {self.faiss_manager.dimension}")
    
    def search(self, query: str, top_k: int = 5) -> tuple[list[dict], float]:
        """검색 수행"""
        start_time = time.time()
        
        query_embedding = self.embedding_service.embed_query(query)
        distances, indices = self.faiss_manager.search(query_embedding, top_k)
        
        valid_indices = indices[indices >= 0]
        documents = self.storage.find_by_indices(valid_indices.tolist())
        
        elapsed = time.time() - start_time
        
        results = []
        for idx, (doc, distance) in enumerate(zip(documents, distances[:len(documents)])):
            similarity = float(1 / (1 + distance))
            result = {
                'rank': idx + 1,
                'content': doc.get('content', '')[:200] + "...",
                'filename': doc.get('paper_filename', ''),
                'domain': doc.get('domain', ''),
                'score': similarity,
                'distance': float(distance),
            }
            results.append(result)
        
        return results, elapsed


def run_benchmark():
    """벤치마크 실행"""
    
    print("="*80)
    print("RAG 벤치마크: Old-RAG vs New-RAG")
    print("="*80)
    
    test_queries = [
        "VirtualFab이란 무엇인가?",
        "반도체 제조 공정에서 딥러닝은 어떻게 활용되는가?",
        "EUV lithography optimization 방법은?",
        "Digital Twin in semiconductor manufacturing",
        "Physics-informed neural networks for semiconductor",
    ]
    
    print("\n테스트 쿼리:")
    for i, query in enumerate(test_queries, 1):
        print(f"  {i}. {query}")
    print()
    
    logger.info("Old-RAG 시스템 초기화...")
    old_embedding = EmbeddingServiceOld(model_name="jhgan/ko-sroberta-multitask")
    old_rag = RAGBenchmark("data/old_vecdb", old_embedding, "Old-RAG")
    
    logger.info("\nNew-RAG 시스템 초기화...")
    new_embedding = EmbeddingService(model_name="BAAI/bge-m3")
    new_rag = RAGBenchmark("data/new_vecdb", new_embedding, "New-RAG")
    
    print("\n" + "="*80)
    print("벤치마크 시작")
    print("="*80)
    
    old_times = []
    new_times = []
    
    for query_idx, query in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"쿼리 {query_idx}: {query}")
        print(f"{'='*80}")
        
        logger.info(f"\n[Old-RAG] 검색 중...")
        old_results, old_time = old_rag.search(query, top_k=3)
        old_times.append(old_time)
        
        print(f"\n[Old-RAG] 결과 (검색 시간: {old_time:.4f}초)")
        print(f"  - 모델: jhgan/ko-sroberta-multitask (768차원)")
        print(f"  - 청킹: 500 문자 고정")
        for result in old_results:
            print(f"\n  [{result['rank']}] Score: {result['score']:.4f} | Distance: {result['distance']:.4f}")
            print(f"      파일: {result['filename']}")
            print(f"      도메인: {result['domain']}")
            print(f"      내용: {result['content']}")
        
        logger.info(f"\n[New-RAG] 검색 중...")
        new_results, new_time = new_rag.search(query, top_k=3)
        new_times.append(new_time)
        
        print(f"\n[New-RAG] 결과 (검색 시간: {new_time:.4f}초)")
        print(f"  - 모델: BAAI/bge-m3 (1024차원)")
        print(f"  - 청킹: 512 토큰 (LangChain)")
        for result in new_results:
            print(f"\n  [{result['rank']}] Score: {result['score']:.4f} | Distance: {result['distance']:.4f}")
            print(f"      파일: {result['filename']}")
            print(f"      도메인: {result['domain']}")
            print(f"      내용: {result['content']}")
        
        speedup = old_time / new_time if new_time > 0 else 0
        print(f"\n  ⚡ 속도 비교: Old-RAG {old_time:.4f}초 vs New-RAG {new_time:.4f}초 (x{speedup:.2f})")
    
    print("\n" + "="*80)
    print("최종 벤치마크 결과")
    print("="*80)
    
    old_avg = np.mean(old_times)
    new_avg = np.mean(new_times)
    speedup_avg = old_avg / new_avg if new_avg > 0 else 0
    
    print(f"\n평균 검색 시간:")
    print(f"  Old-RAG: {old_avg:.4f}초")
    print(f"  New-RAG: {new_avg:.4f}초")
    print(f"  속도 향상: x{speedup_avg:.2f}")
    
    print(f"\n모델 비교:")
    print(f"  Old-RAG:")
    print(f"    - 임베딩: jhgan/ko-sroberta-multitask (768차원)")
    print(f"    - 청킹: 500 문자 고정, 수동 구현")
    print(f"    - 언어: 한국어 특화")
    
    print(f"\n  New-RAG:")
    print(f"    - 임베딩: BAAI/bge-m3 (1024차원)")
    print(f"    - 청킹: 512 토큰, LangChain Recursive")
    print(f"    - 언어: 100+ 다국어 지원")
    
    print(f"\n벡터 통계:")
    print(f"  Old-RAG: {old_rag.faiss_manager.total_vectors}개 벡터")
    print(f"  New-RAG: {new_rag.faiss_manager.total_vectors}개 벡터")
    
    print("\n" + "="*80)
    print("✓ 벤치마크 완료")
    print("="*80)
    
    results_summary = {
        'old_rag': {
            'model': 'jhgan/ko-sroberta-multitask',
            'dimension': 768,
            'chunking': '500 chars fixed',
            'avg_search_time': old_avg,
            'total_vectors': old_rag.faiss_manager.total_vectors,
        },
        'new_rag': {
            'model': 'BAAI/bge-m3',
            'dimension': 1024,
            'chunking': '512 tokens LangChain',
            'avg_search_time': new_avg,
            'total_vectors': new_rag.faiss_manager.total_vectors,
        },
        'improvement': {
            'speedup': speedup_avg,
            'dimension_increase': 1024 / 768,
        }
    }
    
    output_file = Path("benchmark_results.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n결과 저장: {output_file}")


if __name__ == "__main__":
    run_benchmark()

