"""
New-RAG Step 3: Vector DB 구축
- Embedding: BAAI/bge-m3 (1024차원, 다국어)
- FAISS HNSW 인덱스
"""
import sys
from pathlib import Path
import json
import logging
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.vecdb.embedding_service import EmbeddingService
from app.vecdb.faiss_manager import FaissManager
from app.vecdb.local_storage import LocalJSONStorage

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def load_chunks(chunks_dir: Path) -> list[dict]:
    """청크 파일 로드"""
    logger.info(f"[New-RAG] 청크 파일 로드 중: {chunks_dir}")
    
    all_chunks = []
    chunk_files = list(chunks_dir.glob("chunks_*.json"))
    
    logger.info(f"[New-RAG] 발견된 청크 파일: {len(chunk_files)}개")
    
    for chunk_file in chunk_files:
        try:
            with open(chunk_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                for chunk in data.get('chunks', []):
                    chunk_info = {
                        'chunk_id': f"chunk_{len(all_chunks):06d}",
                        'content': chunk['content'],
                        'paper_filename': data['filename'],
                        'domain': data.get('domain', 'General'),
                        'source': data.get('source', 'ArXiv'),
                        'chunk_size': chunk['size'],
                        'chunk_index': chunk['id'],
                    }
                    all_chunks.append(chunk_info)
        
        except Exception as e:
            logger.error(f"[New-RAG] 청크 파일 로드 실패 {chunk_file.name}: {e}")
    
    logger.info(f"[New-RAG] 총 {len(all_chunks)}개 청크 로드 완료")
    return all_chunks


def main():
    parser = argparse.ArgumentParser(description="New-RAG Step 3: Vector DB 구축")
    parser.add_argument(
        "--chunks-dir",
        type=str,
        default="data/new_rag_chunks",
        help="청크 파일 디렉토리"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/new_vecdb",
        help="Vector DB 출력 디렉토리"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="BAAI/bge-m3",
        help="임베딩 모델"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="배치 크기"
    )
    
    args = parser.parse_args()
    
    chunks_dir = Path(args.chunks_dir)
    output_dir = Path(args.output_dir)
    
    if not chunks_dir.exists():
        logger.error(f"[New-RAG] 청크 디렉토리가 없습니다: {chunks_dir}")
        logger.info("먼저 2_chunking.py를 실행하세요")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("New-RAG Vector Database 구축 (개선 버전)")
    print("="*60)
    print(f"- Embedding: {args.model}")
    print("- LangChain 청킹")
    print("- 다국어 지원")
    print("="*60)
    
    logger.info("[New-RAG] 1. 임베딩 서비스 초기화...")
    embedding_service = EmbeddingService(model_name=args.model)
    
    logger.info("[New-RAG] 2. Faiss 인덱스 생성...")
    faiss_manager = FaissManager(dimension=embedding_service.get_dimension())
    
    logger.info("[New-RAG] 3. 로컬 JSON 스토리지 초기화...")
    local_storage = LocalJSONStorage(storage_dir=str(output_dir))
    
    logger.info("[New-RAG] 4. 청크 로드...")
    chunks = load_chunks(chunks_dir)
    
    if not chunks:
        logger.error("[New-RAG] 청크가 없습니다!")
        return
    
    logger.info("[New-RAG] 5. 임베딩 생성 및 인덱싱...")
    batch_size = args.batch_size
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    
    all_documents = []
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(chunks))
        batch_chunks = chunks[start_idx:end_idx]
        
        logger.info(f"[New-RAG] 배치 {batch_idx + 1}/{total_batches} 처리 중... ({len(batch_chunks)}개)")
        
        texts = [chunk['content'] for chunk in batch_chunks]
        embeddings = embedding_service.embed_documents(texts)
        
        faiss_manager.add_vectors(embeddings, batch_chunks)
        
        all_documents.extend(batch_chunks)
    
    logger.info("[New-RAG] 6. 메타데이터 저장...")
    local_storage.insert_many(all_documents)
    
    logger.info("[New-RAG] 7. Faiss 인덱스 저장...")
    index_path = output_dir / "faiss.index"
    faiss_manager.save(str(index_path))
    
    logger.info("="*60)
    logger.info("✓ [New-RAG] Vector Database 구축 완료!")
    logger.info(f"  - 모델: {args.model}")
    logger.info(f"  - 차원: {embedding_service.get_dimension()}")
    logger.info(f"  - 총 문서: {len(all_documents)}개")
    logger.info(f"  - Faiss 인덱스: {index_path}")
    logger.info(f"  - 메타데이터: {output_dir}/")
    logger.info("="*60)
    
    stats = local_storage.get_stats()
    logger.info(f"[New-RAG] 통계: {json.dumps(stats, indent=2, ensure_ascii=False)}")


if __name__ == "__main__":
    main()

