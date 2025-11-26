"""
벡터 데이터베이스 구축 스크립트
"""
import sys
from pathlib import Path
import argparse
import logging
import json
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.vecdb.embedding_service import EmbeddingService
from app.vecdb.faiss_manager import FaissManager
from app.vecdb.mongodb_client import MongoDBClient
from base_config import (
    MONGODB_URI,
    MONGODB_DATABASE,
    MONGODB_COLLECTION,
    EMBEDDING_MODEL,
    FAISS_INDEX_PATH,
)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def load_chunks(chunks_dir: Path) -> list[dict[str, Any]]:
    """청크 파일 로드"""
    logger.info(f"청크 파일 로드 중: {chunks_dir}")
    
    all_chunks: list[dict[str, Any]] = []
    chunk_files = list(chunks_dir.glob("chunks_*.json"))
    
    logger.info(f"발견된 청크 파일: {len(chunk_files)}개")
    
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
                        'sentences': chunk.get('sentences', 1),
                        'chunk_index': chunk['id'],
                    }
                    all_chunks.append(chunk_info)
        
        except Exception as e:
            logger.error(f"청크 파일 로드 실패 {chunk_file.name}: {e}")
    
    logger.info(f"총 {len(all_chunks)}개 청크 로드 완료")
    return all_chunks


def main():
    parser = argparse.ArgumentParser(description="벡터 데이터베이스 구축")
    parser.add_argument(
        "--chunks-dir",
        type=str,
        default="data/chunks",
        help="청크 파일 디렉토리"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="배치 크기"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("벡터 데이터베이스 구축")
    print("="*60)
    
    chunks_dir = Path(args.chunks_dir)
    if not chunks_dir.exists():
        logger.error(f"청크 디렉토리가 없습니다: {chunks_dir}")
        return
    
    logger.info("1. 임베딩 서비스 초기화...")
    embedding_service = EmbeddingService(model_name=EMBEDDING_MODEL)
    
    logger.info("2. Faiss 인덱스 생성...")
    faiss_manager = FaissManager(dimension=embedding_service.get_dimension())
    
    logger.info("3. MongoDB 연결...")
    mongodb_client = MongoDBClient(
        connection_string=MONGODB_URI,
        database_name=MONGODB_DATABASE,
        collection_name=MONGODB_COLLECTION
    )
    
    logger.info("4. 청크 로드...")
    chunks = load_chunks(chunks_dir)
    
    if not chunks:
        logger.error("청크가 없습니다!")
        return
    
    logger.info("5. 임베딩 생성 및 인덱싱...")
    total_batches = (len(chunks) + args.batch_size - 1) // args.batch_size
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * args.batch_size
        end_idx = min((batch_idx + 1) * args.batch_size, len(chunks))
        batch_chunks = chunks[start_idx:end_idx]
        
        logger.info(f"배치 {batch_idx + 1}/{total_batches} 처리 중... ({len(batch_chunks)}개)")
        
        texts = [chunk['content'] for chunk in batch_chunks]
        
        embeddings = embedding_service.embed_documents(texts)
        
        faiss_manager.add_vectors(embeddings, batch_chunks)
        
        mongodb_client.insert_many(batch_chunks)
    
    logger.info("6. Faiss 인덱스 저장...")
    faiss_manager.save(FAISS_INDEX_PATH)
    
    logger.info("="*60)
    logger.info("✓ 벡터 데이터베이스 구축 완료!")
    logger.info(f"  - 총 문서: {len(chunks)}개")
    logger.info(f"  - Faiss 인덱스: {FAISS_INDEX_PATH}")
    logger.info(f"  - MongoDB: {MONGODB_DATABASE}.{MONGODB_COLLECTION}")
    logger.info("="*60)
    
    mongodb_stats = mongodb_client.get_stats()
    logger.info(f"MongoDB 통계: {json.dumps(mongodb_stats, indent=2, ensure_ascii=False)}")
    
    mongodb_client.close()


if __name__ == "__main__":
    main()

