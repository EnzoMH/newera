"""
RAG Retriever (Faiss + MongoDB 통합)
"""
import logging
from typing import Any
import numpy as np

from app.vecdb.faiss_manager import FaissManager
from app.vecdb.mongodb_client import MongoDBClient
from app.vecdb.local_storage import LocalJSONStorage
from app.vecdb.embedding_service import EmbeddingService


logger = logging.getLogger(__name__)


class RAGRetriever:
    """
    RAG 검색기
    - Faiss: 벡터 유사도 검색
    - MongoDB 또는 Local JSON: 메타데이터 조회
    """
    
    def __init__(
        self,
        faiss_manager: FaissManager,
        metadata_store: MongoDBClient | LocalJSONStorage,
        embedding_service: EmbeddingService
    ):
        """
        Args:
            faiss_manager: Faiss 인덱스 관리자
            metadata_store: MongoDB 또는 LocalJSONStorage
            embedding_service: 임베딩 서비스
        """
        self.faiss_manager = faiss_manager
        self.metadata_store = metadata_store
        self.mongodb_client = metadata_store if isinstance(metadata_store, MongoDBClient) else None
        self.embedding_service = embedding_service
        
        storage_type = "MongoDB" if isinstance(metadata_store, MongoDBClient) else "Local JSON"
        logger.info(f"✓ RAG Retriever 초기화 완료 (Storage: {storage_type})")
    
    def retrieve(self, query: str, top_k: int = 5, filter_domain: str | None = None) -> list[dict[str, Any]]:
        """
        쿼리로 관련 문서 검색
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 문서 수
            filter_domain: 도메인 필터 (예: 'VirtualFab', 'FabScheduling')
            
        Returns:
            검색 결과 리스트 (content, metadata, score 포함)
        """
        logger.info(f"검색 쿼리: '{query}' (top_k={top_k})")
        
        query_embedding = self.embedding_service.embed_query(query)
        
        search_k = top_k * 3 if filter_domain else top_k
        distances, indices = self.faiss_manager.search(query_embedding, search_k)
        
        results: list[dict[str, Any]] = []
        
        valid_indices = indices[indices >= 0]
        
        if len(valid_indices) == 0:
            logger.warning("검색 결과 없음")
            return []
        
        # LocalJSONStorage인 경우 인덱스로 직접 검색
        if isinstance(self.metadata_store, LocalJSONStorage):
            documents = self.metadata_store.find_by_indices(valid_indices.tolist())
            doc_map = {i: doc for i, doc in enumerate(documents)}
        else:
            # MongoDB인 경우 chunk_id로 검색
            chunk_ids = [f"chunk_{int(idx):06d}" for idx in valid_indices]
            documents = self.metadata_store.find_by_chunk_ids(chunk_ids)
            doc_map = {doc['chunk_id']: doc for doc in documents}
        
        for i, (idx, distance) in enumerate(zip(valid_indices, distances[:len(valid_indices)])):
            # LocalJSONStorage는 인덱스로, MongoDB는 chunk_id로 접근
            if isinstance(self.metadata_store, LocalJSONStorage):
                if i not in doc_map:
                    continue
                doc = doc_map[i]
                chunk_id = doc.get('chunk_id', f"chunk_{int(idx):06d}")
            else:
                chunk_id = f"chunk_{int(idx):06d}"
                if chunk_id not in doc_map:
                    continue
                doc = doc_map[chunk_id]
            
            if filter_domain and doc.get('domain') != filter_domain:
                continue
            
            similarity = float(1 / (1 + distance))
            
            result = {
                'content': doc.get('content', ''),
                'metadata': {
                    'chunk_id': chunk_id,
                    'paper_filename': doc.get('paper_filename', ''),
                    'domain': doc.get('domain', ''),
                    'source': doc.get('source', ''),
                    'chunk_size': doc.get('chunk_size', 0),
                },
                'score': similarity,
                'distance': float(distance),
            }
            
            results.append(result)
            
            if len(results) >= top_k:
                break
        
        logger.info(f"검색 완료: {len(results)}개 문서 반환")
        
        return results
    
    def retrieve_with_rerank(
        self,
        query: str,
        top_k: int = 5,
        initial_k: int = 20,
        filter_domain: str | None = None
    ) -> list[dict[str, Any]]:
        """
        재순위화(rerank)를 포함한 검색
        
        Args:
            query: 검색 쿼리
            top_k: 최종 반환할 문서 수
            initial_k: 초기 검색 문서 수
            filter_domain: 도메인 필터
            
        Returns:
            재순위화된 검색 결과
        """
        initial_results = self.retrieve(query, initial_k, filter_domain)
        
        if len(initial_results) <= top_k:
            return initial_results
        
        query_embedding = self.embedding_service.embed_query(query)
        
        reranked_results = []
        for result in initial_results:
            content_embedding = self.embedding_service.embed_query(result['content'])
            
            similarity = float(np.dot(query_embedding, content_embedding))
            
            result['rerank_score'] = similarity
            reranked_results.append(result)
        
        reranked_results.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        logger.info(f"재순위화 완료: 상위 {top_k}개 반환")
        
        return reranked_results[:top_k]
    
    def get_stats(self) -> dict[str, Any]:
        """통계 반환"""
        faiss_stats = self.faiss_manager.get_stats()
        metadata_stats = self.metadata_store.get_stats()
        
        return {
            'faiss': faiss_stats,
            'metadata_store': metadata_stats,
            'embedding_dimension': self.embedding_service.get_dimension(),
        }




