"""
로컬 JSON 기반 메타데이터 스토리지 (MongoDB 대체)
"""
import json
import logging
from pathlib import Path
from typing import Any
from datetime import datetime

from app.vecdb.interfaces import MetadataStore


logger = logging.getLogger(__name__)


class LocalJSONStorage(MetadataStore):
    """
    로컬 JSON 파일 기반 메타데이터 스토리지
    MongoDB 없이 빠른 프로토타이핑용
    """
    
    def __init__(self, storage_dir: str = "data/local_vecdb"):
        """
        Args:
            storage_dir: JSON 파일 저장 디렉토리
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.documents_file = self.storage_dir / "documents.json"
        self.metadata_file = self.storage_dir / "metadata.json"
        
        self.documents: list[str] = []
        self.metadatas: list[dict[str, Any]] = []
        
        self._load_data()
        
        logger.info(f"✓ Local JSON Storage 초기화")
        logger.info(f"  - 저장 위치: {self.storage_dir}")
        logger.info(f"  - 문서 수: {len(self.documents)}")
    
    def _load_data(self) -> None:
        """기존 데이터 로드"""
        if self.documents_file.exists():
            with open(self.documents_file, 'r', encoding='utf-8') as f:
                self.documents = json.load(f)
        
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                self.metadatas = json.load(f)
    
    def _save_data(self) -> None:
        """데이터 저장"""
        with open(self.documents_file, 'w', encoding='utf-8') as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)
        
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadatas, f, ensure_ascii=False, indent=2)
    
    def insert_many(self, documents: list[dict[str, Any]]) -> list[str]:
        """
        다중 문서 삽입
        
        Args:
            documents: 문서 리스트 (content + metadata)
            
        Returns:
            삽입된 문서 인덱스 리스트
        """
        if not documents:
            return []
        
        ids = []
        for doc in documents:
            doc_id = f"doc_{len(self.documents)}"
            
            # content 저장
            self.documents.append(doc.get('content', ''))
            
            # metadata 저장
            metadata = {
                'id': doc_id,
                'chunk_id': doc.get('chunk_id', doc_id),
                'paper_filename': doc.get('paper_filename', ''),
                'domain': doc.get('domain', ''),
                'source': doc.get('source', ''),
                'chunk_size': doc.get('chunk_size', 0),
                'created_at': datetime.now().isoformat(),
            }
            self.metadatas.append(metadata)
            ids.append(doc_id)
        
        self._save_data()
        logger.info(f"로컬 저장소에 {len(documents)}개 문서 추가")
        
        return ids
    
    def find_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """ID로 검색"""
        results = []
        for doc_id in ids:
            for i, meta in enumerate(self.metadatas):
                if meta['id'] == doc_id:
                    results.append({
                        'content': self.documents[i] if i < len(self.documents) else '',
                        **meta
                    })
                    break
        return results
    
    def find_by_chunk_ids(self, chunk_ids: list[str]) -> list[dict[str, Any]]:
        """청크 ID로 검색"""
        results = []
        for chunk_id in chunk_ids:
            for i, meta in enumerate(self.metadatas):
                if meta.get('chunk_id') == chunk_id:
                    results.append({
                        'content': self.documents[i] if i < len(self.documents) else '',
                        **meta
                    })
                    break
        return results
    
    def find_by_indices(self, indices: list[int]) -> list[dict[str, Any]]:
        """
        인덱스로 검색 (Faiss 검색 결과용)
        
        Args:
            indices: Faiss에서 반환된 인덱스 리스트
            
        Returns:
            문서 리스트
        """
        results = []
        for idx in indices:
            if 0 <= idx < len(self.documents):
                metadata = self.metadatas[idx] if idx < len(self.metadatas) else {}
                result = {'content': self.documents[idx], **metadata}
                results.append(result)
        return results
    
    def update_one(self, doc_id: str, update_data: dict[str, Any]) -> bool:
        """문서 업데이트"""
        for i, meta in enumerate(self.metadatas):
            if meta['id'] == doc_id:
                self.metadatas[i].update(update_data)
                self.metadatas[i]['updated_at'] = datetime.now().isoformat()
                self._save_data()
                return True
        return False
    
    def count(self) -> int:
        """문서 수 반환"""
        return len(self.documents)
    
    def get_stats(self) -> dict[str, Any]:
        """통계 반환"""
        domain_stats: dict[str, int] = {}
        for meta in self.metadatas:
            domain = meta.get('domain', 'Unknown')
            domain_stats[domain] = domain_stats.get(domain, 0) + 1
        
        return {
            'total_documents': len(self.documents),
            'domains': domain_stats,
            'storage_type': 'local_json',
            'storage_path': str(self.storage_dir),
        }
    
    def close(self) -> None:
        """연결 종료 (JSON이므로 불필요)"""
        logger.info("Local JSON Storage 종료")

