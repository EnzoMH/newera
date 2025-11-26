"""
VectorDB 추상 인터페이스 (SOLID: DIP)
"""
from abc import ABC, abstractmethod
from typing import Any
import numpy as np


class VectorStore(ABC):
    """벡터 스토어 인터페이스"""
    
    @abstractmethod
    def add_vectors(self, vectors: np.ndarray, metadatas: list[dict[str, Any]]) -> None:
        """벡터 추가"""
        pass
    
    @abstractmethod
    def search(self, query_vector: np.ndarray, top_k: int = 5) -> tuple[np.ndarray, np.ndarray]:
        """벡터 검색"""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """인덱스 저장"""
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """인덱스 로드"""
        pass


class EmbeddingProvider(ABC):
    """임베딩 생성 인터페이스"""
    
    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray:
        """단일 텍스트 임베딩"""
        pass
    
    @abstractmethod
    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """배치 임베딩"""
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """임베딩 차원 반환"""
        pass


class MetadataStore(ABC):
    """메타데이터 스토어 인터페이스"""
    
    @abstractmethod
    def insert_many(self, documents: list[dict[str, Any]]) -> list[str]:
        """다중 문서 삽입"""
        pass
    
    @abstractmethod
    def find_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """ID로 검색"""
        pass
    
    @abstractmethod
    def update_one(self, doc_id: str, update_data: dict[str, Any]) -> bool:
        """문서 업데이트"""
        pass
    
    @abstractmethod
    def count(self) -> int:
        """문서 수 반환"""
        pass




