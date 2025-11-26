"""
임베딩 서비스 (BAAI/bge-large-en-v1.5)
"""
import logging
from typing import Any
import numpy as np
import torch
from langchain_huggingface import HuggingFaceEmbeddings

from app.vecdb.interfaces import EmbeddingProvider


logger = logging.getLogger(__name__)


class EmbeddingService(EmbeddingProvider):
    """
    BAAI/bge-large-en-v1.5 임베딩 서비스
    GPU 가용 시 자동 활성화
    """
    
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5", device: str | None = None):
        self.model_name = model_name
        
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        logger.info(f"임베딩 모델 로드 중: {model_name}")
        logger.info(f"디바이스: {self.device}")
        
        try:
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={
                    'device': self.device,
                    'trust_remote_code': True,
                },
                encode_kwargs={
                    'normalize_embeddings': True,
                    'batch_size': 32,
                }
            )
            
            test_embedding = self.embedding_model.embed_query("test")
            self._dimension = len(test_embedding)
            
            logger.info(f"✓ 임베딩 모델 로드 완료")
            logger.info(f"  - 모델: {model_name}")
            logger.info(f"  - 디바이스: {self.device}")
            logger.info(f"  - 차원: {self._dimension}")
            
        except Exception as e:
            logger.error(f"임베딩 모델 로드 실패: {e}")
            raise
    
    def embed_query(self, text: str) -> np.ndarray:
        """단일 텍스트 임베딩"""
        try:
            embedding = self.embedding_model.embed_query(text)
            return np.array(embedding, dtype=np.float32)
        except Exception as e:
            logger.error(f"임베딩 실패: {e}")
            return np.zeros(self._dimension, dtype=np.float32)
    
    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """배치 임베딩"""
        try:
            embeddings = self.embedding_model.embed_documents(texts)
            return np.array(embeddings, dtype=np.float32)
        except Exception as e:
            logger.error(f"배치 임베딩 실패: {e}")
            return np.zeros((len(texts), self._dimension), dtype=np.float32)
    
    def get_dimension(self) -> int:
        """임베딩 차원 반환"""
        return self._dimension




