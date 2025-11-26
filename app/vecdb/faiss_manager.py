"""
Faiss 벡터 인덱스 관리 (GPU HNSW)
"""
import logging
from pathlib import Path
from typing import Any
import numpy as np
import faiss
import torch

from app.vecdb.interfaces import VectorStore


logger = logging.getLogger(__name__)


class FaissManager(VectorStore):
    """
    Faiss HNSW 벡터 인덱스 관리
    - GPU 가용 시 자동 활성화
    - HNSW 알고리즘 적용
    """
    
    def __init__(self, dimension: int, use_gpu: bool | None = None, hnsw_m: int = 32, ef_construction: int = 200):
        """
        Args:
            dimension: 임베딩 차원
            use_gpu: GPU 사용 여부 (None이면 자동 감지)
            hnsw_m: HNSW M 파라미터 (연결 수)
            ef_construction: HNSW efConstruction 파라미터 (인덱싱 품질)
        """
        self.dimension = dimension
        self.hnsw_m = hnsw_m
        self.ef_construction = ef_construction
        
        if use_gpu is None:
            try:
                self.use_gpu = torch.cuda.is_available() and hasattr(faiss, 'get_num_gpus') and faiss.get_num_gpus() > 0
            except Exception:
                self.use_gpu = False
                logger.warning("Faiss GPU 사용 불가 - CPU 모드로 전환")
        else:
            self.use_gpu = use_gpu
        
        logger.info(f"Faiss 인덱스 초기화")
        logger.info(f"  - 차원: {dimension}")
        logger.info(f"  - GPU: {self.use_gpu}")
        logger.info(f"  - HNSW M: {hnsw_m}")
        logger.info(f"  - efConstruction: {ef_construction}")
        
        self.index = self._create_index()
        self.total_vectors = 0
    
    def _create_index(self) -> faiss.Index:
        """Faiss 인덱스 생성"""
        cpu_index = faiss.IndexHNSWFlat(self.dimension, self.hnsw_m)
        cpu_index.hnsw.efConstruction = self.ef_construction
        cpu_index.hnsw.efSearch = 64
        
        if self.use_gpu:
            try:
                if not hasattr(faiss, 'StandardGpuResources'):
                    logger.warning("Faiss GPU 지원 없음 (faiss-cpu 설치됨) - CPU 사용")
                    self.use_gpu = False
                    return cpu_index
                    
                res = faiss.StandardGpuResources()
                gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
                logger.info("✓ GPU 인덱스 생성 완료")
                return gpu_index
            except Exception as e:
                logger.warning(f"GPU 인덱스 생성 실패, CPU로 fallback: {e}")
                self.use_gpu = False
                return cpu_index
        
        logger.info("✓ CPU 인덱스 생성 완료")
        return cpu_index
    
    def add_vectors(self, vectors: np.ndarray, metadatas: list[dict[str, Any]]) -> None:
        """벡터 추가"""
        if vectors.shape[0] != len(metadatas):
            raise ValueError(f"벡터 수({vectors.shape[0]})와 메타데이터 수({len(metadatas)})가 일치하지 않습니다")
        
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"벡터 차원({vectors.shape[1]})이 인덱스 차원({self.dimension})과 다릅니다")
        
        vectors = vectors.astype(np.float32)
        
        self.index.add(vectors)
        self.total_vectors += vectors.shape[0]
        
        logger.info(f"벡터 추가 완료: {vectors.shape[0]}개 (총 {self.total_vectors}개)")
    
    def search(self, query_vector: np.ndarray, top_k: int = 5) -> tuple[np.ndarray, np.ndarray]:
        """
        벡터 검색
        
        Returns:
            (distances, indices)
        """
        if query_vector.shape[0] != self.dimension:
            raise ValueError(f"쿼리 벡터 차원({query_vector.shape[0]})이 인덱스 차원({self.dimension})과 다릅니다")
        
        query_vector = query_vector.astype(np.float32).reshape(1, -1)
        
        distances, indices = self.index.search(query_vector, top_k)
        
        return distances[0], indices[0]
    
    def save(self, path: str) -> None:
        """인덱스 저장"""
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        if self.use_gpu:
            try:
                cpu_index = faiss.index_gpu_to_cpu(self.index)
                faiss.write_index(cpu_index, str(path_obj))
            except Exception as e:
                logger.error(f"GPU 인덱스 저장 실패: {e}")
                raise
        else:
            faiss.write_index(self.index, str(path_obj))
        
        logger.info(f"Faiss 인덱스 저장: {path_obj}")
    
    def load(self, path: str) -> None:
        """인덱스 로드"""
        path_obj = Path(path)
        
        if not path_obj.exists():
            raise FileNotFoundError(f"인덱스 파일이 없습니다: {path_obj}")
        
        cpu_index = faiss.read_index(str(path_obj))
        
        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
                logger.info(f"✓ GPU 인덱스 로드 완료: {path_obj}")
            except Exception as e:
                logger.warning(f"GPU 인덱스 로드 실패, CPU 사용: {e}")
                self.index = cpu_index
        else:
            self.index = cpu_index
            logger.info(f"✓ CPU 인덱스 로드 완료: {path_obj}")
        
        self.total_vectors = self.index.ntotal
    
    def get_stats(self) -> dict[str, Any]:
        """인덱스 통계"""
        return {
            'total_vectors': self.total_vectors,
            'dimension': self.dimension,
            'index_type': 'HNSW',
            'use_gpu': self.use_gpu,
            'gpu_enabled': self.use_gpu,
            'hnsw_m': self.hnsw_m,
            'ef_construction': self.ef_construction,
        }

