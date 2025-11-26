"""
FAISS 인덱스 팩토리
- 다양한 인덱스 타입 생성 지원
- HNSW, IVF, Flat 선택 가능
"""
import logging
from enum import Enum
import faiss
import numpy as np

logger = logging.getLogger(__name__)


class IndexType(str, Enum):
    """FAISS 인덱스 타입"""
    FLAT = "flat"           # 정확 검색 (작은 데이터셋)
    HNSW = "hnsw"           # 빠른 근사 검색 (프로덕션)
    IVF = "ivf"             # 메모리 효율 (대규모)
    IVF_PQ = "ivf_pq"       # 압축 + 메모리 효율 (초대규모)


class FaissIndexFactory:
    """
    FAISS 인덱스 생성 팩토리
    
    사용 예:
        factory = FaissIndexFactory(dimension=768)
        index = factory.create(IndexType.HNSW)
    """
    
    def __init__(self, dimension: int):
        self.dimension = dimension
        logger.info(f"FAISS 팩토리 초기화: {dimension}차원")
    
    def create(
        self, 
        index_type: IndexType = IndexType.HNSW,
        **kwargs
    ) -> faiss.Index:
        """
        인덱스 생성
        
        Args:
            index_type: 인덱스 타입
            **kwargs: 타입별 추가 파라미터
            
        Returns:
            FAISS 인덱스
        """
        if index_type == IndexType.FLAT:
            return self._create_flat()
        elif index_type == IndexType.HNSW:
            return self._create_hnsw(**kwargs)
        elif index_type == IndexType.IVF:
            return self._create_ivf(**kwargs)
        elif index_type == IndexType.IVF_PQ:
            return self._create_ivf_pq(**kwargs)
        else:
            raise ValueError(f"지원하지 않는 인덱스 타입: {index_type}")
    
    def _create_flat(self) -> faiss.Index:
        """
        Flat 인덱스 생성 (정확 검색)
        
        장점: 100% 정확, 구축 빠름
        단점: 검색 느림 (O(n))
        사용: ~10K 벡터
        """
        index = faiss.IndexFlatL2(self.dimension)
        logger.info("✓ IndexFlatL2 생성 완료")
        logger.info("  - 타입: Flat (정확 검색)")
        logger.info("  - 정확도: 100%")
        logger.info("  - 적합: ~10K 벡터")
        return index
    
    def _create_hnsw(
        self, 
        M: int = 32, 
        efConstruction: int = 200,
        efSearch: int = 64
    ) -> faiss.Index:
        """
        HNSW 인덱스 생성 (빠른 근사 검색)
        
        Args:
            M: 연결 수 (16-64, 높을수록 정확하지만 메모리 증가)
            efConstruction: 구축 품질 (100-500, 높을수록 정확)
            efSearch: 검색 품질 (16-512, 높을수록 정확하지만 느림)
        
        장점: 매우 빠름, 정확도 높음 (99%+)
        단점: 메모리 많이 사용
        사용: ~1M 벡터, 프로덕션
        """
        index = faiss.IndexHNSWFlat(self.dimension, M)
        index.hnsw.efConstruction = efConstruction
        index.hnsw.efSearch = efSearch
        
        logger.info("✓ IndexHNSWFlat 생성 완료")
        logger.info(f"  - 타입: HNSW (빠른 근사 검색)")
        logger.info(f"  - M: {M}")
        logger.info(f"  - efConstruction: {efConstruction}")
        logger.info(f"  - efSearch: {efSearch}")
        logger.info(f"  - 정확도: ~99%")
        logger.info(f"  - 적합: ~1M 벡터")
        return index
    
    def _create_ivf(
        self,
        nlist: int = 100,
        nprobe: int = 10
    ) -> faiss.Index:
        """
        IVF 인덱스 생성 (메모리 효율)
        
        Args:
            nlist: 클러스터 수 (sqrt(N) 권장)
            nprobe: 검색할 클러스터 수 (높을수록 정확하지만 느림)
        
        장점: 메모리 효율, 빠름
        단점: 학습 필요 (train), 정확도 중간 (90-95%)
        사용: 100K+ 벡터
        
        주의: train() 호출 필요!
        """
        quantizer = faiss.IndexFlatL2(self.dimension)
        index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
        index.nprobe = nprobe
        
        logger.info("✓ IndexIVFFlat 생성 완료")
        logger.info(f"  - 타입: IVF (메모리 효율)")
        logger.info(f"  - nlist (클러스터): {nlist}")
        logger.info(f"  - nprobe (검색): {nprobe}")
        logger.info(f"  - 정확도: ~90-95%")
        logger.info(f"  - 적합: 100K+ 벡터")
        logger.warning("  ⚠️  train() 호출 필요!")
        return index
    
    def _create_ivf_pq(
        self,
        nlist: int = 100,
        m: int = 8,
        nbits: int = 8
    ) -> faiss.Index:
        """
        IVF-PQ 인덱스 생성 (압축 + 메모리 효율)
        
        Args:
            nlist: 클러스터 수
            m: PQ 서브벡터 개수 (dimension의 약수여야 함)
            nbits: 서브벡터당 비트 (8 권장)
        
        장점: 메모리 매우 적음 (1/10 ~ 1/100)
        단점: 정확도 낮음 (80-90%), 학습 필요
        사용: 1M+ 벡터, 메모리 제약 심함
        """
        quantizer = faiss.IndexFlatL2(self.dimension)
        index = faiss.IndexIVFPQ(quantizer, self.dimension, nlist, m, nbits)
        
        logger.info("✓ IndexIVFPQ 생성 완료")
        logger.info(f"  - 타입: IVF-PQ (압축 + 메모리 효율)")
        logger.info(f"  - nlist: {nlist}")
        logger.info(f"  - PQ m: {m}")
        logger.info(f"  - nbits: {nbits}")
        logger.info(f"  - 정확도: ~80-90%")
        logger.info(f"  - 메모리: 1/10 ~ 1/100")
        logger.info(f"  - 적합: 1M+ 벡터")
        logger.warning("  ⚠️  train() 호출 필요!")
        return index


def recommend_index_type(num_vectors: int, memory_constraint: bool = False) -> IndexType:
    """
    데이터셋 크기에 따른 최적 인덱스 추천
    
    Args:
        num_vectors: 벡터 개수
        memory_constraint: 메모리 제약 여부
        
    Returns:
        추천 인덱스 타입
    """
    if num_vectors < 10_000:
        # 작은 데이터셋: Flat (정확)
        return IndexType.FLAT
    
    elif num_vectors < 100_000:
        # 중간 데이터셋: HNSW (빠르고 정확)
        if memory_constraint:
            return IndexType.IVF
        return IndexType.HNSW
    
    elif num_vectors < 1_000_000:
        # 큰 데이터셋: IVF
        if memory_constraint:
            return IndexType.IVF_PQ
        return IndexType.IVF
    
    else:
        # 초대규모: IVF-PQ
        return IndexType.IVF_PQ


if __name__ == "__main__":
    # 사용 예시
    logging.basicConfig(level=logging.INFO)
    
    # 현재 newera 프로젝트
    num_vectors = 3647
    dimension = 1024
    
    recommended = recommend_index_type(num_vectors)
    print(f"\n추천 인덱스: {recommended}")
    print(f"  - 벡터 개수: {num_vectors:,}")
    print(f"  - 차원: {dimension}")
    
    factory = FaissIndexFactory(dimension)
    index = factory.create(recommended)
    print(f"\n생성된 인덱스: {type(index)}")

