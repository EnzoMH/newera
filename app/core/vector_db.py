"""
FAISS 기반 Vector Database
반도체 도메인 문서의 벡터 검색을 위한 FAISS 구현
"""
import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

logger = logging.getLogger(__name__)

# FAISS GPU 검사
try:
    GPU_AVAILABLE = hasattr(faiss, 'StandardGpuResources') and faiss.get_num_gpus() > 0
    if GPU_AVAILABLE:
        logger.info(f" FAISS GPU 사용 가능 ({faiss.get_num_gpus()}개 GPU 감지)")
    else:
        logger.info(" FAISS CPU 모드")
except Exception as e:
    GPU_AVAILABLE = False
    logger.warning(f" FAISS GPU 체크 실패, CPU 모드로 실행: {e}")

from ..memory.conversation_simple import SimpleConversationMemory

logger = logging.getLogger(__name__)


class SentenceTransformerEmbeddings:
    """LangChain 호환 SentenceTransformer 임베딩 래퍼"""

    def __init__(self, model):
        self.model = model

    def __call__(self, texts: List[str]) -> List[List[float]]:
        """FAISS 호환 호출 메소드"""
        return self.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        """단일 텍스트 임베딩"""
        return self.model.encode(text).tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """여러 텍스트 임베딩"""
        return self.model.encode(texts).tolist()


class FAISSVectorDB:
    """
    FAISS 기반 벡터 데이터베이스 (GPU 최적화)
    - Sentence Transformers 임베딩 사용
    - GPU 가속 인덱싱 (IVF-PQ, Flat)
    - 고성능 유사도 검색
    """

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2", 
        index_path: str = "app/data/vectorstore/faiss_index",
        persist_directory: str = "app/data/vectorstore",
        index_type: str = "auto",  # "auto", "flat", "ivf_pq", "hnsw"
        use_gpu: bool = True
    ):
        self.embedding_model = embedding_model
        self.index_path = Path(index_path)
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.index_type = index_type
        self.use_gpu = use_gpu and GPU_AVAILABLE

        # Sentence Transformer 임베딩 모델 초기화
        self.embeddings_model = SentenceTransformer(embedding_model)
        self.embedding_dim = self.embeddings_model.get_sentence_embedding_dimension()

        # LangChain 호환 임베딩 래퍼
        self.embeddings = SentenceTransformerEmbeddings(self.embeddings_model)

        # FAISS 벡터 스토어
        self.vectorstore: Optional[FAISS] = None

        # GPU 리소스 (GPU 사용시)
        self.gpu_resource = None
        if self.use_gpu:
            try:
                self.gpu_resource = faiss.StandardGpuResources()
                logger.info("🚀 FAISS GPU 리소스 초기화 완료")
            except Exception as e:
                logger.warning(f"⚠️ GPU 리소스 초기화 실패, CPU 모드로 전환: {e}")
                self.use_gpu = False

        # 메타데이터 저장
        self.metadata_file = self.persist_directory / "metadata.json"

        logger.info(f"🎯 FAISS VectorDB 초기화: {embedding_model} ({'GPU' if self.use_gpu else 'CPU'} 모드, {self.index_type} 인덱스)")

    def initialize(self) -> bool:
        """벡터 데이터베이스 초기화"""
        try:
            # 기존 인덱스 로드 시도
            if self._load_index():
                logger.info("✅ 기존 FAISS 인덱스 로드 성공")
                return True

            # 새 인덱스 생성
            self._create_empty_index()
            logger.info("✅ 새 FAISS 인덱스 생성")
            return True

        except Exception as e:
            logger.error(f"❌ FAISS VectorDB 초기화 실패: {e}")
            return False

    def _create_empty_index(self):
        """빈 FAISS 인덱스 생성 (GPU 최적화, 다양한 인덱스 타입 지원)"""
        # 인덱스 타입 결정
        if self.index_type == "auto":
            # 자동 선택: GPU 사용 시 Flat, 아니면 IVF-PQ
            index_type = "flat" if self.use_gpu else "ivf_pq"
        else:
            index_type = self.index_type

        # FAISS 인덱스 생성
        if index_type == "flat":
            # Inner Product for cosine similarity (L2 정규화 임베딩 필요)
            index = faiss.IndexFlatIP(self.embedding_dim)
            logger.info("📍 Flat 인덱스 (정확한 검색, 빠른 소규모 DB)")
            
        elif index_type == "ivf_pq":
            # IVF-PQ 인덱스 (메모리 효율적, 대용량 DB용)
            nlist = min(100, max(4, int(np.sqrt(10000))))  # 클러스터 수 (최소 4, 최대 100)
            m = 8        # PQ 세그먼트 수
            nbits = 8    # 비트 수
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            index = faiss.IndexIVFPQ(quantizer, self.embedding_dim, nlist, m, nbits)
            # IVF 인덱스는 train이 필요하지만 빈 상태에서는 스킵
            logger.info(f"🗂️ IVF-PQ 인덱스 (메모리 효율, nlist={nlist})")
            
        elif index_type == "hnsw":
            # HNSW 인덱스 (빠른 근사 검색)
            M = 32  # 연결 수
            index = faiss.IndexHNSWFlat(self.embedding_dim, M)
            index.hnsw.efConstruction = 200  # 구축 시 탐색 깊이
            index.hnsw.efSearch = 100        # 검색 시 탐색 깊이
            logger.info(f"🕸️ HNSW 인덱스 (빠른 근사 검색, M={M})")
            
        else:
            # 기본값: Flat
            index = faiss.IndexFlatIP(self.embedding_dim)
            logger.info("📍 기본 Flat 인덱스")

        # GPU 사용 시 GPU로 이동
        if self.use_gpu and index_type != "ivf_pq":  # IVF-PQ는 GPU 지원 제한적
            try:
                gpu_index = faiss.index_cpu_to_gpu(self.gpu_resource, 0, index)
                index = gpu_index
                logger.info(f"🚀 {index_type.upper()} 인덱스 GPU로 이동 완료")
            except Exception as e:
                logger.warning(f"⚠️ GPU 이동 실패, CPU에서 실행: {e}")

        # LangChain FAISS 래퍼로 생성
        self.vectorstore = FAISS(
            embedding_function=self.embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}
        )

        logger.info(f"✅ {index_type.upper()} 인덱스 생성 완료 (차원: {self.embedding_dim}, {'GPU' if self.use_gpu and index_type != 'ivf_pq' else 'CPU'})")

    def _load_index(self) -> bool:
        """기존 FAISS 인덱스 로드"""
        try:
            if not self.index_path.exists():
                return False

            self.vectorstore = FAISS.load_local(
                str(self.index_path),
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            return True

        except Exception as e:
            logger.warning(f"기존 인덱스 로드 실패: {e}")
            return False

    def save_index(self):
        """FAISS 인덱스 저장"""
        try:
            if self.vectorstore:
                self.vectorstore.save_local(str(self.index_path))
                logger.info(f"💾 FAISS 인덱스 저장: {self.index_path}")
        except Exception as e:
            logger.error(f"❌ 인덱스 저장 실패: {e}")

    def add_documents(self, documents: List[Document]):
        """문서 추가 및 인덱싱"""
        try:
            if not self.vectorstore:
                logger.error("벡터 스토어가 초기화되지 않았습니다")
                return False

            # 문서 추가
            self.vectorstore.add_documents(documents)

            # 인덱스 저장
            self.save_index()

            logger.info(f"✅ 문서 {len(documents)}개 추가 및 인덱싱 완료")
            return True

        except Exception as e:
            logger.error(f"❌ 문서 추가 실패: {e}")
            return False

    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None):
        """텍스트 직접 추가"""
        try:
            if not self.vectorstore:
                logger.error("벡터 스토어가 초기화되지 않았습니다")
                return False

            # 메타데이터 기본값 설정
            if metadatas is None:
                metadatas = [{"source": f"text_{i}", "chunk_id": i} for i in range(len(texts))]

            # 텍스트 추가
            self.vectorstore.add_texts(texts, metadatas=metadatas)

            # 인덱스 저장
            self.save_index()

            logger.info(f"✅ 텍스트 {len(texts)}개 추가 및 인덱싱 완료")
            return True

        except Exception as e:
            logger.error(f"❌ 텍스트 추가 실패: {e}")
            return False

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        score_threshold: float = 0.0
    ) -> List[Tuple[Document, float]]:
        """유사도 검색"""
        try:
            if not self.vectorstore:
                logger.error("벡터 스토어가 초기화되지 않았습니다")
                return []

            # 유사도 검색 (점수 포함)
            docs_and_scores = self.vectorstore.similarity_search_with_score(
                query,
                k=k
            )

            # 점수 필터링
            filtered_results = [
                (doc, score) for doc, score in docs_and_scores
                if score >= score_threshold
            ]

            logger.info(f"🔍 유사도 검색 완료: {len(filtered_results)}개 결과 (쿼리: {query[:50]}...)")
            return filtered_results

        except Exception as e:
            logger.error(f"❌ 유사도 검색 실패: {e}")
            return []

    def get_stats(self) -> Dict[str, Any]:
        """벡터 DB 통계"""
        try:
            if not self.vectorstore:
                return {"status": "not_initialized"}

            # 기본 통계
            stats = {
                "status": "initialized",
                "embedding_model": self.embedding_model,
                "index_path": str(self.index_path),
                "total_documents": len(self.vectorstore.docstore._dict) if hasattr(self.vectorstore.docstore, '_dict') else 0
            }

            return stats

        except Exception as e:
            logger.error(f"통계 조회 실패: {e}")
            return {"status": "error", "error": str(e)}

    def clear_index(self):
        """인덱스 초기화"""
        try:
            self._create_empty_index()
            self.save_index()
            logger.info("🧹 FAISS 인덱스 초기화 완료")
        except Exception as e:
            logger.error(f"❌ 인덱스 초기화 실패: {e}")


# 전역 인스턴스
_vector_db_instance: Optional[FAISSVectorDB] = None

def get_vector_db() -> FAISSVectorDB:
    """FAISS VectorDB 싱글톤 인스턴스"""
    global _vector_db_instance
    if _vector_db_instance is None:
        _vector_db_instance = FAISSVectorDB()
        _vector_db_instance.initialize()
    return _vector_db_instance

def initialize_sample_data():
    """샘플 반도체 문서 데이터 추가"""
    vector_db = get_vector_db()

    # 샘플 반도체 문서들
    sample_documents = [
        Document(
            page_content="반도체 제조 공정은 크게 8단계로 나뉩니다: 웨이퍼 제조, 산화, 포토리소그래피, 식각, 이온주입, 금속화, 패시베이션, 패키징입니다.",
            metadata={"source": "semiconductor_fundamentals.pdf", "chunk_id": 1, "topic": "제조공정"}
        ),
        Document(
            page_content="VirtualFab은 반도체 공장을 가상으로 시뮬레이션하는 Digital Twin 기술입니다. 이를 통해 공정 최적화와 품질 향상을 실현할 수 있습니다.",
            metadata={"source": "virtualfab_guide.pdf", "chunk_id": 2, "topic": "VirtualFab"}
        ),
        Document(
            page_content="Digital Twin은 물리적 시스템의 가상 복제본으로, 실시간 모니터링과 예측 최적화를 가능하게 합니다. 반도체 산업에서 특히 유용합니다.",
            metadata={"source": "digital_twin_overview.pdf", "chunk_id": 3, "topic": "DigitalTwin"}
        ),
        Document(
            page_content="반도체 8대 공정: 1) 웨이퍼 제조 2) 산화막 형성 3) 포토리소그래피 4) 식각 5) 이온주입 6) 금속 배선 7) 패시베이션 8) 패키징",
            metadata={"source": "process_guide.pdf", "chunk_id": 4, "topic": "8대공정"}
        ),
        Document(
            page_content="VirtualFab 플랫폼은 클라우드 기반 반도체 설계 및 시뮬레이션 환경을 제공합니다. AI 기반 최적화 알고리즘을 활용합니다.",
            metadata={"source": "platform_features.pdf", "chunk_id": 5, "topic": "플랫폼"}
        )
    ]

    # 문서 추가
    success = vector_db.add_documents(sample_documents)

    if success:
        logger.info("✅ 샘플 반도체 문서 데이터 추가 완료")
    else:
        logger.error("❌ 샘플 데이터 추가 실패")

    return success
