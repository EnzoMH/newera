# newera - VirtualFab RAG System

반도체 제조(VirtualFab/Digital Twin) 도메인 특화 RAG 시스템

## 프로젝트 개요

ArXiv 논문을 크롤링하여 FAISS Vector DB를 구축하고, Gemini/Ollama LLM과 연계한 질의응답 시스템입니다.

## 주요 기능

- ✅ ArXiv 논문 자동 크롤링
- ✅ PDF 전처리 및 청킹
- ✅ 다국어 임베딩 (BAAI/bge-m3)
- ✅ FAISS GPU HNSW 인덱스
- ✅ FastAPI 기반 REST API
- ✅ Gemini/Ollama LLM 통합
- ✅ Old-RAG vs New-RAG 벤치마크

## 아키텍처

```
PDF 문서 → 청킹 → 임베딩 → Vector DB → 검색 & 생성
   ①        ②       ③         ④          ⑤
```

## 기술 스택

### Core
- **Python**: 3.12.7
- **Framework**: FastAPI, LangChain
- **Vector DB**: FAISS (GPU HNSW)
- **LLM**: Google Gemini 1.5 Pro, Ollama

### Embedding Models
- **New-RAG**: BAAI/bge-m3 (1024차원, 다국어)
- **Old-RAG**: jhgan/ko-sroberta-multitask (768차원, 한국어)

### Libraries
- `sentence-transformers`: 임베딩
- `langchain`: RAG 파이프라인
- `pymupdf`: PDF 파싱
- `faiss-gpu`: 벡터 검색

## 디렉토리 구조

```
newera/
├── app/
│   ├── chat/                    # FastAPI 라우터
│   │   ├── router/
│   │   ├── services/
│   │   └── dto/
│   ├── crawl/                   # 크롤러
│   │   ├── arxiv_crawler.py
│   │   └── preprocessor.py
│   └── vecdb/                   # Vector DB
│       ├── old/                 # Old-RAG (Baseline)
│       ├── embedding_service.py # New-RAG
│       ├── chunking_service.py
│       ├── faiss_manager.py
│       └── retriever.py
│
├── scripts/
│   ├── old_rag/                 # Old-RAG 파이프라인
│   │   └── build_vectordb_old.py
│   ├── new_rag/                 # New-RAG 파이프라인
│   │   ├── 1_load_pdf.py
│   │   ├── 2_chunking.py
│   │   └── 3_build_vectordb.py
│   ├── benchmark_compare.py     # 벤치마크
│   ├── crawl_arxiv.py
│   └── preprocess_pdfs.py
│
├── data/
│   ├── chunks/                  # 전처리된 청크
│   ├── old_vecdb/               # Old-RAG Vector DB
│   ├── new_vecdb/               # New-RAG Vector DB
│   └── crawled/
│
├── base_config.py
├── main.py
├── rag_main.py
└── requirements.txt
```

## 빠른 시작

### 1. 설치

```bash
# 의존성 설치
pip install -r requirements.txt

# CUDA 설정 확인 (선택)
python check_cuda_setup.py
```

### 2. Old-RAG vs New-RAG 구축

#### Old-RAG (Baseline)

```bash
python scripts/old_rag/build_vectordb_old.py
```

#### New-RAG (개선 버전)

```bash
# Step 1: PDF 로딩
python scripts/new_rag/1_load_pdf.py

# Step 2: 청킹
python scripts/new_rag/2_chunking.py

# Step 3: Vector DB 구축
python scripts/new_rag/3_build_vectordb.py
```

### 3. 벤치마크 실행

```bash
python scripts/benchmark_compare.py
```

### 4. API 서버 실행

```bash
# Gemini API 키 설정
export GOOGLE_API_KEY=your_key

# 서버 시작
python main.py
```

## 성능 비교: Old-RAG vs New-RAG

### Old-RAG (Baseline)

| 항목 | 사양 |
|------|------|
| 임베딩 | jhgan/ko-sroberta-multitask (768차원) |
| 청킹 | 500 문자 고정, 수동 구현 |
| 언어 | 한국어 특화 |

### New-RAG (개선 버전)

| 항목 | 사양 |
|------|------|
| 임베딩 | BAAI/bge-m3 (1024차원) |
| 청킹 | 512 토큰, LangChain Recursive |
| 언어 | 100+ 다국어 지원 |

### 개선 효과

- ✅ **다국어 지원**: 영문 논문 검색 정확도 향상
- ✅ **구조 보존**: Recursive 청킹으로 문맥 유지
- ✅ **모듈화**: 각 단계 독립 실행 가능
- ✅ **확장성**: 8192 토큰 컨텍스트 지원

## 주요 성과

- 🚀 **FAISS GPU 최적화**: CPU → GPU 마이그레이션
- 📊 **23개 논문 처리**: 3647개 청크 생성
- 🔧 **모듈화 파이프라인**: 실험 및 튜닝 용이

## 크롤링

### ArXiv 논문 크롤링

```bash
python scripts/crawl_arxiv.py
```

크롤링된 논문은 `data/crawled/ArXiv/`에 저장됩니다.

## API 사용

### RAG 쿼리

```bash
curl -X POST http://localhost:8000/api/rag/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "VirtualFab이란 무엇인가?",
    "top_k": 5,
    "use_gemini": true
  }'
```

## 환경 변수

```bash
# .env 파일 예시
GOOGLE_API_KEY=your_gemini_api_key
OLLAMA_BASE_URL=http://localhost:11434
MODEL_NAME=exaone
EMBEDDING_MODEL=BAAI/bge-m3
```

## 개발 환경

- **OS**: Windows 10 / Linux
- **Python**: 3.12.7
- **CUDA**: 11.8+ (GPU 사용 시)
- **RAM**: 16GB+ 권장
- **GPU**: NVIDIA RTX 시리즈 권장

## 라이선스

MIT License

## 문서

- [improvement.md](improvement.md): RAG 파이프라인 4단계 상세 가이드
- [CUDA_SETUP_GUIDE.md](CUDA_SETUP_GUIDE.md): CUDA 설정 가이드
- [scripts/old_rag/README.md](scripts/old_rag/README.md): Old-RAG 가이드
- [scripts/new_rag/README.md](scripts/new_rag/README.md): New-RAG 가이드

## 기여

면접 준비 및 포트폴리오 프로젝트입니다.

## 참고

- 브레인크루(Teddynote Labs) 채용 대비 프로젝트
- RAG팀 AI Research Engineer 포지션 지원용

