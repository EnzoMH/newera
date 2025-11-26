# New-RAG 파이프라인 (개선 버전)

## 개요

improvement.md에 기반한 개선된 RAG 파이프라인입니다.

### 개선 사항

- **임베딩**: jhgan/ko-sroberta → BAAI/bge-m3 (다국어 지원)
- **청킹**: 500 문자 고정 → 512 토큰 LangChain Recursive
- **모듈화**: 4단계 파이프라인으로 분리

## 실행 방법

### 1. PDF 로딩

```bash
python scripts/new_rag/1_load_pdf.py \
    --input-dir data/crawled/ArXiv \
    --output-dir data/new_rag_texts
```

### 2. 청킹

```bash
python scripts/new_rag/2_chunking.py \
    --input-dir data/new_rag_texts \
    --output-dir data/new_rag_chunks \
    --chunk-size 512 \
    --chunk-overlap 50
```

### 3. Vector DB 구축

```bash
python scripts/new_rag/3_build_vectordb.py \
    --chunks-dir data/new_rag_chunks \
    --output-dir data/new_vecdb \
    --model BAAI/bge-m3 \
    --batch-size 50
```

## 파라미터 튜닝

### 청킹 파라미터 실험

```bash
# 큰 청크 (긴 문맥)
python scripts/new_rag/2_chunking.py --chunk-size 1024 --chunk-overlap 100

# 작은 청크 (정확한 검색)
python scripts/new_rag/2_chunking.py --chunk-size 256 --chunk-overlap 25
```

### 임베딩 모델 변경

```bash
# multilingual-e5-large
python scripts/new_rag/3_build_vectordb.py --model intfloat/multilingual-e5-large

# ko-sroberta (한국어 특화)
python scripts/new_rag/3_build_vectordb.py --model jhgan/ko-sroberta-multitask
```

## 디렉토리 구조

```
data/
├── new_rag_texts/         # Step 1 출력
│   └── text_*.json
├── new_rag_chunks/        # Step 2 출력
│   └── chunks_*.json
└── new_vecdb/             # Step 3 출력
    ├── faiss.index
    ├── documents.json
    └── metadata.json
```

## 기술 스택

- **PDF Parsing**: PyMuPDF (fitz)
- **Chunking**: LangChain RecursiveCharacterTextSplitter
- **Embedding**: BAAI/bge-m3 (1024차원)
- **Vector DB**: FAISS HNSW (GPU 지원)

