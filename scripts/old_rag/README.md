# Old-RAG 파이프라인 (Baseline)

## 개요

비교 기준점(Baseline)으로 사용되는 Old-RAG 버전입니다.

### 사양

- **임베딩**: jhgan/ko-sroberta-multitask (768차원)
- **청킹**: 500 문자 고정, 100 오버랩
- **파싱**: PyMuPDF만 사용
- **언어**: 한국어 특화

## 실행 방법

### Vector DB 구축

```bash
python scripts/old_rag/build_vectordb_old.py
```

기존 `data/chunks/` 디렉토리의 청크를 재사용합니다.

## 출력

```
data/old_vecdb/
├── faiss.index        # FAISS 인덱스 (768차원)
├── documents.json     # 문서 저장소
└── metadata.json      # 메타데이터
```

## 용도

New-RAG와의 성능 비교 기준점으로 사용됩니다.

```bash
# 벤치마크 실행
python scripts/benchmark_compare.py
```

