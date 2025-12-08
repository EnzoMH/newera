# Newera - VirtualFab RAG System

반도체 제조(VirtualFab/Digital Twin) 도메인 특화 RAG 시스템 |
A RAG system specialized for semiconductor manufacturing (VirtualFab/Digital Twin) domains.

## 프로젝트 개요 | Project Overview

LangGraph 기반 Agent 시스템으로, FAISS VectorDB와 Ollama LLM을 통합한 반도체 제조 도메인 특화 AI 어시스턴트입니다.

This is a LangGraph-based agent system that integrates FAISS VectorDB with Ollama LLM, specialized for semiconductor manufacturing domain AI assistant.

## 주요 기능 | Main Features

- ✅ **LangGraph Agent**: 워크플로우 기반 지능형 에이전트 | Workflow-based intelligent agent
- ✅ **FAISS VectorDB**: Sentence Transformers 기반 벡터 검색 | Sentence Transformers-based vector search
- ✅ **LangChain Memory**: 대화 컨텍스트 유지 | Conversation context preservation
- ✅ **Ollama LLM**: 로컬 LLM 통합 (Qwen2.5, Exaone 등) | Local LLM integration (Qwen2.5, Exaone, etc.)
- ✅ **FastAPI REST API**: 고성능 REST API 서버 | High-performance REST API server
- ✅ **MCP 지원**: Model Context Protocol 기반 도구 통합 | Model Context Protocol-based tool integration
- ✅ **Gradio Web UI**: 직관적인 웹 인터페이스 | Intuitive web interface

## 아키텍처 | Architecture

```
사용자 쿼리 → LangGraph Agent → VectorDB 검색 → LLM 생성 → 응답
     ↓              ↓              ↓            ↓         ↓
  REST API     워크플로우 실행    FAISS 검색    Ollama    JSON 응답
```

## 기술 스택 | Technology Stack

### Core Components | 핵심 구성 요소

- **Python**: 3.12.9
- **Agent Framework**: LangGraph (StateGraph 기반 워크플로우) | LangGraph (StateGraph-based workflow)
- **Vector Database**: FAISS CPU/GPU + Sentence Transformers | FAISS CPU/GPU + Sentence Transformers
- **LLM**: Ollama (Qwen2.5 우선, Exaone 대체) | Ollama (Qwen2.5 primary, Exaone fallback)
- **Memory**: LangChain ConversationBufferMemory | LangChain ConversationBufferMemory
- **API Framework**: FastAPI + Pydantic v2 | FastAPI + Pydantic v2
- **Web UI**: Gradio 5.9.1 | Gradio 5.9.1

### 주요 라이브러리 | Key Libraries

- `langgraph`: Agent 워크플로우 관리 | Agent workflow management
- `langchain`: LLM 및 메모리 통합 | LLM and memory integration
- `faiss-cpu`: 고성능 벡터 검색 | High-performance vector search
- `sentence-transformers`: 다국어 텍스트 임베딩 | Multilingual text embedding
- `fastapi`: 비동기 REST API | Asynchronous REST API
- `gradio`: 웹 UI 프레임워크 | Web UI framework

## 디렉토리 구조 | Directory Structure

```
newera/
├── app/
│   ├── agents/                   # LangGraph 에이전트 시스템
│   │   ├── base.py              # 기본 에이전트 클래스
│   │   ├── rag_agent.py         # RAG 특화 에이전트
│   │   └── graph/               # LangGraph 워크플로우
│   │       ├── nodes.py         # 워크플로우 노드들
│   │       ├── state.py         # 상태 정의 (TypedDict)
│   │       └── workflow.py      # 워크플로우 컴파일
│   ├── api/                     # FastAPI REST API
│   │   ├── routers/             # API 엔드포인트
│   │   │   ├── agent.py         # Agent API (/api/agent/*)
│   │   │   ├── rag.py           # RAG API (/api/rag/*)
│   │   │   └── health.py        # 헬스체크 API
│   │   ├── schemas/             # Pydantic 스키마
│   │   │   ├── request.py       # 요청 모델들
│   │   │   └── response.py      # 응답 모델들
│   │   └── dependencies.py      # 의존성 주입
│   ├── core/                    # 핵심 비즈니스 로직
│   │   ├── llm.py              # Ollama LLM Provider
│   │   ├── vector_db.py        # FAISS VectorDB 관리
│   │   ├── rag.py              # RAG 시스템 오케스트레이션
│   │   └── crawler/            # 문서 크롤러 (ArXiv 등)
│   ├── memory/                  # 대화 메모리 시스템
│   │   ├── conversation_simple.py # 간단한 버퍼 메모리
│   │   └── storage/             # 영구 저장소 (MongoDB)
│   ├── tools/                   # LangChain 도구들
│   │   ├── registry.py          # 도구 레지스트리
│   │   ├── rag_tools.py         # RAG 관련 도구들
│   │   └── mcp_tools.py         # MCP 기반 도구들
│   ├── mcp/                     # Model Context Protocol
│   │   ├── server.py            # MCP 서버
│   │   └── config.py            # MCP 설정
│   └── web/                     # 웹 인터페이스
│       └── gradio_ui.py         # Gradio 기반 UI
├── data/
│   └── vectorstore/             # FAISS 인덱스 저장소
│       ├── faiss_index/         # 벡터 인덱스 파일들
│       └── metadata.json        # 메타데이터
├── config.py                    # 환경 설정
├── main.py                      # FastAPI 서버 진입점
├── mcp_server.py               # MCP 서버 독립 실행
├── requirements.txt             # Python 의존성
└── README.md                    # 이 파일
```

## 빠른 시작 | Quick Start

### 1. 환경 설정 | Environment Setup

```bash
# 저장소 클론 (또는 다운로드)
git clone <repository-url>
cd newera

# Python 가상환경 생성 (권장)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 의존성 설치
pip install -r requirements.txt
```

### 2. Ollama LLM 설정 | Ollama LLM Setup

```bash
# Ollama 설치 (https://ollama.ai/download)
# 권장 모델들:
ollama pull hf.co/MyeongHo0621/Qwen2.5-3B-Korean:Q4_K_M  # Qwen2.5 한국어
ollama pull exaone-1.2b:latest                          # Exaone 경량 모델

# 모델 목록 확인
ollama list
```

### 3. 서버 실행 | Run Server

```bash
# 기본 설정으로 서버 시작 (포트 자동 할당)
python main.py

# 또는 수동 포트 지정
API_HOST=0.0.0.0 API_PORT=8000 python main.py
```

### 4. API 테스트 | API Testing

```bash
# 헬스체크
curl http://localhost:8000/health

# Agent 쿼리
curl -X POST http://localhost:8000/api/agent/query \
  -H "Content-Type: application/json" \
  -d '{"question": "반도체 제조 공정에 대해 설명해주세요"}'

# Agent 상태 확인
curl http://localhost:8000/api/agent/status

# 메모리 클리어
curl -X POST http://localhost:8000/api/agent/memory/clear
```

## API 엔드포인트 | API Endpoints

### Agent API | Agent API

- `POST /api/agent/query` - RAG 기반 질의응답 | RAG-based Q&A
- `GET /api/agent/status` - Agent 상태 조회 | Agent status check
- `POST /api/agent/memory/clear` - 대화 메모리 클리어 | Clear conversation memory
- `GET /api/agent/tools` - 사용 가능한 도구 목록 | Available tools list

### RAG API | RAG API

- `POST /api/rag/query` - 직접 RAG 질의 (Agent 우회) | Direct RAG query (bypass agent)

### 시스템 API | System API

- `GET /health` - 헬스체크 | Health check
- `GET /api/system/status` - 시스템 상태 | System status
- `GET /api/system/info` - 시스템 정보 | System info

## 환경 변수 | Environment Variables

```bash
# .env 파일 예시
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=hf.co/MyeongHo0621/Qwen2.5-3B-Korean:Q4_K_M
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS=*

# 선택적 설정
LOG_LEVEL=INFO
RELOAD=false
```

## 개발 환경 | Development Environment

- **운영체제 | OS**: Windows 10/11, Linux, macOS
- **Python**: 3.12.9
- **RAM**: 8GB+ 권장 | 8GB+ recommended
- **저장공간 | Storage**: 5GB+ (모델 및 벡터DB용) | 5GB+ (for models and vector DB)
- **Ollama**: v0.3.0+ | v0.3.0+

## 샘플 쿼리 | Sample Queries

### 반도체 제조 관련 | Semiconductor Manufacturing

```
"반도체 제조 공정에 대해 알려주세요"
"VirtualFab이 무엇인가요?"
"Digital Twin 기술의 장점은?"
"8대 공정 중 식각 공정에 대해 설명해주세요"
```

### 일반 쿼리 | General Queries

```
"안녕하세요"
"오늘 날씨는 어떻나요?" (일반 대화로 응답)
```

## 주요 특징 | Key Features

### 🧠 지능형 Agent | Intelligent Agent

- **LangGraph 워크플로우**: 구조화된 에이전트 실행 흐름
- **컨텍스트 인식**: 반도체 도메인 전문성
- **메모리 유지**: 대화 히스토리 기반 응답

### 🔍 고성능 검색 | High-Performance Search

- **FAISS VectorDB**: GPU 가속 벡터 검색
- **Sentence Transformers**: 다국어 임베딩 지원
- **유사도 기반**: 의미론적 검색

### 💬 자연어 처리 | Natural Language Processing

- **Ollama 통합**: 로컬 LLM 우선 사용
- **컨텍스트 보존**: 검색 결과 기반 답변 생성
- **메모리 관리**: 대화 맥락 유지

### 🛠 확장성 | Scalability

- **모듈화 아키텍처**: 독립적 컴포넌트 교체 가능
- **MCP 지원**: 외부 도구 통합 용이
- **API 우선**: 마이크로서비스 친화적

## 문제 해결 | Troubleshooting

### Ollama 연결 오류 | Ollama Connection Error

```bash
# Ollama 서비스 확인
ollama list
ollama serve

# 모델 다운로드 확인
ollama pull exaone-1.2b:latest
```

### 포트 충돌 | Port Conflict

```bash
# 사용 가능한 포트 자동 할당됨
# 수동 지정 시:
API_PORT=8001 python main.py
```

### 메모리 부족 | Memory Issues

```bash
# 경량 모델 사용
OLLAMA_MODEL=exaone-1.2b:latest python main.py

# 또는 시스템 RAM 증가
```

## 라이선스 | License

MIT License

## 기여 | Contributing

이 프로젝트는 개인 포트폴리오 및 학습 목적으로 개발되었습니다.

This project was developed for personal portfolio and learning purposes.

## 참고 | References

- **LangGraph**: https://langchain-ai.github.io/langgraph/
- **LangChain**: https://python.langchain.com/
- **FAISS**: https://github.com/facebookresearch/faiss
- **Ollama**: https://ollama.ai/
- **FastAPI**: https://fastapi.tiangolo.com/

---

**VirtualFab RAG System** - 반도체 제조 AI 어시스턴트 | Semiconductor Manufacturing AI Assistant
