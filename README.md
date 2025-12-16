# VirtualFab RAG System

> 반도체 제조 도메인 특화 AI 플랫폼  
> LangChain + LangGraph + RAG 기반

## 🚀 빠른 시작

### 필수 요구사항
- Python 3.12+
- Docker & Docker Compose
- CUDA (GPU 사용 시)
- Node.js 18+ (Frontend)

### 로컬 개발 환경

```bash
# 1. 저장소 클론
git clone <repository-url>
cd newera

# 2. 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 의존성 설치
pip install -r requirements.txt

# 4. 환경 변수 설정
cp .env.example .env
# .env 파일 수정

# 5. Redis & MongoDB 실행
docker-compose -f docker-compose.dev.yml up -d

# 6. 서버 실행
python main.py
```

서버 접속: http://localhost:8000  
API 문서: http://localhost:8000/docs

### Docker로 전체 스택 실행

```bash
# 전체 서비스 실행
docker-compose up

# 백그라운드 실행
docker-compose up -d

# 로그 확인
docker-compose logs -f backend

# 종료
docker-compose down
```

## 📁 프로젝트 구조

```
newera/
├── app/                    # 애플리케이션 코드
│   ├── agents/            # LangGraph Agent
│   ├── api/               # FastAPI REST API
│   ├── core/              # 비즈니스 로직
│   ├── memory/            # 대화 메모리
│   ├── mcp/               # MCP 서버
│   ├── tools/             # LangChain Tools
│   └── web/               # Gradio UI
├── models/                # LLM 모델 파일
├── scripts/               # 유틸리티 스크립트
├── main.py                # 메인 엔트리포인트
├── config.py              # 전역 설정
├── requirements.txt       # Python 의존성
├── Dockerfile             # Docker 이미지
├── docker-compose.yml     # Docker Compose 설정
└── ARCHITECTURE.md        # 아키텍처 문서 📖
```

## 📚 주요 기능

### 현재 구현
- ✅ FastAPI REST API
- ✅ LangGraph Agent 워크플로우
- ✅ FAISS Vector Store (GPU 지원)
- ✅ MCP 서버 (4가지 Tools)
- ✅ Conversation Memory
- ✅ Gradio Web UI

### 개발 예정
- ⏳ Next.js Frontend
- ⏳ Streaming API (SSE/WebSocket)
- ⏳ Document Upload
- ⏳ Analytics Dashboard
- ⏳ User Authentication

## 🔌 포트 맵

| 서비스 | 포트 | 설명 |
|--------|------|------|
| Frontend | 3000 | Next.js |
| Backend API | 8000 | FastAPI |
| MCP Server | 8083 | MCP Tools |
| Redis | 6379 | 캐시 |
| MongoDB | 27017 | 데이터베이스 |
| Gradio UI | 7860 | Web UI (개발) |

## 📖 문서

자세한 아키텍처 설계는 [ARCHITECTURE.md](./ARCHITECTURE.md)를 참조하세요.

### 주요 내용
- 시스템 개요
- 마이크로서비스 아키텍처
- 포트 할당 계획
- Docker Compose 구성
- API 엔드포인트 목록
- 데이터 흐름
- 개발 로드맵

## 🛠️ 개발 가이드

### API 테스트

```bash
# 헬스체크
curl http://localhost:8000/health

# RAG 질의
curl -X POST http://localhost:8000/api/rag/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "반도체 8대 공정에 대해 알려주세요",
    "temperature": 0.1
  }'
```

### 코드 스타일

```bash
# Black 포맷팅
black app/

# Flake8 린팅
flake8 app/

# isort import 정렬
isort app/
```

## 🤝 기여 가이드

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 라이선스

This project is licensed under the MIT License.

## 👥 팀

- Architecture Team
- Backend Team
- Frontend Team
- AI/ML Team

## 📞 문의

- 이슈: GitHub Issues
- 이메일: team@virtualfab.com

---

**Version**: 2.1.0  
**Last Updated**: 2025-12-16
