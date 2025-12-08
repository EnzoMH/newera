"""
API Response Schemas
단일 책임: API 응답 데이터 구조 정의
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime


class SourceDocument(BaseModel):
    """참고 문서 정보"""
    filename: Optional[str] = Field(None, description="파일명")
    source: Optional[str] = Field(None, description="출처 정보")
    content: str = Field(..., description="문서 내용 미리보기", max_length=500)
    score: Optional[float] = Field(None, description="유사도 점수", ge=0.0)
    domain: Optional[str] = Field(None, description="도메인 분류")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="추가 메타데이터")


class QueryResponse(BaseModel):
    """RAG 질의 응답"""
    answer: str = Field(..., description="AI 답변")
    sources: List[SourceDocument] = Field(
        default_factory=list, 
        description="참고 문서들"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, 
        description="메타정보"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now, 
        description="응답 시간"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "반도체 8대 공정은...",
                "sources": [],
                "metadata": {
                    "llm_provider": "ollama",
                    "model": "qwen2.5-3b-instruct",
                    "rag_enabled": False
                },
                "timestamp": "2025-12-08T10:30:00"
            }
        }


class HealthResponse(BaseModel):
    """헬스체크 응답"""
    status: str = Field(..., description="시스템 상태 (healthy/unhealthy)")
    initialized: bool = Field(..., description="초기화 완료 여부")
    llm_available: bool = Field(..., description="LLM 사용 가능 여부")
    vector_store_available: bool = Field(..., description="벡터 저장소 사용 가능 여부")
    domain: str = Field(..., description="전문 도메인")
    version: str = Field(..., description="시스템 버전")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="응답 시간"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "initialized": True,
                "llm_available": True,
                "vector_store_available": False,
                "domain": "VirtualFab/Digital Twin",
                "version": "2.1.0",
                "timestamp": "2025-12-08T10:30:00"
            }
        }


class SystemStatusResponse(BaseModel):
    """상세 시스템 상태 응답"""
    initialized: bool
    llm_available: bool
    vector_store_available: bool
    crawler_available: bool
    retriever_available: bool
    domain: str
    components: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_schema_extra = {
            "example": {
                "initialized": True,
                "llm_available": True,
                "vector_store_available": False,
                "crawler_available": False,
                "retriever_available": False,
                "domain": "VirtualFab/Digital Twin",
                "components": {
                    "llm": {"provider": "ollama", "model": "qwen2.5-3b-instruct"},
                    "vecdb": {"status": "not_configured"},
                    "memory": {"status": "not_configured"}
                },
                "timestamp": "2025-12-08T10:30:00"
            }
        }


class AgentQueryResponse(BaseModel):
    """Agent 질의 응답"""
    answer: str = Field(..., description="AI 답변")
    sources: List[SourceDocument] = Field(
        default_factory=list,
        description="참고 문서들"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="메타정보 (agent 상태, 진행률 등)"
    )
    conversation_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="대화 히스토리"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="응답 시간"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "answer": "반도체 제조 공정은 8개의 주요 단계로 구성됩니다...",
                "sources": [],
                "metadata": {
                    "agent": "RAGAgent",
                    "status": "completed",
                    "progress": 100,
                    "conversation_id": "conv_001"
                },
                "conversation_history": [
                    {"human": "반도체 공정에 대해 알려주세요", "ai": "..."}
                ],
                "timestamp": "2025-12-08T10:30:00"
            }
        }


class ErrorResponse(BaseModel):
    """에러 응답"""
    error: str = Field(..., description="에러 메시지")
    code: str = Field(..., description="에러 코드")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="에러 발생 시간"
    )
    details: Optional[Dict[str, Any]] = Field(
        None,
        description="추가 에러 정보"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "error": "시스템이 초기화되지 않았습니다",
                "code": "NOT_INITIALIZED",
                "timestamp": "2025-12-08T10:30:00",
                "details": {"component": "rag_system"}
            }
        }

