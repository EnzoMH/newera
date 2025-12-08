"""
API Request Schemas
단일 책임: API 요청 데이터 구조 정의
"""
from pydantic import BaseModel, Field
from typing import Optional


class QueryRequest(BaseModel):
    """RAG 질의 요청"""
    question: str = Field(
        ...,
        description="사용자 질문",
        min_length=1,
        max_length=1000,
        examples=["VirtualFab이란 무엇인가요?"]
    )
    temperature: float = Field(
        default=0.1,
        description="응답 다양성 (0.0-1.0)",
        ge=0.0,
        le=1.0
    )
    max_tokens: Optional[int] = Field(
        default=None,
        description="최대 토큰 수",
        gt=0
    )

    class Config:
        json_schema_extra = {
            "example": {
                "question": "반도체 8대 공정에 대해 설명해주세요",
                "temperature": 0.1,
                "max_tokens": 2000
            }
        }


class AgentQueryRequest(BaseModel):
    """Agent 질의 요청"""
    question: str = Field(
        ...,
        description="사용자 질문",
        min_length=1,
        max_length=1000,
        examples=["VirtualFab에 대해 자세히 설명해주세요"]
    )
    conversation_id: Optional[str] = Field(
        default=None,
        description="대화 ID (없으면 자동 생성)",
        max_length=100
    )
    use_memory: bool = Field(
        default=True,
        description="대화 메모리 사용 여부"
    )
    temperature: float = Field(
        default=0.1,
        description="응답 다양성 (0.0-1.0)",
        ge=0.0,
        le=1.0
    )

    class Config:
        json_schema_extra = {
            "example": {
                "question": "반도체 공정에 대해 알려주세요",
                "conversation_id": "conv_001",
                "use_memory": True,
                "temperature": 0.1
            }
        }
