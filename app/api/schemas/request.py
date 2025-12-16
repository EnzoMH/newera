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

