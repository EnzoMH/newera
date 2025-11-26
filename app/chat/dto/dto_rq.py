from pydantic import BaseModel
from typing import Optional


class ChatRequest(BaseModel):
    message: str
    system_prompt: Optional[str] = "당신은 도움이 되는 AI 어시스턴트입니다."
    temperature: Optional[float] = 0.6
    top_p: Optional[float] = 0.95






