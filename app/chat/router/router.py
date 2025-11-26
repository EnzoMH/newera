from fastapi import APIRouter, HTTPException
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from app.chat.dto.dto_rq import ChatRequest
from app.chat.dto.dto_rp import ChatResponse
from base_config import OLLAMA_BASE_URL, MODEL_NAME


router = APIRouter(
    prefix="/chat",
    tags=["chat"]
)


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """메인 채팅 엔드포인트"""
    try:
        llm = OllamaLLM(
            model=MODEL_NAME,
            base_url=OLLAMA_BASE_URL,
            temperature=request.temperature,
            top_p=request.top_p
        )
        
        prompt_template = PromptTemplate(
            input_variables=["system", "user_input"],
            template="[|system|]{system}[|endofturn|]\n[|user|]{user_input}[|endofturn|]\n[|assistant|]"
        )
        
        chain = LLMChain(llm=llm, prompt=prompt_template)
        
        response = chain.invoke({
            "system": request.system_prompt,
            "user_input": request.message
        })
        
        return ChatResponse(
            response=response["text"],
            model=MODEL_NAME
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")