from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_ollama import OllamaLLM
from dotenv import load_dotenv

from base_config import OLLAMA_BASE_URL, MODEL_NAME
from app.chat.router.router import router as chat_router

load_dotenv()

app = FastAPI(
    title="Project Agentic System",
    description="Langchain + Ollama + FastAPI Application",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)



@app.get("/")
async def root() -> dict[str, str]:
    """헬스 체크 엔드포인트"""
    return {
        "status": "healthy",
        "message": "EXAONE AI Application is running"
    }


@app.get("/health")
async def health_check() -> dict[str, str]:
    """상세 헬스 체크"""
    try:
        llm = OllamaLLM(
            model=MODEL_NAME,
            base_url=OLLAMA_BASE_URL,
            temperature=0.1
        )
        llm.invoke("test")
        return {
            "status": "healthy",
            "ollama": "connected",
            "model": MODEL_NAME
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Ollama connection failed: {str(e)}")





if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )