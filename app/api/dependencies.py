"""FastAPI Dependencies"""
from typing import Optional
from fastapi import HTTPException
from ..core.rag import RAGSystem

_rag_system: Optional[RAGSystem] = None


def set_rag_system(rag_system: RAGSystem) -> None:
    global _rag_system
    _rag_system = rag_system


def get_rag_system() -> RAGSystem:
    if _rag_system is None:
        raise HTTPException(status_code=503, detail="RAG not initialized")
    return _rag_system


def check_rag_initialized() -> bool:
    return _rag_system is not None


def clear_rag_system() -> None:
    global _rag_system
    _rag_system = None
