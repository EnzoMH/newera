"""
API Routers Package
"""
from .rag import router as rag_router
from .health import router as health_router
from .system import router as system_router
from .agent import router as agent_router

__all__ = [
    "rag_router",
    "health_router",
    "system_router",
    "agent_router"
]

# 하위 호환성을 위한 router 별칭
router = rag_router

