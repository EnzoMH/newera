"""
Memory Package
LangChain Memory 시스템
"""
from .conversation_simple import SimpleConversationMemory, get_conversation_memory

# 하위 호환성을 위한 별칭
ConversationBufferMemory = SimpleConversationMemory

__all__ = ["SimpleConversationMemory", "ConversationBufferMemory", "get_conversation_memory"]
