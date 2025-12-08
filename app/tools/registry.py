"""
Tool Registry
LangChain Tools 등록 및 관리
"""
import logging
from typing import Dict, Any, List, Optional, Type
from functools import lru_cache

from langchain.tools import BaseTool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Tool 등록 및 관리 시스템
    동적 Tool 로드 및 캐싱
    """

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._tool_classes: Dict[str, Type[BaseTool]] = {}
        self.logger = logger

        logger.info("🔧 Tool Registry 초기화")

    def register_tool(self, name: str, tool_class: Type[BaseTool], **kwargs) -> None:
        """
        Tool 클래스 등록

        Args:
            name: Tool 이름
            tool_class: Tool 클래스
            **kwargs: Tool 초기화 파라미터
        """
        try:
            self._tool_classes[name] = tool_class

            # 즉시 인스턴스화 (필요시)
            if kwargs.get('instantiate', True):
                tool_instance = tool_class(**kwargs)
                self._tools[name] = tool_instance

            logger.info(f"✅ Tool 등록: {name}")

        except Exception as e:
            logger.error(f"❌ Tool 등록 실패 ({name}): {e}")

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """
        Tool 인스턴스 가져오기

        Args:
            name: Tool 이름

        Returns:
            Tool 인스턴스 또는 None
        """
        # 이미 인스턴스화된 경우 반환
        if name in self._tools:
            return self._tools[name]

        # 클래스에서 인스턴스화
        if name in self._tool_classes:
            try:
                tool_instance = self._tool_classes[name]()
                self._tools[name] = tool_instance
                return tool_instance
            except Exception as e:
                logger.error(f"Tool 인스턴스화 실패 ({name}): {e}")

        return None

    def get_all_tools(self) -> List[BaseTool]:
        """
        모든 Tool 인스턴스 목록

        Returns:
            Tool 인스턴스 리스트
        """
        tools = []
        for name in self._tool_classes.keys():
            tool = self.get_tool(name)
            if tool:
                tools.append(tool)
        return tools

    def get_tool_names(self) -> List[str]:
        """
        등록된 Tool 이름 목록

        Returns:
            Tool 이름 리스트
        """
        return list(self._tool_classes.keys())

    def has_tool(self, name: str) -> bool:
        """
        Tool 존재 여부 확인

        Args:
            name: Tool 이름

        Returns:
            존재 여부
        """
        return name in self._tool_classes

    def remove_tool(self, name: str) -> bool:
        """
        Tool 제거

        Args:
            name: Tool 이름

        Returns:
            제거 성공 여부
        """
        try:
            if name in self._tools:
                del self._tools[name]
            if name in self._tool_classes:
                del self._tool_classes[name]

            logger.info(f"🗑️ Tool 제거: {name}")
            return True

        except Exception as e:
            logger.error(f"Tool 제거 실패 ({name}): {e}")
            return False

    def clear_all_tools(self) -> None:
        """
        모든 Tool 클리어
        """
        self._tools.clear()
        self._tool_classes.clear()
        logger.info("🧹 모든 Tool 클리어됨")

    def get_registry_stats(self) -> Dict[str, Any]:
        """
        Registry 통계 정보

        Returns:
            통계 정보
        """
        return {
            "total_tools": len(self._tool_classes),
            "instantiated_tools": len(self._tools),
            "tool_names": self.get_tool_names()
        }


# 싱글톤 패턴
_registry_instance = None


def get_tool_registry() -> ToolRegistry:
    """
    Tool Registry 싱글톤 인스턴스

    Returns:
        ToolRegistry 인스턴스
    """
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = ToolRegistry()
    return _registry_instance


def register_default_tools():
    """
    기본 Tool들 등록
    애플리케이션 시작 시 호출
    """
    registry = get_tool_registry()

    # TODO: 실제 Tool들 등록
    # registry.register_tool("vector_search", VectorSearchTool)
    # registry.register_tool("web_crawler", WebCrawlerTool)
    # registry.register_tool("pdf_parser", PDFParserTool)

    logger.info("📋 기본 Tool들 등록 준비됨 (아직 구현되지 않음)")
