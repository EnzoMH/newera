"""
Web Crawler MCP Tool
ArXiv 논문 크롤링 기능 제공
"""
import asyncio
import logging
from typing import Dict, Any, List
from pathlib import Path

from ..config import MCPConfig

logger = logging.getLogger(__name__)


class WebCrawlerTool:
    """ArXiv 논문 크롤러 MCP Tool"""

    def __init__(self, config: MCPConfig):
        self.config = config
        self.tool_config = config.get_tool_config("web_crawler")

    def get_tool_schema(self) -> Dict[str, Any]:
        """MCP Tool 스키마 반환"""
        return {
            "name": "web_crawler",
            "description": "ArXiv 논문 웹 크롤링 및 다운로드",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "categories": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "크롤링할 ArXiv 카테고리 목록",
                        "default": self.tool_config["arxiv_categories"]
                    },
                    "max_papers": {
                        "type": "integer",
                        "description": "최대 크롤링할 논문 수",
                        "default": self.tool_config["max_papers"]
                    },
                    "keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "검색 키워드 목록",
                        "default": ["VirtualFab", "Digital Twin", "semiconductor"]
                    }
                },
                "required": []
            }
        }

    async def execute(self, arguments: Dict[str, Any]) -> str:
        """Tool 실행"""
        try:
            categories = arguments.get("categories", self.tool_config["arxiv_categories"])
            max_papers = arguments.get("max_papers", self.tool_config["max_papers"])
            keywords = arguments.get("keywords", ["VirtualFab", "Digital Twin", "semiconductor"])

            logger.info(f"🕷️ ArXiv 크롤링 시작: {categories}, 최대 {max_papers}개")

            # 실제 크롤링 로직 (간단한 시뮬레이션)
            results = await self._crawl_arxiv(categories, max_papers, keywords)

            return f"""✅ ArXiv 크롤링 완료

📊 결과 요약:
- 카테고리: {', '.join(categories)}
- 최대 논문 수: {max_papers}
- 검색 키워드: {', '.join(keywords)}
- 발견된 논문: {len(results)}

📁 저장 위치: {self.tool_config['output_dir']}

📝 상세 결과:
{chr(10).join(f"- {paper['title']} ({paper['id']})" for paper in results[:5])}
{f'... 외 {len(results) - 5}개' if len(results) > 5 else ''}"""

        except Exception as e:
            logger.error(f"크롤링 실패: {e}")
            return f"❌ ArXiv 크롤링 실패: {str(e)}"

    async def _crawl_arxiv(self, categories: List[str], max_papers: int, keywords: List[str]) -> List[Dict[str, Any]]:
        """ArXiv 크롤링 실행"""
        from pathlib import Path
        from ...core.crawler import ArXivCrawler, KeywordFilter

        # 크롤러 초기화
        output_dir = Path(self.tool_config["output_dir"])
        crawler = ArXivCrawler(output_dir=output_dir)

        # 크롤링 실행
        results = await crawler.crawl(
            categories=categories,
            keywords=keywords,
            max_results=max_papers
        )

        # 키워드 필터링 (필요한 경우)
        if keywords:
            results = KeywordFilter.filter_by_keywords(results, keywords)

        # 결과 저장
        if results:
            crawler.save_results(results)

        return results