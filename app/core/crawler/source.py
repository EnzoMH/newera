"""
다양한 소스 크롤러 팩토리 및 관리
"""
import logging
from typing import Dict, Type, Optional, Any, List
from pathlib import Path

from .base import BaseCrawler
from .arxiv_crawler import ArXivCrawler

logger = logging.getLogger(__name__)


class CrawlerFactory:
    """
    크롤러 팩토리 클래스
    소스 타입에 따라 적절한 크롤러 인스턴스 생성
    """

    _crawlers: Dict[str, Type[BaseCrawler]] = {
        "arxiv": ArXivCrawler,
        # 추후 추가 가능:
        # "pubmed": PubMedCrawler,
        # "ieee": IEEECrawler,
        # "acm": ACMCrawler,
    }

    @classmethod
    def create_crawler(
        cls,
        source_type: str,
        output_dir: Optional[Path] = None
    ) -> BaseCrawler:
        """
        크롤러 인스턴스 생성

        Args:
            source_type: 소스 타입 ("arxiv", "pubmed" 등)
            output_dir: 출력 디렉토리

        Returns:
            크롤러 인스턴스

        Raises:
            ValueError: 지원하지 않는 소스 타입인 경우
        """
        crawler_class = cls._crawlers.get(source_type.lower())
        if not crawler_class:
            available = ", ".join(cls._crawlers.keys())
            raise ValueError(
                f"지원하지 않는 소스 타입: {source_type}. "
                f"사용 가능한 타입: {available}"
            )

        logger.info(f"🏭 크롤러 생성: {source_type}")
        return crawler_class(output_dir=output_dir)

    @classmethod
    def register_crawler(cls, source_type: str, crawler_class: Type[BaseCrawler]):
        """
        새로운 크롤러 등록

        Args:
            source_type: 소스 타입 이름
            crawler_class: 크롤러 클래스
        """
        cls._crawlers[source_type.lower()] = crawler_class
        logger.info(f"📝 크롤러 등록: {source_type}")

    @classmethod
    def get_available_sources(cls) -> list[str]:
        """
        사용 가능한 소스 타입 목록 반환

        Returns:
            소스 타입 리스트
        """
        return list(cls._crawlers.keys())


class MultiSourceCrawler:
    """
    여러 소스에서 동시에 크롤링하는 크롤러
    """

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = Path(output_dir) if output_dir else None
        self.factory = CrawlerFactory()

    async def crawl_multiple(
        self,
        sources: List[str],
        **kwargs
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        여러 소스에서 동시에 크롤링

        Args:
            sources: 크롤링할 소스 타입 리스트
            **kwargs: 각 크롤러에 전달할 파라미터

        Returns:
            소스별 결과 딕셔너리
        """
        import asyncio

        results = {}

        # 각 소스별로 크롤링 실행
        tasks = []
        for source in sources:
            try:
                crawler = self.factory.create_crawler(source, self.output_dir)
                task = crawler.crawl(**kwargs)
                tasks.append((source, task))
            except Exception as e:
                logger.error(f"❌ {source} 크롤러 생성 실패: {e}")
                results[source] = []

        # 병렬 실행
        for source, task in tasks:
            try:
                source_results = await task
                results[source] = source_results
            except Exception as e:
                logger.error(f"❌ {source} 크롤링 실패: {e}")
                results[source] = []

        return results
