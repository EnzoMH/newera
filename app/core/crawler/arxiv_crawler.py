"""
ArXiv 논문 크롤러
ArXiv API를 사용한 논문 검색 및 다운로드
"""
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import arxiv
import asyncio

from .base import BaseCrawler

logger = logging.getLogger(__name__)


class ArXivCrawler(BaseCrawler):
    """
    ArXiv 논문 크롤러
    - ArXiv API를 통한 논문 검색
    - PDF 다운로드
    - 메타데이터 추출
    """

    def __init__(self, output_dir: Optional[Path] = None):
        super().__init__(output_dir)
        self.client = arxiv.Client()

    def get_source_name(self) -> str:
        return "arxiv"

    async def crawl(
        self,
        categories: Optional[List[str]] = None,
        keywords: Optional[List[str]] = None,
        max_results: int = 100,
        sort_by: str = "submittedDate",
        sort_order: str = "descending"
    ) -> List[Dict[str, Any]]:
        """
        ArXiv 논문 크롤링

        Args:
            categories: ArXiv 카테고리 목록 (예: ["cs.AI", "cs.LG"])
            keywords: 검색 키워드 목록
            max_results: 최대 결과 수
            sort_by: 정렬 기준 ("relevance", "lastUpdatedDate", "submittedDate")
            sort_order: 정렬 순서 ("ascending", "descending")

        Returns:
            논문 메타데이터 리스트
        """
        try:
            logger.info(f"🔍 ArXiv 크롤링 시작: categories={categories}, keywords={keywords}, max={max_results}")

            # 검색 쿼리 구성
            query = self._build_query(categories, keywords)

            # ArXiv 검색 실행 (동기 함수를 비동기로 래핑)
            loop = asyncio.get_event_loop()
            search = await loop.run_in_executor(
                None,
                lambda: arxiv.Search(
                    query=query,
                    max_results=max_results,
                    sort_by=getattr(arxiv.SortCriterion, sort_by),
                    sort_order=getattr(arxiv.SortOrder, sort_order)
                )
            )

            # 결과 처리
            results = []
            for paper in await loop.run_in_executor(None, lambda: list(self.client.results(search))):
                paper_data = {
                    "id": paper.entry_id.split('/')[-1],
                    "title": paper.title,
                    "authors": [author.name for author in paper.authors],
                    "summary": paper.summary,
                    "published": paper.published.isoformat() if paper.published else None,
                    "updated": paper.updated.isoformat() if paper.updated else None,
                    "categories": paper.categories,
                    "pdf_url": paper.pdf_url,
                    "primary_category": paper.primary_category,
                    "doi": paper.doi if hasattr(paper, 'doi') else None,
                }

                # PDF 다운로드 (선택사항)
                if self.output_dir:
                    await self._download_pdf(paper, paper_data["id"])

                results.append(paper_data)

            logger.info(f"✅ ArXiv 크롤링 완료: {len(results)}개 논문 발견")
            return results

        except Exception as e:
            logger.error(f"❌ ArXiv 크롤링 실패: {e}")
            raise

    def _build_query(self, categories: Optional[List[str]], keywords: Optional[List[str]]) -> str:
        """
        ArXiv 검색 쿼리 구성

        Args:
            categories: 카테고리 목록
            keywords: 키워드 목록

        Returns:
            검색 쿼리 문자열
        """
        query_parts = []

        # 카테고리 필터
        if categories:
            cat_query = " OR ".join([f"cat:{cat}" for cat in categories])
            query_parts.append(f"({cat_query})")

        # 키워드 검색
        if keywords:
            keyword_query = " OR ".join([f'"{kw}"' for kw in keywords])
            query_parts.append(f"({keyword_query})")

        if not query_parts:
            # 기본값: VirtualFab 관련
            query_parts.append('("VirtualFab" OR "Digital Twin" OR "semiconductor")')

        return " AND ".join(query_parts)

    async def _download_pdf(self, paper: arxiv.Result, paper_id: str) -> Optional[Path]:
        """
        논문 PDF 다운로드

        Args:
            paper: ArXiv 논문 객체
            paper_id: 논문 ID

        Returns:
            다운로드된 파일 경로
        """
        if not self.output_dir:
            return None

        try:
            pdf_dir = self.output_dir / "pdfs"
            pdf_dir.mkdir(parents=True, exist_ok=True)

            pdf_path = pdf_dir / f"{paper_id}.pdf"

            # 이미 다운로드된 경우 스킵
            if pdf_path.exists():
                logger.debug(f"📄 PDF 이미 존재: {pdf_path}")
                return pdf_path

            # PDF 다운로드
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, paper.download_pdf, str(pdf_dir))

            # 파일명 변경 (ArXiv는 기본적으로 "arxiv_id.pdf"로 저장)
            downloaded_file = pdf_dir / f"{paper_id.replace('.', '_')}.pdf"
            if downloaded_file.exists():
                downloaded_file.rename(pdf_path)

            logger.info(f"📥 PDF 다운로드 완료: {pdf_path}")
            return pdf_path

        except Exception as e:
            logger.warning(f"⚠️ PDF 다운로드 실패 ({paper_id}): {e}")
            return None

