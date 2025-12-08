"""
Base Crawler Abstract Class
모든 크롤러의 기본 인터페이스 정의
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class BaseCrawler(ABC):
    """
    크롤러 기본 추상 클래스
    모든 크롤러는 이 클래스를 상속받아 구현
    """

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Args:
            output_dir: 크롤링 결과 저장 디렉토리
        """
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    async def crawl(self, **kwargs) -> List[Dict[str, Any]]:
        """
        크롤링 실행 (추상 메서드)

        Returns:
            크롤링된 데이터 리스트
        """
        pass

    @abstractmethod
    def get_source_name(self) -> str:
        """
        크롤러 소스 이름 반환

        Returns:
            소스 이름 (예: "arxiv", "pubmed" 등)
        """
        pass

    def save_results(self, results: List[Dict[str, Any]], filename: Optional[str] = None) -> Path:
        """
        크롤링 결과를 파일로 저장

        Args:
            results: 저장할 데이터 리스트
            filename: 저장할 파일명 (없으면 자동 생성)

        Returns:
            저장된 파일 경로
        """
        if not self.output_dir:
            raise ValueError("output_dir가 설정되지 않았습니다.")

        import json
        from datetime import datetime

        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.get_source_name()}_{timestamp}.json"

        file_path = self.output_dir / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info(f"💾 크롤링 결과 저장: {file_path}")
        return file_path

