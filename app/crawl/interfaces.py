"""
크롤러 추상 인터페이스 (SOLID: DIP - Dependency Inversion Principle)
"""
from abc import ABC, abstractmethod
from typing import Any
from pathlib import Path


class BaseCrawler(ABC):
    """크롤러 기본 인터페이스"""
    
    @abstractmethod
    def scrape(self, max_results: int = 100) -> list[dict[str, Any]]:
        """
        데이터 수집
        
        Args:
            max_results: 최대 수집 건수
            
        Returns:
            메타데이터 딕셔너리 리스트
        """
        pass
    
    @abstractmethod
    def save_metadata(self, data: list[dict[str, Any]], output_path: Path) -> None:
        """
        메타데이터 저장
        
        Args:
            data: 수집된 데이터
            output_path: 저장 경로
        """
        pass
    
    @abstractmethod
    def is_duplicate(self, item: dict[str, Any]) -> tuple[bool, str]:
        """
        중복 검사
        
        Args:
            item: 검사할 아이템
            
        Returns:
            (중복 여부, 이유)
        """
        pass


class DocumentProcessor(ABC):
    """문서 처리 인터페이스"""
    
    @abstractmethod
    def extract_text(self, file_path: Path) -> str:
        """
        파일에서 텍스트 추출
        
        Args:
            file_path: 파일 경로
            
        Returns:
            추출된 텍스트
        """
        pass
    
    @abstractmethod
    def chunk_text(self, text: str, chunk_size: int, overlap: int) -> list[dict[str, Any]]:
        """
        텍스트 청킹
        
        Args:
            text: 원본 텍스트
            chunk_size: 청크 크기 (토큰 수)
            overlap: 오버랩 크기
            
        Returns:
            청크 리스트
        """
        pass
    
    @abstractmethod
    def filter_chunks(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        청크 필터링
        
        Args:
            chunks: 청크 리스트
            
        Returns:
            필터링된 청크 리스트
        """
        pass

