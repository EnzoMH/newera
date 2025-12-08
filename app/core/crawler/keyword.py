"""
키워드 기반 필터링 및 검색 유틸리티
"""
import logging
from typing import List, Dict, Any, Optional
import re

logger = logging.getLogger(__name__)


class KeywordFilter:
    """
    키워드 기반 필터링 및 검색
    """

    @staticmethod
    def filter_by_keywords(
        items: List[Dict[str, Any]],
        keywords: List[str],
        fields: Optional[List[str]] = None,
        match_all: bool = False
    ) -> List[Dict[str, Any]]:
        """
        키워드로 아이템 필터링

        Args:
            items: 필터링할 아이템 리스트
            keywords: 검색 키워드 리스트
            fields: 검색할 필드 목록 (None이면 title, summary, abstract 검색)
            match_all: True면 모든 키워드가 포함되어야 함, False면 하나라도 포함되면 됨

        Returns:
            필터링된 아이템 리스트
        """
        if not keywords:
            return items

        if fields is None:
            fields = ["title", "summary", "abstract", "content"]

        def matches(item: Dict[str, Any]) -> bool:
            # 검색할 텍스트 수집
            search_texts = []
            for field in fields:
                if field in item and item[field]:
                    search_texts.append(str(item[field]).lower())

            if not search_texts:
                return False

            combined_text = " ".join(search_texts)

            # 키워드 매칭
            keyword_lower = [kw.lower() for kw in keywords]
            if match_all:
                return all(kw in combined_text for kw in keyword_lower)
            else:
                return any(kw in combined_text for kw in keyword_lower)

        filtered = [item for item in items if matches(item)]
        logger.info(f"🔍 키워드 필터링: {len(items)}개 → {len(filtered)}개")
        return filtered

    @staticmethod
    def extract_keywords(text: str, min_length: int = 3) -> List[str]:
        """
        텍스트에서 키워드 추출 (간단한 버전)

        Args:
            text: 분석할 텍스트
            min_length: 최소 키워드 길이

        Returns:
            추출된 키워드 리스트
        """
        # 간단한 키워드 추출 (실제로는 더 정교한 NLP 기법 사용 가능)
        words = re.findall(r'\b\w+\b', text.lower())
        # 길이 필터링 및 중복 제거
        keywords = list(set([w for w in words if len(w) >= min_length]))
        return keywords

    @staticmethod
    def score_by_keywords(
        item: Dict[str, Any],
        keywords: List[str],
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        키워드 매칭 점수 계산

        Args:
            item: 점수를 계산할 아이템
            keywords: 검색 키워드
            weights: 필드별 가중치 (기본값: title=2.0, summary=1.5, abstract=1.0)

        Returns:
            매칭 점수 (0.0 ~ 1.0)
        """
        if weights is None:
            weights = {"title": 2.0, "summary": 1.5, "abstract": 1.0, "content": 1.0}

        total_score = 0.0
        total_weight = sum(weights.values())

        keyword_lower = [kw.lower() for kw in keywords]

        for field, weight in weights.items():
            if field in item and item[field]:
                text = str(item[field]).lower()
                matches = sum(1 for kw in keyword_lower if kw in text)
                field_score = matches / len(keyword_lower) if keyword_lower else 0.0
                total_score += field_score * weight

        return total_score / total_weight if total_weight > 0 else 0.0
