"""
Tech Blog 크롤러 (Lam Research, Applied Materials, McKinsey, AWS 등)
"""
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import feedparser
import requests
from bs4 import BeautifulSoup

from app.crawl.interfaces import BaseCrawler


logger = logging.getLogger(__name__)


class TechBlogCrawler(BaseCrawler):
    """
    반도체 제조 관련 Tech Blog 및 케이스 스터디 크롤러
    """
    
    def __init__(self, output_dir: str | Path = "data/crawled/TechBlogs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.existing_articles = self._load_existing_articles()
        
        self.rss_feeds = {
            'semiconductor_digest': 'https://www.semiconductor-digest.com/feed/',
            'semi_engineering': 'https://semiengineering.com/feed/',
        }
        
        self.target_urls = {
            'lam_research': {
                'base': 'https://www.lamresearch.com',
                'paths': ['/blog/', '/semiverse/'],
            },
            'applied_materials': {
                'base': 'https://www.appliedmaterials.com',
                'paths': ['/blog/'],
            },
        }
        
        self.keywords = [
            'virtual fabrication', 'Semiverse', 'digital twin', 'process control',
            'etch', 'deposition', 'EUV', 'yield', 'metrology', 'defect',
            'smart fab', 'virtual metrology', 'fab scheduling', 'AI',
            'machine learning', 'deep learning', 'reinforcement learning',
            'wafer', 'lithography', 'CMP', 'CVD', 'ALD',
        ]
        
        logger.info(f"Tech Blog 크롤러 초기화 완료: {self.output_dir}")
    
    def _load_existing_articles(self) -> dict[str, set]:
        """기존 아티클 로드"""
        existing = {
            'titles': set(),
            'links': set(),
        }
        
        metadata_file = self.output_dir / "metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for article in data.get('articles', []):
                        if title := article.get('title'):
                            existing['titles'].add(title.strip().lower())
                        if link := article.get('link'):
                            existing['links'].add(link)
                            
                logger.info(f"기존 아티클 로드: {len(existing['titles'])}개")
            except Exception as e:
                logger.warning(f"메타데이터 로드 실패: {e}")
        
        return existing
    
    def is_duplicate(self, item: dict[str, Any]) -> tuple[bool, str]:
        """중복 검사"""
        title = item.get('title', '').strip().lower()
        link = item.get('link', '')
        
        if title and title in self.existing_articles['titles']:
            return True, f"중복 제목: {title[:50]}..."
        if link and link in self.existing_articles['links']:
            return True, f"중복 링크: {link}"
        
        return False, ""
    
    def _scrape_rss_feed(self, source_name: str, feed_url: str, max_articles: int = 50) -> list[dict[str, Any]]:
        """RSS 피드 크롤링"""
        logger.info(f"[{source_name}] RSS 피드 수집 시작...")
        
        articles: list[dict[str, Any]] = []
        
        try:
            feed = feedparser.parse(feed_url)
            
            for entry in feed.entries[:max_articles]:
                title = entry.get('title', '').lower()
                summary = entry.get('summary', '').lower()
                
                if any(kw.lower() in title or kw.lower() in summary for kw in self.keywords):
                    article = {
                        'source': source_name,
                        'title': entry.get('title', ''),
                        'link': entry.get('link', ''),
                        'published': entry.get('published', ''),
                        'summary': entry.get('summary', ''),
                        'content': entry.get('content', [{}])[0].get('value', '') if entry.get('content') else '',
                        'collected_at': datetime.now().isoformat(),
                        'document_type': 'blog',
                    }
                    
                    is_dup, reason = self.is_duplicate(article)
                    if not is_dup:
                        articles.append(article)
                        self.existing_articles['titles'].add(article['title'].strip().lower())
                        self.existing_articles['links'].add(article['link'])
                        logger.info(f"✓ 수집: {article['title'][:60]}...")
            
            logger.info(f"[{source_name}] {len(articles)}개 수집 완료")
            
        except Exception as e:
            logger.error(f"[{source_name}] RSS 피드 수집 실패: {e}")
        
        return articles
    
    def _scrape_website(self, source_name: str, config: dict[str, Any]) -> list[dict[str, Any]]:
        """웹사이트 크롤링"""
        logger.info(f"[{source_name}] 웹사이트 크롤링 시작...")
        logger.info("⚠️  이 기능은 JavaScript 렌더링이 필요할 수 있습니다")
        logger.info("⚠️  수동 수집을 권장합니다")
        
        return []
    
    def scrape(self, max_results: int = 50) -> list[dict[str, Any]]:
        """모든 소스에서 아티클 수집"""
        logger.info("Tech Blog 아티클 수집 시작")
        
        all_articles: list[dict[str, Any]] = []
        
        for source_name, feed_url in self.rss_feeds.items():
            articles = self._scrape_rss_feed(source_name, feed_url, max_results)
            all_articles.extend(articles)
            time.sleep(2)
        
        for source_name, config in self.target_urls.items():
            articles = self._scrape_website(source_name, config)
            all_articles.extend(articles)
            time.sleep(2)
        
        logger.info(f"총 {len(all_articles)}개 아티클 수집 완료")
        return all_articles
    
    def save_metadata(self, data: list[dict[str, Any]], output_path: Path | None = None) -> None:
        """메타데이터 저장"""
        if output_path is None:
            output_path = self.output_dir / "metadata.json"
        
        metadata = {
            "collection_info": {
                "created_date": datetime.now().isoformat(),
                "total_articles": len(data),
                "sources": list(set(article.get('source', '') for article in data)),
            },
            "articles": data
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        txt_file = self.output_dir.parent / f"techblogs_{datetime.now().strftime('%Y%m%d')}.txt"
        with open(txt_file, 'w', encoding='utf-8') as f:
            for article in data:
                f.write(f"{'='*80}\n")
                f.write(f"제목: {article['title']}\n")
                f.write(f"출처: {article['source']}\n")
                f.write(f"링크: {article['link']}\n")
                f.write(f"날짜: {article.get('published', 'N/A')}\n")
                f.write(f"\n내용:\n{article.get('summary', '')}\n")
                f.write(f"{'='*80}\n\n")
        
        logger.info(f"메타데이터 저장: {output_path}")
        logger.info(f"텍스트 파일 저장: {txt_file}")




