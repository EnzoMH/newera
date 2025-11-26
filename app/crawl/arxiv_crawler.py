"""
반도체 제조 관련 ArXiv 논문 크롤러
"""
import os
import json
import re
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import arxiv

from app.crawl.interfaces import BaseCrawler


logger = logging.getLogger(__name__)


class SemiconductorArxivCrawler(BaseCrawler):
    """
    반도체 제조 관련 ArXiv 논문 수집기
    FEBRAG_archtecturing.md 기반 키워드 검색
    """
    
    def __init__(self, output_dir: str | Path = "data/crawled/ArXiv"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.existing_papers = self._load_existing_papers()
        
        self.domain_search_keywords = {
            'VirtualFab': [
                "semiconductor virtual fabrication",
                "virtual fab semiconductor",
                "semiconductor digital twin manufacturing",
                "physics-informed neural networks semiconductor",
            ],
            'FabScheduling': [
                "semiconductor fab scheduling deep reinforcement learning",
                "wafer fab scheduling reinforcement learning",
                "semiconductor manufacturing capacity planning optimization",
                "queue time wafer fab optimization",
                "dispatching rule semiconductor machine learning",
            ],
            'VirtualMetrology': [
                "virtual metrology semiconductor",
                "virtual metrology CVD semiconductor",
                "virtual metrology CMP semiconductor",
                "run-to-run control virtual metrology",
                "process control semiconductor machine learning",
            ],
            'DefectInspection': [
                "semiconductor defect detection deep learning",
                "wafer defect classification cnn",
                "photolithography defect detection",
                "semiconductor inspection computer vision",
                "SEM images defect classification",
            ],
            'DomainLLM': [
                "semiconductor large language model",
                "domain-specific LLM semiconductor",
                "AI semiconductor manufacturing optimization",
                "machine learning semiconductor manufacturing review",
            ],
        }
        
        self.domain_keywords = {
            'VirtualFab': ['virtual fab', 'digital twin', 'physics-informed', 'PINN', 'simulation', 'process modeling'],
            'FabScheduling': ['scheduling', 'dispatching', 'reinforcement learning', 'capacity planning', 'queue time', 'throughput'],
            'VirtualMetrology': ['virtual metrology', 'VM', 'CVD', 'CMP', 'etch', 'deposition', 'run-to-run'],
            'DefectInspection': ['defect', 'inspection', 'SEM', 'computer vision', 'classification', 'detection'],
            'DomainLLM': ['LLM', 'large language model', 'NLP', 'text mining', 'knowledge graph'],
        }
        
        logger.info(f"ArXiv 크롤러 초기화 완료: {self.output_dir}")
    
    def _load_existing_papers(self) -> dict[str, set]:
        """기존 논문 정보 로드"""
        existing = {
            'titles': set(),
            'urls': set(),
            'ids': set(),
        }
        
        metadata_file = self.output_dir / "metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for paper in data.get('papers', []):
                        if title := paper.get('title'):
                            existing['titles'].add(title.strip().lower())
                        if url := paper.get('pdf_url'):
                            existing['urls'].add(url)
                        if paper_id := paper.get('id'):
                            existing['ids'].add(str(paper_id))
                            
                logger.info(f"기존 논문 로드: {len(existing['titles'])}편")
            except Exception as e:
                logger.warning(f"메타데이터 로드 실패: {e}")
        
        return existing
    
    def is_duplicate(self, item: dict[str, Any]) -> tuple[bool, str]:
        """중복 검사"""
        title = item.get('title', '').strip().lower()
        url = item.get('pdf_url', '')
        paper_id = item.get('id', '')
        
        if title and title in self.existing_papers['titles']:
            return True, f"중복 제목: {title[:50]}..."
        if url and url in self.existing_papers['urls']:
            return True, f"중복 URL: {url}"
        if paper_id and str(paper_id) in self.existing_papers['ids']:
            return True, f"중복 ID: {paper_id}"
        
        return False, ""
    
    def _classify_domain(self, title: str, abstract: str) -> str:
        """논문 도메인 분류"""
        text = (title + " " + abstract).lower()
        
        domain_scores: dict[str, int] = {}
        for domain, keywords in self.domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword.lower() in text)
            if score > 0:
                domain_scores[domain] = score
        
        if domain_scores:
            return max(domain_scores.items(), key=lambda x: x[1])[0]
        return 'General'
    
    def scrape(self, max_results: int = 100) -> list[dict[str, Any]]:
        """ArXiv에서 논문 수집"""
        logger.info(f"ArXiv 논문 수집 시작 (도메인별 최대 {max_results}편)")
        
        papers: list[dict[str, Any]] = []
        
        for domain, keywords in self.domain_search_keywords.items():
            logger.info(f"\n[{domain}] 도메인 검색 시작...")
            logger.info(f"  키워드: {', '.join(keywords)}")
            
            search_query = " OR ".join([f'"{kw}"' for kw in keywords])
            
            try:
                search = arxiv.Search(
                    query=search_query,
                    max_results=max_results,
                    sort_by=arxiv.SortCriterion.SubmittedDate,
                    sort_order=arxiv.SortOrder.Descending
                )
                
                domain_count = 0
                for result in search.results():
                    paper_data = {
                        "id": result.entry_id,
                        "title": result.title,
                        "authors": [author.name for author in result.authors],
                        "abstract": result.summary,
                        "published": result.published.strftime("%Y-%m-%d"),
                        "categories": result.categories,
                        "pdf_url": result.pdf_url,
                        "source": "ArXiv",
                        "domain": domain,
                    }
                    
                    is_dup, reason = self.is_duplicate(paper_data)
                    if is_dup:
                        logger.debug(f"중복 스킵: {reason}")
                        continue
                    
                    papers.append(paper_data)
                    domain_count += 1
                    
                    self.existing_papers['titles'].add(paper_data['title'].strip().lower())
                    self.existing_papers['urls'].add(paper_data['pdf_url'])
                    self.existing_papers['ids'].add(str(paper_data['id']))
                    
                    try:
                        clean_title = re.sub(r'[<>:"/\\|?*]', '_', result.title[:40])
                        filename = f"{len(papers):03d}_{domain}_{clean_title}.pdf"
                        file_path = self.output_dir / filename
                        
                        if not file_path.exists():
                            result.download_pdf(dirpath=str(self.output_dir), filename=filename)
                            logger.info(f"[{domain}] 다운로드: {filename}")
                        else:
                            logger.debug(f"파일 존재: {filename}")
                            
                    except Exception as e:
                        logger.warning(f"PDF 다운로드 실패: {e}")
                    
                    time.sleep(1)
                
                logger.info(f"[{domain}] 수집 완료: {domain_count}편")
                
            except Exception as e:
                logger.error(f"[{domain}] 검색 실패: {e}")
        
        logger.info(f"총 {len(papers)}편 수집 완료")
        return papers
    
    def save_metadata(self, data: list[dict[str, Any]], output_path: Path | None = None) -> None:
        """메타데이터 저장"""
        if output_path is None:
            output_path = self.output_dir / "metadata.json"
        
        metadata = {
            "collection_info": {
                "created_date": datetime.now().isoformat(),
                "total_papers": len(data),
                "source": "ArXiv",
                "domains": list(self.domain_search_keywords.keys()),
            },
            "papers": data
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        logger.info(f"메타데이터 저장: {output_path}")
    
    def generate_report(self, papers: list[dict[str, Any]]) -> None:
        """수집 보고서 생성"""
        domain_stats: dict[str, int] = {}
        for paper in papers:
            domain = paper.get('domain', 'General')
            domain_stats[domain] = domain_stats.get(domain, 0) + 1
        
        domain_breakdown = '\n'.join([
            f'- {domain}: {count}편'
            for domain, count in sorted(domain_stats.items(), key=lambda x: x[1], reverse=True)
        ])
        
        report = f"""
반도체 제조 ArXiv 논문 수집 보고서
=================================

수집 날짜: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
총 논문 수: {len(papers)}

도메인별 분류:
{domain_breakdown}

다음 단계:
1. preprocessor.py로 PDF 텍스트 추출 및 청킹
2. faiss_manager.py로 벡터DB 구축
3. RAG 시스템 통합
"""
        
        report_file = self.output_dir / "scraping_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"수집 보고서 저장: {report_file}")
        print(report)




