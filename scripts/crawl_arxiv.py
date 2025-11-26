"""
ArXiv 논문 크롤링 스크립트
"""
import sys
from pathlib import Path
import argparse
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.crawl.arxiv_crawler import SemiconductorArxivCrawler


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def main():
    parser = argparse.ArgumentParser(description="반도체 관련 ArXiv 논문 크롤링")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/crawled/ArXiv",
        help="논문 저장 디렉토리"
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=100,
        help="도메인당 최대 수집 논문 수"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("반도체 제조 ArXiv 논문 크롤러")
    print("="*60)
    print(f"출력 디렉토리: {args.output_dir}")
    print(f"도메인당 최대 수집: {args.max_results}편")
    print("="*60)
    
    crawler = SemiconductorArxivCrawler(output_dir=args.output_dir)
    
    papers = crawler.scrape(max_results=args.max_results)
    
    crawler.save_metadata(papers)
    
    crawler.generate_report(papers)
    
    print(f"\n✓ 크롤링 완료: 총 {len(papers)}편 수집")


if __name__ == "__main__":
    main()




