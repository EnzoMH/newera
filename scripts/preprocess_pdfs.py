"""
PDF 전처리 스크립트 (텍스트 추출 + 청킹)
"""
import sys
from pathlib import Path
import argparse
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.crawl.preprocessor import DocumentPreprocessor


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="PDF 전처리 (텍스트 추출 + 청킹)")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/crawled/ArXiv",
        help="PDF 파일 디렉토리"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/chunks",
        help="청크 출력 디렉토리"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="청크 크기 (문자 수)"
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=100,
        help="청크 오버랩"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("PDF 전처리 파이프라인")
    print("="*60)
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        logger.error(f"입력 디렉토리가 없습니다: {input_dir}")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    preprocessor = DocumentPreprocessor(
        chunk_size=args.chunk_size,
        overlap=args.overlap
    )
    
    pdf_files = list(input_dir.glob("*.pdf"))
    logger.info(f"발견된 PDF 파일: {len(pdf_files)}개")
    
    processed_count = 0
    total_chunks = 0
    
    for pdf_file in pdf_files:
        logger.info(f"\n처리 중: {pdf_file.name}")
        
        result = preprocessor.process_pdf(pdf_file, output_dir)
        
        if result:
            processed_count += 1
            total_chunks += result.get('total_chunks', 0)
    
    print("\n" + "="*60)
    print("✓ 전처리 완료")
    print(f"  - 처리된 파일: {processed_count}/{len(pdf_files)}")
    print(f"  - 총 청크: {total_chunks}개")
    print(f"  - 출력 디렉토리: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()




