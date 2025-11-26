"""
New-RAG Step 1: PDF 로딩
- PDF 파일에서 텍스트 추출
- 원본 텍스트를 JSON으로 저장
"""
import sys
from pathlib import Path
import json
import logging
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import fitz
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def extract_text_pymupdf(pdf_path: Path) -> str:
    """PyMuPDF로 텍스트 추출"""
    try:
        doc = fitz.open(str(pdf_path))
        text_content = ""
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_text = page.get_text()
            if page_text.strip():
                text_content += f"\n=== Page {page_num + 1} ===\n{page_text}"
        
        doc.close()
        return text_content
    except Exception as e:
        logger.error(f"PyMuPDF 실패: {e}")
        return ""


def main():
    parser = argparse.ArgumentParser(description="New-RAG Step 1: PDF 로딩")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/crawled/ArXiv",
        help="PDF 파일 디렉토리"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/new_rag_texts",
        help="텍스트 출력 디렉토리"
    )
    
    args = parser.parse_args()
    
    if not PYMUPDF_AVAILABLE:
        logger.error("PyMuPDF가 설치되지 않았습니다: pip install pymupdf")
        return
    
    print("="*60)
    print("New-RAG Step 1: PDF 로딩")
    print("="*60)
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        logger.error(f"입력 디렉토리가 없습니다: {input_dir}")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pdf_files = list(input_dir.glob("*.pdf"))
    logger.info(f"발견된 PDF 파일: {len(pdf_files)}개")
    
    processed_count = 0
    
    for pdf_file in pdf_files:
        logger.info(f"처리 중: {pdf_file.name}")
        
        text = extract_text_pymupdf(pdf_file)
        
        if text:
            domain = 'General'
            parts = pdf_file.stem.split('_', 2)
            if len(parts) >= 2:
                domain = parts[1]
            
            result = {
                'source': 'ArXiv',
                'filename': pdf_file.name,
                'domain': domain,
                'total_chars': len(text),
                'text': text,
            }
            
            output_file = output_dir / f"text_{pdf_file.stem}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            processed_count += 1
            logger.info(f"저장 완료: {output_file.name} ({len(text)} chars)")
    
    print("\n" + "="*60)
    print("✓ PDF 로딩 완료")
    print(f"  - 처리된 파일: {processed_count}/{len(pdf_files)}")
    print(f"  - 출력 디렉토리: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()

