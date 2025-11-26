"""
New-RAG Step 2: 청킹
- LangChain RecursiveCharacterTextSplitter 사용
- 512 토큰 기준
"""
import sys
from pathlib import Path
import json
import logging
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.vecdb.chunking_service import ChunkingService

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="New-RAG Step 2: 청킹")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/new_rag_texts",
        help="텍스트 파일 디렉토리"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/new_rag_chunks",
        help="청크 출력 디렉토리"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="청크 크기 (토큰 근사)"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="청크 오버랩"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("New-RAG Step 2: 청킹")
    print("="*60)
    print(f"- chunk_size: {args.chunk_size} (토큰 근사)")
    print(f"- chunk_overlap: {args.chunk_overlap}")
    print(f"- LangChain RecursiveCharacterTextSplitter")
    print("="*60)
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        logger.error(f"입력 디렉토리가 없습니다: {input_dir}")
        logger.info("먼저 1_load_pdf.py를 실행하세요")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    chunking_service = ChunkingService(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
    text_files = list(input_dir.glob("text_*.json"))
    logger.info(f"발견된 텍스트 파일: {len(text_files)}개")
    
    processed_count = 0
    total_chunks = 0
    
    for text_file in text_files:
        logger.info(f"처리 중: {text_file.name}")
        
        with open(text_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        text = data.get('text', '')
        chunks = chunking_service.chunk_text(text)
        
        result = {
            'source': data.get('source', 'ArXiv'),
            'filename': data.get('filename', ''),
            'domain': data.get('domain', 'General'),
            'total_chars': data.get('total_chars', 0),
            'total_chunks': len(chunks),
            'chunks': chunks,
        }
        
        output_file = output_dir / f"chunks_{text_file.stem.replace('text_', '')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        processed_count += 1
        total_chunks += len(chunks)
        logger.info(f"저장 완료: {output_file.name} ({len(chunks)}개 청크)")
    
    print("\n" + "="*60)
    print("✓ 청킹 완료")
    print(f"  - 처리된 파일: {processed_count}/{len(text_files)}")
    print(f"  - 총 청크: {total_chunks}개")
    print(f"  - 출력 디렉토리: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()

