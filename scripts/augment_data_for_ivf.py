"""
데이터 증강으로 IVF 테스트용 벡터 생성
- 기존 3,647개 → 36,470개 (10배)
- 청크 슬라이딩 윈도우 기법
"""
import sys
from pathlib import Path
import json
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def augment_chunks(chunk_content: str, chunk_id: int) -> list[dict]:
    """
    청크 증강 (슬라이딩 윈도우)
    
    원본 → 5가지 변형:
    1. 원본 (100%)
    2. 앞 70%
    3. 중간 70%
    4. 뒷 70%
    5. 앞 50%
    """
    length = len(chunk_content)
    
    augmented = []
    
    # 1. 원본
    augmented.append({
        'content': chunk_content,
        'type': 'original',
        'chunk_id': f"{chunk_id}_0"
    })
    
    # 2. 앞 70%
    if length > 100:
        augmented.append({
            'content': chunk_content[:int(length * 0.7)],
            'type': 'front_70',
            'chunk_id': f"{chunk_id}_1"
        })
    
    # 3. 중간 70%
    if length > 100:
        start = int(length * 0.15)
        end = int(length * 0.85)
        augmented.append({
            'content': chunk_content[start:end],
            'type': 'middle_70',
            'chunk_id': f"{chunk_id}_2"
        })
    
    # 4. 뒷 70%
    if length > 100:
        augmented.append({
            'content': chunk_content[int(length * 0.3):],
            'type': 'back_70',
            'chunk_id': f"{chunk_id}_3"
        })
    
    # 5. 앞 50%
    if length > 100:
        augmented.append({
            'content': chunk_content[:int(length * 0.5)],
            'type': 'front_50',
            'chunk_id': f"{chunk_id}_4"
        })
    
    return augmented


def main():
    chunks_dir = Path("data/chunks")
    output_dir = Path("data/augmented_chunks")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("데이터 증강: IVF 테스트용")
    print("="*60)
    print("증강 전략: 슬라이딩 윈도우 (5배)")
    print("="*60)
    
    chunk_files = list(chunks_dir.glob("chunks_*.json"))
    logger.info(f"발견된 청크 파일: {len(chunk_files)}개")
    
    total_original = 0
    total_augmented = 0
    
    for chunk_file in chunk_files:
        with open(chunk_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        original_chunks = data.get('chunks', [])
        total_original += len(original_chunks)
        
        augmented_chunks = []
        
        for chunk in original_chunks:
            content = chunk['content']
            chunk_id = chunk['id']
            
            aug_chunks = augment_chunks(content, chunk_id)
            
            for aug in aug_chunks:
                augmented_chunks.append({
                    'id': aug['chunk_id'],
                    'content': aug['content'],
                    'size': len(aug['content']),
                    'type': aug['type'],
                    'original_id': chunk_id,
                })
        
        total_augmented += len(augmented_chunks)
        
        result = {
            'source': data.get('source', 'ArXiv'),
            'filename': data['filename'],
            'domain': data.get('domain', 'General'),
            'total_original': len(original_chunks),
            'total_augmented': len(augmented_chunks),
            'chunks': augmented_chunks,
        }
        
        output_file = output_dir / chunk_file.name
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"{chunk_file.name}: {len(original_chunks)} → {len(augmented_chunks)}개")
    
    print("\n" + "="*60)
    print("✓ 데이터 증강 완료")
    print(f"  - 원본 청크: {total_original:,}개")
    print(f"  - 증강 청크: {total_augmented:,}개")
    print(f"  - 증강 비율: {total_augmented/total_original:.1f}배")
    print(f"  - 출력 디렉토리: {output_dir}")
    print("="*60)
    
    print("\n다음 단계:")
    print("  1. IVF 인덱스 구축:")
    print("     python scripts/build_ivf_index.py")
    print("  2. HNSW vs IVF 벤치마크:")
    print("     python scripts/benchmark_hnsw_vs_ivf.py")


if __name__ == "__main__":
    main()

