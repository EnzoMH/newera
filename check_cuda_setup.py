#!/usr/bin/env python3
"""CUDA 및 GPU 라이브러리 확인 스크립트"""
import sys

print("="*60)
print("CUDA 및 GPU 라이브러리 확인")
print("="*60)

# Python 버전
print(f"\n[Python]")
print(f"  버전: {sys.version}")

# PyTorch 확인
try:
    import torch
    print(f"\n[PyTorch]")
    print(f"  버전: {torch.__version__}")
    print(f"  CUDA 사용 가능: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA 버전: {torch.version.cuda}")
        print(f"  cuDNN 버전: {torch.backends.cudnn.version()}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("  ⚠️  CUDA 사용 불가 - PyTorch CUDA 버전 설치 필요")
except ImportError:
    print(f"\n[PyTorch]")
    print("  ⚠️  설치되지 않음")

# Faiss 확인
try:
    import faiss
    print(f"\n[Faiss]")
    print(f"  버전: {faiss.__version__}")
    print(f"  GPU 지원: {hasattr(faiss, 'StandardGpuResources')}")
    if hasattr(faiss, 'get_num_gpus'):
        print(f"  GPU 개수: {faiss.get_num_gpus()}")
    else:
        print("  GPU 개수: 확인 불가")
    
    if hasattr(faiss, 'StandardGpuResources'):
        print("  ✅ Faiss-GPU 설치됨")
    else:
        print("  ⚠️  Faiss-CPU만 설치됨 (GPU 버전 설치 필요)")
except ImportError:
    print(f"\n[Faiss]")
    print("  ⚠️  설치되지 않음")

print("\n" + "="*60)
print("확인 완료!")
print("="*60)


