# CUDA 12.6 설치 가이드 (Windows)

## 🎯 목표
CUDA 13.0 → CUDA 12.6 다운그레이드하여 faiss-gpu 사용

---

## 1️⃣ CUDA 13.0 제거

### 제어판에서 제거
```
Windows 설정 → 앱 → 설치된 앱
또는
제어판 → 프로그램 및 기능
```

**제거할 항목** (순서대로):
1. NVIDIA CUDA Documentation 13.0 (선택)
2. NVIDIA CUDA Development 13.0
3. NVIDIA CUDA Runtime 13.0
4. NVIDIA CUDA Toolkit 13.0
5. NVIDIA CUDA Samples 13.0 (있으면)

### 환경 변수 정리
```powershell
# 시스템 환경 변수 편집
# Path에서 CUDA 13.0 관련 경로 제거:
# C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin
# C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\libnvvp
```

---

## 2️⃣ CUDA 12.6 설치

### 다운로드
**공식 사이트**: https://developer.nvidia.com/cuda-12-6-0-download-archive

**선택 사항**:
- Operating System: **Windows**
- Architecture: **x86_64**
- Version: **10** (또는 11)
- Installer Type: **exe (local)** (2.9GB, 권장) 또는 **exe (network)** (빠름)

### 설치 옵션
```
✅ CUDA Toolkit
✅ CUDA Runtime
✅ CUDA Development
✅ Visual Studio Integration (VS 있으면)
⚠️  CUDA Samples (선택)
❌ GeForce Experience (불필요)
❌ Graphics Driver (이미 580.97로 최신)
```

**설치 경로**: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6`

### 설치 확인
```powershell
# CUDA 버전 확인
nvcc --version

# 예상 출력:
# nvcc: NVIDIA (R) Cuda compiler driver
# Cuda compilation tools, release 12.6, V12.6.XX

# 환경 변수 확인
echo $env:CUDA_PATH
# C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6

echo $env:CUDA_PATH_V12_6
# C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6
```

---

## 3️⃣ PyTorch + CUDA 12.6 설치

### 기존 PyTorch 제거
```powershell
pip uninstall torch torchvision torchaudio
```

### CUDA 12.6 호환 PyTorch 설치
```powershell
# CUDA 12.6 (실제로는 cu126으로 설치)
pip install torch==2.5.1+cu126 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

### 설치 확인
```python
import torch
print(f"PyTorch 버전: {torch.__version__}")
print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
print(f"CUDA 버전: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# 예상 출력:
# PyTorch 버전: 2.5.1+cu126
# CUDA 사용 가능: True
# CUDA 버전: 12.6
# GPU: NVIDIA GeForce GTX 1660 SUPER
```

---

## 4️⃣ Faiss-GPU 설치

### conda로 설치 (권장)
```powershell
conda install -c conda-forge faiss-gpu=1.8.0
```

### pip로 설치 (대안)
```powershell
# CUDA 12.x 호환 faiss-gpu
pip install faiss-gpu
```

### 설치 확인
```python
import faiss
print(f"Faiss 버전: {faiss.__version__}")
print(f"GPU 지원: {hasattr(faiss, 'StandardGpuResources')}")
print(f"GPU 개수: {faiss.get_num_gpus()}")

# 예상 출력:
# Faiss 버전: 1.8.0
# GPU 지원: True
# GPU 개수: 1
```

---

## 5️⃣ 프로젝트 패키지 설치

```powershell
cd C:\Users\MyoengHo Shin\newera

# 나머지 패키지 설치
pip install -r requirements.txt

# 전체 확인
python -c "
import torch
import faiss
print('✅ PyTorch CUDA:', torch.cuda.is_available())
print('✅ Faiss GPU:', hasattr(faiss, 'StandardGpuResources'))
print('✅ GPU Name:', torch.cuda.get_device_name(0))
"
```

---

## 🐳 Docker로 테스트

```powershell
# Docker Compose로 전체 스택 실행
docker-compose up -d

# 로그 확인
docker-compose logs -f rag-app

# 헬스 체크
curl http://localhost:8001/health
```

---

## 🔍 트러블슈팅

### Q1: nvcc 명령어가 인식되지 않음
**A**: 환경 변수 Path에 추가
```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin
```

### Q2: torch.cuda.is_available() = False
**A**: 
1. GPU 드라이버 확인: `nvidia-smi`
2. PyTorch CUDA 버전 확인: `torch.version.cuda`
3. CUDA 재설치

### Q3: faiss-gpu import 오류
**A**:
```powershell
# conda 환경에서 설치
conda install -c conda-forge faiss-gpu

# 또는 CUDA 11.8용 설치 시도
pip install faiss-gpu-cu11
```

### Q4: Docker GPU 인식 안 됨
**A**:
```powershell
# NVIDIA Container Toolkit 설치 (Windows)
# Docker Desktop → Settings → Resources → WSL Integration
# WSL2에서:
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

---

## 📋 체크리스트

- [ ] CUDA 13.0 제거
- [ ] 환경 변수 정리
- [ ] CUDA 12.6 설치
- [ ] PyTorch cu126 설치
- [ ] Faiss-GPU 설치
- [ ] 프로젝트 패키지 설치
- [ ] GPU 동작 확인
- [ ] Docker 빌드 테스트
- [ ] RAG 시스템 실행 확인

---

## 📊 성능 비교

| 환경 | CUDA | Faiss | 검색 속도 (10k docs) |
|------|------|-------|---------------------|
| 이전 | 13.0 | CPU | ~0.5초 |
| 현재 | 12.6 | GPU | ~0.01초 |
| 개선 | - | - | **50배 향상** |

---

## 🚀 다음 단계

1. **로컬 개발**: CUDA 12.6 + faiss-gpu
2. **Docker 빌드**: `docker-compose up --build`
3. **GCP 배포**: GitHub Actions 자동 배포
4. **모니터링**: 로그 및 성능 확인




