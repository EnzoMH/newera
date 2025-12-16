"""
LLM Provider (LlamaCpp - Exaone)
단일 책임: LLM과의 상호작용
"""
# ⚠️ 중요: 다른 import 전에 Jinja2 패치를 먼저 적용
try:
    import llama_cpp.llama_chat_format as chat_format
    from jinja2 import Environment
    
    # Jinja2ChatFormatter 패치 (loopcontrols extension 활성화)
    OriginalFormatter = chat_format.Jinja2ChatFormatter
    
    class PatchedJinja2ChatFormatter(OriginalFormatter):
        def __init__(self, template, eos_token, bos_token, add_generation_prompt=True, stop_token_ids=None):
            # 원본 속성 설정
            self.template = template
            self.eos_token = eos_token
            self.bos_token = bos_token
            self.add_generation_prompt = add_generation_prompt
            self.stop_token_ids = set(stop_token_ids) if stop_token_ids is not None else None
            
            # loopcontrols extension이 활성화된 Jinja2 환경으로 템플릿 컴파일
            from jinja2.sandbox import ImmutableSandboxedEnvironment
            import jinja2
            
            env = ImmutableSandboxedEnvironment(
                loader=jinja2.BaseLoader(),
                extensions=['jinja2.ext.loopcontrols'],  # 핵심: loopcontrols 추가!
                trim_blocks=True,
                lstrip_blocks=True,
            )
            self._environment = env.from_string(self.template)
    
    chat_format.Jinja2ChatFormatter = PatchedJinja2ChatFormatter
except ImportError:
    # llama-cpp-python 미설치 시 스킵
    pass

import logging
from typing import Optional
import os

from app.core.llm.dto import OllamaRequest, OllamaResponse

logger = logging.getLogger(__name__)
logger.info("✅ Jinja2 loopcontrols extension 패치 적용 완료")


class LLMProvider:
    """
    LLM Provider (Ollama 또는 LlamaCpp)
    - 단일 책임: LLM과의 상호작용 및 응답 생성
    """

    def __init__(self, model_name: str = "LGAI-EXAONE/EXAONE-4.0-1.2B-GGUF"):
        # 순수 llama-cpp-python 사용 (LangChain wrapper 버전 충돌 회피)
        from llama_cpp import Llama

        self.model_name = model_name
        self.filename = os.getenv("LLAMA_CPP_FILENAME", "EXAONE-4.0-1.2B-Q4_K_M.gguf")

        # GPU 메모리 설정
        n_gpu_layers = int(os.getenv("LLAMA_CPP_N_GPU_LAYERS", "35"))  # GPU 사용
        n_ctx = int(os.getenv("LLAMA_CPP_N_CTX", "4096"))  # 컨텍스트 길이
        n_batch = int(os.getenv("LLAMA_CPP_N_BATCH", "512"))  # 배치 크기

        logger.info(f"🔄 Exaone 모델 다운로드 및 로드 중... (최초 실행 시 몇 분 소요)")
        logger.info(f"   - Repo: {self.model_name}")
        logger.info(f"   - File: {self.filename}")
        logger.info(f"   - GPU Layers: {n_gpu_layers}")

        try:
            # HuggingFace Hub에서 특정 파일만 다운로드
            from huggingface_hub import hf_hub_download
            
            logger.info(f"📥 모델 파일 다운로드 중: {self.filename} (약 700MB)")
            
            # 특정 파일만 다운로드
            model_path = hf_hub_download(
                repo_id=self.model_name,
                filename=self.filename,
                cache_dir="models/exaone"  # 로컬 캐시
            )
            
            logger.info(f"✅ 모델 다운로드 완료: {model_path}")
            
            # 다운로드된 파일로 모델 로드
            self.llm = Llama(
                model_path=model_path,
                n_gpu_layers=n_gpu_layers,
                n_ctx=n_ctx,
                n_batch=n_batch,
                verbose=False  # 로그 간소화
            )

            logger.info(f"✅ Exaone LlamaCpp LLM 초기화 완료: {self.filename}")
            logger.info(f"   - GPU: {n_gpu_layers} layers, Context: {n_ctx}, Batch: {n_batch}")
            logger.info(f"   - 파일 크기: ~700MB (Q4_K_M 양자화)")

        except Exception as e:
            logger.error(f"❌ Exaone 모델 로드 실패: {e}")
            logger.error("💡 해결 방법:")
            logger.error("   1. llama-cpp-python 버전 확인: pip show llama-cpp-python")
            logger.error("   2. GPU 드라이버 확인 (CUDA 필요)")
            logger.error("   3. 모델 파일 직접 다운로드: https://huggingface.co/" + self.model_name)
            raise

    def generate_response(self, request: OllamaRequest) -> OllamaResponse:
        """
        기본 응답 생성

        Args:
            request: Ollama 요청 객체

        Returns:
            Ollama 응답 객체
        """
        try:
            # 기본 프롬프트 생성
            full_prompt = request.prompt

            # 시스템 프롬프트가 있는 경우 추가
            if request.system_prompt:
                full_prompt = f"[시스템]\n{request.system_prompt}\n\n[질문]\n{request.prompt}"

            # 컨텍스트가 있는 경우 추가
            if request.context:
                full_prompt = f"[컨텍스트]\n{request.context}\n\n{full_prompt}"

            # LLM 호출 (create_chat_completion API)
            response = self.llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": request.system_prompt or "당신은 반도체 제조 분야의 전문 AI 어시스턴트입니다."},
                    {"role": "user", "content": full_prompt}
                ],
                max_tokens=request.max_tokens or 1024,
                temperature=request.temperature or 0.1,
                top_p=0.9,
                top_k=40,
                repeat_penalty=1.1,
                stop=["[|endofturn|]", "Human:", "User:"]
            )

            # 응답 텍스트 추출
            response_text = response['choices'][0]['message']['content'].strip()

            return OllamaResponse(
                response=response_text,
                model_name=self.model_name
            )

        except Exception as e:
            logger.error(f"Ollama 응답 생성 실패: {e}")
            return OllamaResponse(
                response=f"오류가 발생했습니다: {str(e)}",
                model_name=self.model_name
            )

    def generate_simple_response(self, prompt: str, temperature: float = 0.1, max_tokens: int = 1024) -> str:
        """
        간단한 텍스트 응답 생성

        Args:
            prompt: 프롬프트 텍스트
            temperature: 온도 설정 (0.0 ~ 1.0)
            max_tokens: 최대 생성 토큰 수

        Returns:
            응답 텍스트
        """
        try:
            # create_chat_completion API 사용
            response = self.llm.create_chat_completion(
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                top_k=40,
                repeat_penalty=1.1,
                stop=["[|endofturn|]", "Human:", "User:"]
            )

            # 응답 텍스트 추출
            response_text = response['choices'][0]['message']['content'].strip()
            return response_text

        except Exception as e:
            logger.error(f"LLM 간단 응답 생성 실패: {e}")
            return f"오류가 발생했습니다: {str(e)}"

    def is_available(self) -> bool:
        """
        LLM 사용 가능 여부 확인

        Returns:
            사용 가능 여부
        """
        try:
            # 간단한 테스트 호출
            test_response = self.llm.create_chat_completion(
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5
            )
            return bool(test_response)
        except Exception:
            return False


# 호환성을 위한 별칭
OllamaLLMProvider = LLMProvider