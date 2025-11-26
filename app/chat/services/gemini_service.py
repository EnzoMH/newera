"""
Google Gemini 통합 서비스
"""
import logging
from typing import Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage


logger = logging.getLogger(__name__)


class GeminiService:
    """
    Google Gemini LLM 서비스
    langchain-google-genai 활용
    """
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-2.5-pro",
        temperature: float = 0.7,
        max_tokens: int = 4096
    ):
        """
        Args:
            api_key: Google API 키
            model_name: Gemini 모델명
            temperature: 생성 온도
            max_tokens: 최대 토큰 수
        """
        self.model_name = model_name
        
        try:
            self.llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                convert_system_message_to_human=True,
            )
            
            logger.info(f"✓ Gemini 서비스 초기화 완료")
            logger.info(f"  - 모델: {model_name}")
            logger.info(f"  - Temperature: {temperature}")
            logger.info(f"  - Max Tokens: {max_tokens}")
            
        except Exception as e:
            logger.error(f"Gemini 초기화 실패: {e}")
            raise
    
    def generate(
        self,
        prompt: str,
        context: str | None = None,
        system_prompt: str | None = None
    ) -> str:
        """
        텍스트 생성
        
        Args:
            prompt: 사용자 프롬프트
            context: RAG 컨텍스트
            system_prompt: 시스템 프롬프트
            
        Returns:
            생성된 텍스트
        """
        messages = []
        
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        
        if context:
            user_content = f"""다음은 관련 문서 정보입니다:

{context}

질문: {prompt}

위 정보를 바탕으로 답변해주세요."""
        else:
            user_content = prompt
        
        messages.append(HumanMessage(content=user_content))
        
        try:
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"생성 실패: {e}")
            return f"오류가 발생했습니다: {str(e)}"
    
    def generate_with_history(
        self,
        prompt: str,
        history: list[dict[str, str]],
        context: str | None = None
    ) -> str:
        """
        대화 히스토리를 포함한 생성
        
        Args:
            prompt: 현재 프롬프트
            history: 대화 히스토리 [{"role": "user"/"assistant", "content": "..."}]
            context: RAG 컨텍스트
            
        Returns:
            생성된 텍스트
        """
        messages = []
        
        for msg in history:
            if msg['role'] == 'user':
                messages.append(HumanMessage(content=msg['content']))
            elif msg['role'] == 'assistant':
                messages.append(AIMessage(content=msg['content']))
        
        if context:
            user_content = f"""다음은 관련 문서 정보입니다:

{context}

질문: {prompt}"""
        else:
            user_content = prompt
        
        messages.append(HumanMessage(content=user_content))
        
        try:
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"생성 실패: {e}")
            return f"오류가 발생했습니다: {str(e)}"

