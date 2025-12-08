"""
Web UI Layer - Gradio Interface
단일 책임: 사용자 친화적인 채팅 인터페이스 제공
"""
import logging
import gradio as gr
from typing import Tuple, List

from ..core.rag import RAGSystem

logger = logging.getLogger(__name__)


def create_gradio_app(rag_system: RAGSystem) -> gr.Blocks:
    """
    VirtualFab RAG System용 Gradio 인터페이스 생성

    Args:
        rag_system: 초기화된 RAG 시스템 인스턴스

    Returns:
        Gradio Blocks 애플리케이션
    """

    def chat_with_rag(message: str, history: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """
        RAG 시스템과 채팅

        Args:
            message: 사용자 메시지
            history: 채팅 히스토리

        Returns:
            업데이트된 채팅 히스토리 (튜플 리스트)
        """
        if not message or message.strip() == "":
            return history

        try:
            logger.info(f"🎨 웹 UI 질의: {message[:50]}...")

            # RAG 시스템에 질의
            result = rag_system.query(message)

            # 응답 구성
            answer = result["answer"]
            sources = result.get("sources", [])
            metadata = result.get("metadata", {})

            # 응답 포맷팅
            response_parts = [answer]

            # 메타정보 추가
            if metadata:
                response_parts.append("\n\n---")
                response_parts.append(f"🤖 모델: {metadata.get('model', 'Unknown')}")

                if sources:
                    response_parts.append(f"📚 참고 문서: {len(sources)}개")
                else:
                    response_parts.append("💭 일반 대화 모드")

            bot_response = "\n".join(response_parts)

            # 히스토리에 새 메시지 추가 (Gradio 형식: [(user, bot), ...])
            history.append((message, bot_response))

            return history

        except Exception as e:
            logger.error(f"🎨 웹 UI 오류: {e}")
            error_message = f"❌ 오류가 발생했습니다: {str(e)}"
            history.append((message, error_message))
            return history

    # Gradio 인터페이스 생성
    with gr.Blocks(
        title="🔬 VirtualFab RAG 시스템",
        theme=gr.themes.Soft()
    ) as demo:

        gr.Markdown("""
        # VirtualFab RAG 시스템

        반도체 제조(VirtualFab/Digital Twin) 도메인 특화 AI 어시스턴트입니다.

        ## HOW-TO-USE | 사용법
        - 반도체 공정, Virtual Metrology, Digital Twin 등에 대해 물어보세요
        - 구체적인 질문일수록 더 정확한 답변을 받을 수 있습니다

        ## EXAMPLES | 예시 질문
        - 반도체 8대 공정에 대해 설명해주세요
        - Virtual Metrology란 무엇인가요?
        - Digital Twin의 장점은 무엇인가요?
        - Predictive Maintenance는 어떻게 작동하나요?
        """)

        # 채팅 인터페이스
        chatbot = gr.Chatbot(
            height=500,
            show_label=False,
            container=True
        )

        # 입력 텍스트박스
        msg = gr.Textbox(
            placeholder="질문을 입력하세요...",
            show_label=False,
            container=False
        )

        # 버튼들
        with gr.Row():
            submit_btn = gr.Button("전송", variant="primary")
            clear_btn = gr.Button("대화 초기화")

        # 예시 질문들
        gr.Examples(
            examples=[
                "반도체 8대 공정에 대해 설명해주세요",
                "Virtual Metrology란 무엇인가요?",
                "Digital Twin의 장점은 무엇인가요?",
                "Predictive Maintenance는 어떻게 작동하나요?",
                "Yield Management 전략은 무엇인가요?"
            ],
            inputs=msg,
            label="빠른 시작 예시들"
        )

        # 이벤트 핸들러
        msg.submit(
            fn=chat_with_rag,
            inputs=[msg, chatbot],
            outputs=[chatbot]
        ).then(
            fn=lambda: "",
            inputs=None,
            outputs=[msg]
        )

        submit_btn.click(
            fn=chat_with_rag,
            inputs=[msg, chatbot],
            outputs=[chatbot]
        ).then(
            fn=lambda: "",
            inputs=None,
            outputs=[msg]
        )

        clear_btn.click(
            fn=lambda: ([], ""),
            inputs=None,
            outputs=[chatbot, msg]
        )

    logger.info("Gradio 인터페이스 생성 완료")
    return demo
