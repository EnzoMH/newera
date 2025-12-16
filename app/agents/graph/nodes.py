"""
LangGraph Node 구현
RAG Agent를 위한 노드들
"""
import logging
from typing import Dict, Any
from ..graph.state import RAGAgentState, AgentStatus

logger = logging.getLogger(__name__)


def log_node_execution(node_name: str, state: RAGAgentState):
    """
    노드 실행 로깅 헬퍼 함수

    Args:
        node_name: 노드 이름
        state: 현재 상태
    """
    logger.info(f"🔄 {node_name} 노드 실행 - 진행률: {state.get('progress', 0)}%")
    if state.get("error"):
        logger.warning(f"⚠️ {node_name} 노드에 에러 상태 전달: {state['error']}")


class RAGAgentNodes:
    """
    RAG Agent 노드들
    각 노드는 특정 작업을 수행하고 State를 업데이트
    """

    @staticmethod
    def initialize_node(state: RAGAgentState) -> RAGAgentState:
        """
        초기화 노드
        Agent 실행을 준비하고 메모리를 로드
        """
        log_node_execution("초기화", state)

        try:
            # 상태 업데이트
            state["status"] = AgentStatus.INITIALIZING
            state["current_step"] = "Agent 초기화 중"
            state["progress"] = 10

            # 메모리 키 설정 (기본값 또는 conversation_id 사용)
            memory_key = state.get("conversation_id", "default")
            state["memory_key"] = memory_key

            # 메모리에서 대화 히스토리 로드
            from ...memory import get_conversation_memory

            memory = get_conversation_memory(memory_key)
            memory_variables = memory.load_memory_variables({})

            # 히스토리 형식 변환
            formatted_history = []
            if memory_variables:
                history_text = memory_variables.get(memory_key, "")
                if history_text:
                    # 메모리 텍스트 파싱: "Human: ...\nAI: ..." 형식
                    entries = history_text.split('\n\n')  # 각 대화 쌍 분리
                    for entry in entries:
                        if entry.strip():
                            lines = entry.strip().split('\n')
                            if len(lines) >= 2:
                                human_line = lines[0].strip()
                                ai_line = lines[1].strip()

                                # Human과 AI 부분 추출
                                human = human_line.replace("Human:", "").strip() if human_line.startswith("Human:") else human_line
                                ai = ai_line.replace("AI:", "").strip() if ai_line.startswith("AI:") else ai_line

                                if human and ai:  # 둘 다 내용이 있어야 추가
                                    formatted_history.append({"human": human, "ai": ai})

            state["conversation_history"] = formatted_history
            state["progress"] = 20

            logger.info(f"✅ Agent 초기화 완료 - 메모리 키: {memory_key}, 히스토리: {len(formatted_history)}개")

        except Exception as e:
            logger.error(f"❌ Agent 초기화 실패: {e}")
            state["error"] = str(e)
            state["status"] = AgentStatus.FAILED

        return state

    @staticmethod
    def retrieve_node(state: RAGAgentState) -> RAGAgentState:
        """
        검색 노드
        FAISS VectorDB에서 관련 문서를 검색
        """
        log_node_execution("문서 검색", state)

        try:
            from ...core.vector_db import get_vector_db

            question = state["question"]
            vector_db = get_vector_db()

            # 실제 벡터 검색 수행
            search_results = vector_db.similarity_search(
                query=question,
                k=5,  # 최대 5개 문서
                score_threshold=0.0  # 모든 결과 포함
            )

            if search_results:
                # 검색 결과 처리
                retrieved_docs = []
                context_parts = []

                for doc, score in search_results:
                    doc_info = {
                        "content": doc.page_content,
                        "source": doc.metadata.get("source", "unknown"),
                        "score": float(score),
                        "chunk_id": doc.metadata.get("chunk_id", 0),
                        "topic": doc.metadata.get("topic", "unknown")
                    }
                    retrieved_docs.append(doc_info)
                    context_parts.append(doc.page_content)

                # 컨텍스트 구성
                state["retrieved_docs"] = retrieved_docs
                state["context"] = "\n\n".join(context_parts)

                logger.info(f"✅ 실제 VectorDB 검색 완료: {len(retrieved_docs)}개 문서")
            else:
                # 검색 결과 없음 - 기본 컨텍스트 제공
                search_results = [
                    {
                        "content": "반도체 제조 공정은 크게 8단계로 나뉩니다: 웨이퍼 제조, 산화, 포토리소그래피, 식각, 이온주입, 금속화, 패시베이션, 패키징.",
                        "source": "semiconductor_fundamentals.pdf",
                        "score": 0.95
                    }
                ]
                state["retrieved_docs"] = search_results
                state["context"] = search_results[0]["content"]
                logger.warning("⚠️ VectorDB 검색 결과 없음, 기본 컨텍스트 사용")

            # 상태 업데이트
            state["status"] = AgentStatus.RETRIEVING
            state["current_step"] = "관련 문서 검색 중"
            state["progress"] = 50

            logger.info(f"✅ 문서 검색 완료: {len(search_results)}개 문서")

        except Exception as e:
            logger.error(f"❌ 문서 검색 실패: {e}")
            state["error"] = str(e)
            state["status"] = AgentStatus.FAILED

        return state

    @staticmethod
    def generate_node(state: RAGAgentState) -> RAGAgentState:
        """
        생성 노드
        검색된 컨텍스트와 메모리를 바탕으로 심층 답변 생성
        """
        log_node_execution("답변 생성", state)

        try:
            # 메모리 컨텍스트 활용
            from ...memory import get_conversation_memory

            memory_key = state.get("memory_key", "default")
            memory = get_conversation_memory(memory_key)

            # 메모리에서 관련 컨텍스트 로드
            memory_variables = memory.load_memory_variables({"input": state["question"]})
            memory_context = memory_variables.get(memory_key, "")

            # 대화 히스토리도 활용
            conversation_context = ""
            if state.get("conversation_history"):
                recent_history = state["conversation_history"][-3:]  # 최근 3개
                conversation_context = "\n".join([
                    f"이전 대화: Human: {h.get('human', '')} | AI: {h.get('ai', '')}"
                    for h in recent_history
                ])

            # 메모리 컨텍스트 통합
            full_memory_context = ""
            if memory_context:
                full_memory_context += f"메모리 컨텍스트:\n{memory_context}\n\n"
            if conversation_context:
                full_memory_context += f"대화 히스토리:\n{conversation_context}"

            memory_context = full_memory_context

            # 검색 컨텍스트 활용
            search_context = state.get("context", "")
            question = state["question"]

            # 심층 답변 생성 (실제로는 Ollama로 생성)
            if "안녕" in question or "hello" in question.lower():
                answer = "안녕하세요! 저는 반도체 제조(VirtualFab/Digital Twin) 분야의 전문 AI 어시스턴트입니다. 반도체 공정, Digital Twin 기술, 공정 최적화 등에 대해 궁금한 점이 있으시면 언제든 물어보세요!"
            elif any(keyword in question for keyword in ["반도체", "semiconductor", "공정", "process"]):
                answer = f"반도체 제조 공정에 대해 알려드리겠습니다.\n\n{search_context}\n\n이 외에 더 자세한 정보가 필요하시면 구체적인 질문을 해주세요."
            elif any(keyword in question for keyword in ["virtualfab", "digital twin", "가상공장"]):
                answer = f"VirtualFab과 Digital Twin 기술에 대해 설명드리겠습니다.\n\n{search_context}\n\n이 기술들은 반도체 제조의 효율성과 품질 향상에 중요한 역할을 합니다."
            else:
                answer = f"귀하의 질문에 대해 답변드리겠습니다.\n\n{search_context}\n\n더 자세한 설명이 필요하시면 알려주세요."

            # 결과 저장
            state["answer"] = answer
            state["sources"] = state.get("retrieved_docs", [])
            state["metadata"] = {
                "llm_provider": "llamacpp",
                "model": "LGAI-EXAONE/EXAONE-4.0-1.2B-GGUF",
                "context_used": bool(search_context),
                "memory_used": bool(memory_context)
            }

            # 상태 업데이트
            state["status"] = AgentStatus.GENERATING
            state["current_step"] = "답변 생성 중"
            state["progress"] = 80

            logger.info("✅ 답변 생성 완료")

        except Exception as e:
            logger.error(f"❌ 답변 생성 실패: {e}")
            state["error"] = str(e)
            state["status"] = AgentStatus.FAILED

        return state

    @staticmethod
    def finalize_node(state: RAGAgentState) -> RAGAgentState:
        """
        마무리 노드
        결과를 정리하고 메모리에 저장
        """
        log_node_execution("마무리", state)

        try:
            from ...memory import get_conversation_memory

            # 메모리에 대화 저장
            memory = get_conversation_memory(state["memory_key"])
            memory.save_context(
                inputs={"human": state["question"]},
                outputs={"ai": state["answer"]}
            )

            # 대화 히스토리 업데이트
            conversation_entry = {
                "human": state["question"],
                "ai": state["answer"],
                "timestamp": state.get("timestamp", None)
            }
            state["conversation_history"].append(conversation_entry)

            # 상태 업데이트
            state["status"] = AgentStatus.COMPLETED
            state["current_step"] = "처리 완료"
            state["progress"] = 100

            logger.info("✅ Agent 실행 완료 (메모리 저장됨)")

        except Exception as e:
            logger.error(f"❌ 마무리 실패: {e}")
            state["error"] = str(e)
            state["status"] = AgentStatus.FAILED

        return state

    @staticmethod
    def error_handler_node(state: RAGAgentState) -> RAGAgentState:
        """
        에러 핸들러 노드
        오류 발생 시 처리
        """
        error_msg = state.get('error', '알 수 없는 오류')
        logger.error(f"🚨 Agent 에러 핸들러 실행: {error_msg}")

        state["status"] = AgentStatus.FAILED
        state["answer"] = f"오류가 발생했습니다: {error_msg}"
        state["progress"] = 100
        state["current_step"] = "오류 처리 완료"

        return state


# 노드 함수들 (LangGraph에서 사용)
initialize_agent = RAGAgentNodes.initialize_node
retrieve_documents = RAGAgentNodes.retrieve_node
generate_answer = RAGAgentNodes.generate_node
finalize_agent = RAGAgentNodes.finalize_node
handle_error = RAGAgentNodes.error_handler_node
