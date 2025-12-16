#!/usr/bin/env python3
"""
Exaone 통합 최종 시스템 테스트
"""
import sys
from pathlib import Path
import time

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent))

def test_rag_system_with_exaone():
    """RAG 시스템 Exaone 통합 테스트"""
    print("🚀 Exaone + RAG 시스템 최종 통합 테스트")
    print("=" * 60)
    
    try:
        from app.core.rag import RAGSystem
        
        print("🔄 1. RAG 시스템 초기화 중 (Exaone 모델 로드)...")
        print("   ⚠️  첫 실행 시 모델 다운로드로 1~2분 소요 (700MB)")
        
        start = time.time()
        rag_system = RAGSystem()
        init_success = rag_system.initialize()
        init_time = time.time() - start
        
        if not init_success:
            print("❌ RAG 시스템 초기화 실패")
            return False
        
        print(f"✅ RAG 시스템 초기화 완료 ({init_time:.1f}초)")
        
        # 시스템 상태 확인
        print("\n🔄 2. 시스템 상태 확인...")
        status = rag_system.get_status()
        print(f"   - LLM: {status.get('llm_provider', 'unknown')}")
        print(f"   - VectorDB: {status.get('vector_store', 'unknown')}")
        print(f"   - 초기화 상태: {status.get('initialized', False)}")
        
        # 쿼리 테스트
        test_queries = [
            "안녕하세요",
            "반도체 제조 공정에 대해 설명해주세요",
            "VirtualFab이란 무엇인가요?",
        ]
        
        print("\n🔄 3. RAG 쿼리 테스트...")
        for i, query in enumerate(test_queries, 1):
            print(f"\n   질문 {i}: '{query}'")
            
            start = time.time()
            response = rag_system.query(query, top_k=3)
            query_time = time.time() - start
            
            print(f"   ✅ 응답 생성 완료 ({query_time:.1f}초)")
            print(f"   📝 답변: {response.get('answer', '')[:150]}...")
            print(f"   📚 참고 문서: {len(response.get('sources', []))}개")
        
        print("\n" + "="*60)
        print("🎉 Exaone + RAG 시스템 통합 성공!")
        print("="*60)
        return True
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_agent_with_exaone():
    """LangGraph Agent + Exaone 테스트"""
    print("\n\n🚀 LangGraph Agent + Exaone 테스트")
    print("=" * 60)
    
    try:
        from app.agents.rag_agent import RAGAgent
        from app.core.rag import RAGSystem
        
        print("🔄 1. RAG Agent 초기화 중...")
        
        rag_system = RAGSystem()
        rag_system.initialize()
        
        agent = RAGAgent(rag_system=rag_system)
        agent.initialize()
        
        print("✅ RAG Agent 초기화 완료")
        
        # Agent 상태 확인
        status = agent.get_status()
        print(f"\n📊 Agent 상태:")
        print(f"   - 이름: {status.get('name', 'unknown')}")
        print(f"   - 워크플로우: {'컴파일됨' if status.get('workflow_compiled', False) else '미컴파일'}")
        
        # Agent 쿼리 테스트
        print("\n🔄 2. Agent 쿼리 테스트...")
        test_query = "반도체 8대 공정에 대해 자세히 알려주세요"
        
        print(f"   질문: '{test_query}'")
        start = time.time()
        
        result = agent.process_query(
            question=test_query,
            conversation_id="test_user_exaone",
            use_memory=True
        )
        
        query_time = time.time() - start
        
        print(f"✅ Agent 응답 완료 ({query_time:.1f}초)")
        print(f"📝 답변: {result.get('answer', '')[:200]}...")
        print(f"📚 참고 문서: {len(result.get('sources', []))}개")
        print(f"💬 대화 히스토리: {len(result.get('conversation_history', []))}개")
        
        print("\n" + "="*60)
        print("🎉 LangGraph Agent + Exaone 통합 성공!")
        print("="*60)
        return True
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🌟 Newera VirtualFab RAG System - Exaone 최종 통합 테스트")
    print("=" * 70)
    print()
    
    # 1. RAG 시스템 테스트
    rag_ok = test_rag_system_with_exaone()
    
    # 2. Agent 시스템 테스트
    agent_ok = test_agent_with_exaone()
    
    print("\n\n" + "="*70)
    if rag_ok and agent_ok:
        print("🏆 모든 시스템 테스트 통과!")
        print("✅ Exaone 모델이 RAG 및 Agent 시스템과 완벽히 통합되었습니다.")
    else:
        print("⚠️  일부 시스템 테스트 실패")
    print("="*70)
