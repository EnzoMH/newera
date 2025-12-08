#!/usr/bin/env python3
"""
NewEra MCP Server 진입점
Claude Desktop 등 MCP 클라이언트와 연결
"""
import sys
import os
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 환경변수 로드
from dotenv import load_dotenv
load_dotenv()

# MCP 서버 실행
if __name__ == "__main__":
    from app.mcp.server import serve

    print("🚀 NewEra MCP Server 시작 중...")
    print("   VirtualFab RAG System MCP Tools")
    print("   - Web Crawler (ArXiv)")
    print("   - PDF Parser")
    print("   - Vector DB Manager")
    print("   - MongoDB Manager")
    print()

    serve()