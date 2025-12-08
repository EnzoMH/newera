"""
로깅 미들웨어
단일 책임: HTTP 요청/응답 로깅
"""
import time
import logging
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    HTTP 요청/응답 로깅 미들웨어
    
    모든 HTTP 요청과 응답을 로깅합니다.
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        요청 처리 및 로깅
        
        Args:
            request: FastAPI Request 객체
            call_next: 다음 미들웨어/엔드포인트 호출 함수
            
        Returns:
            Response: HTTP 응답
        """
        # 요청 시작 시간
        start_time = time.time()
        
        # 요청 정보 로깅
        client_host = request.client.host if request.client else "unknown"
        method = request.method
        path = request.url.path
        query_params = str(request.query_params) if request.query_params else ""
        
        logger.info(
            f"📥 요청 시작: {method} {path}"
            f"{f'?{query_params}' if query_params else ''} "
            f"(클라이언트: {client_host})"
        )
        
        try:
            # 다음 미들웨어/엔드포인트 호출
            response = await call_next(request)
            
            # 처리 시간 계산
            process_time = time.time() - start_time
            
            # 응답 정보 로깅
            status_code = response.status_code
            logger.info(
                f"📤 응답 완료: {method} {path} "
                f"→ {status_code} "
                f"({process_time:.3f}초)"
            )
            
            # 응답 헤더에 처리 시간 추가
            response.headers["X-Process-Time"] = str(process_time)
            
            return response
            
        except Exception as e:
            # 오류 발생 시 로깅
            process_time = time.time() - start_time
            logger.error(
                f"❌ 요청 처리 실패: {method} {path} "
                f"→ 오류 발생 ({process_time:.3f}초): {str(e)}",
                exc_info=True
            )
            raise

