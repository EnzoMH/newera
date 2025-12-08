"""
에러 핸들러 미들웨어
단일 책임: 전역 예외 처리 및 에러 응답 생성
"""
import logging
from typing import Callable
from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException

from ..schemas import ErrorResponse
from datetime import datetime

logger = logging.getLogger(__name__)


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """
    전역 예외 처리 미들웨어
    
    모든 예외를 일관된 형식으로 처리합니다.
    """
    
    async def dispatch(self, request: Request, call_next: Callable):
        """
        요청 처리 및 예외 처리
        
        Args:
            request: FastAPI Request 객체
            call_next: 다음 미들웨어/엔드포인트 호출 함수
            
        Returns:
            Response: HTTP 응답
        """
        try:
            response = await call_next(request)
            return response
            
        except RequestValidationError as e:
            # 요청 검증 오류
            logger.warning(f"요청 검증 실패: {e.errors()}")
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content=ErrorResponse(
                    error="요청 데이터 검증 실패",
                    code="VALIDATION_ERROR",
                    details={"errors": e.errors()}
                ).model_dump()
            )
            
        except HTTPException as e:
            # FastAPI HTTP 예외
            logger.warning(f"HTTP 예외 발생: {e.status_code} - {e.detail}")
            return JSONResponse(
                status_code=e.status_code,
                content=ErrorResponse(
                    error=str(e.detail),
                    code=f"HTTP_{e.status_code}",
                    details={"status_code": e.status_code}
                ).model_dump()
            )
            
        except StarletteHTTPException as e:
            # Starlette HTTP 예외
            logger.warning(f"Starlette HTTP 예외 발생: {e.status_code} - {e.detail}")
            return JSONResponse(
                status_code=e.status_code,
                content=ErrorResponse(
                    error=str(e.detail),
                    code=f"HTTP_{e.status_code}",
                    details={"status_code": e.status_code}
                ).model_dump()
            )
            
        except Exception as e:
            # 기타 예외
            logger.error(
                f"예상치 못한 오류 발생: {str(e)}",
                exc_info=True
            )
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=ErrorResponse(
                    error="서버 내부 오류가 발생했습니다",
                    code="INTERNAL_SERVER_ERROR",
                    details={"error_type": type(e).__name__}
                ).model_dump()
            )

