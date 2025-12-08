"""
포트 관리 유틸리티
단일 책임: 사용 가능한 포트 찾기 및 포트 할당
"""
import socket
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def is_port_available(port: int, host: str = "0.0.0.0") -> bool:
    """
    포트가 사용 가능한지 확인
    
    Args:
        port: 확인할 포트 번호
        host: 호스트 주소 (기본값: 0.0.0.0)
        
    Returns:
        포트 사용 가능 여부
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            result = sock.bind((host, port))
            return True
    except OSError:
        return False


def find_available_port(start_port: int = 8000, max_attempts: int = 10, host: str = "0.0.0.0") -> Optional[int]:
    """
    사용 가능한 포트를 찾기 (start_port부터 순차적으로 확인)
    
    Args:
        start_port: 시작 포트 번호 (기본값: 8000)
        max_attempts: 최대 시도 횟수 (기본값: 10)
        host: 호스트 주소 (기본값: 0.0.0.0)
        
    Returns:
        사용 가능한 포트 번호, 없으면 None
        
    Raises:
        RuntimeError: 사용 가능한 포트를 찾을 수 없는 경우
    """
    for attempt in range(max_attempts):
        port = start_port + attempt
        if is_port_available(port, host):
            logger.info(f"✅ 사용 가능한 포트 발견: {port}")
            return port
        else:
            logger.debug(f"포트 {port}는 사용 중입니다. 다음 포트 확인 중...")
    
    raise RuntimeError(
        f"사용 가능한 포트를 찾을 수 없습니다. "
        f"시작 포트: {start_port}, 최대 시도: {max_attempts}"
    )


def get_port_from_env(default: int = 8000) -> int:
    """
    환경변수에서 포트를 가져오거나 사용 가능한 포트를 찾기
    
    Args:
        default: 환경변수가 없을 때 사용할 기본 포트
        
    Returns:
        포트 번호
    """
    import os
    
    env_port = os.getenv("API_PORT")
    if env_port:
        try:
            port = int(env_port)
            if is_port_available(port):
                logger.info(f"환경변수에서 포트 로드: {port}")
                return port
            else:
                logger.warning(
                    f"환경변수 포트 {port}가 사용 중입니다. "
                    f"사용 가능한 포트를 자동으로 찾습니다."
                )
        except ValueError:
            logger.warning(
                f"잘못된 포트 값: {env_port}. "
                f"기본 포트 {default}부터 사용 가능한 포트를 찾습니다."
            )
    
    # 환경변수가 없거나 사용 불가능한 경우 자동 할당
    port = find_available_port(start_port=default)
    return port

