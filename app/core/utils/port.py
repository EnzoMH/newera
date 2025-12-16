"""Port Utility"""
import os
import logging

logger = logging.getLogger(__name__)


def get_port_from_env(default: int = 8000) -> int:
    try:
        port_str = os.getenv("API_PORT", str(default))
        port = int(port_str)
        if 1 <= port <= 65535:
            return port
        return default
    except ValueError:
        return default
