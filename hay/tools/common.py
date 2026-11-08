"""Общие помощники для инструментов."""

from __future__ import annotations

import functools
import logging
from typing import Callable

logger = logging.getLogger("hay.tools")

HTTP_TIMEOUT = 10  # секунд


def logged_tool(tool_name: str) -> Callable:
    """Логирует вызов инструмента, его аргументы и результат или ошибку."""

    def decorator(func: Callable[..., str]) -> Callable[..., str]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> str:
            logger.info("Вызов инструмента %s, аргументы: %s", tool_name, kwargs or args)
            try:
                result = func(*args, **kwargs)
            except Exception:
                logger.exception("Инструмент %s упал с ошибкой", tool_name)
                return f"Инструмент {tool_name} временно недоступен. Попробуйте позже."
            logger.info("Инструмент %s вернул: %.200s", tool_name, result)
            return result

        return wrapper

    return decorator
