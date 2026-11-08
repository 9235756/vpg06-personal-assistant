"""CatFactTool: случайный факт о кошках с catfact.ninja."""

from __future__ import annotations

import logging

import requests

from hay.tools.common import HTTP_TIMEOUT, logged_tool

logger = logging.getLogger(__name__)

CAT_FACT_URL = "https://catfact.ninja/fact"
TOOL_NAME = "cat_fact"


@logged_tool(TOOL_NAME)
def get_cat_fact() -> str:
    """Возвращает факт о кошках (на английском, агент переводит его сам)."""
    try:
        response = requests.get(CAT_FACT_URL, timeout=HTTP_TIMEOUT)
        response.raise_for_status()
        fact = response.json().get("fact", "").strip()
    except requests.Timeout:
        logger.warning("catfact.ninja не ответил за %s с", HTTP_TIMEOUT)
        return "Сервис фактов о кошках не ответил вовремя. Попробуйте позже."
    except (requests.RequestException, ValueError) as exc:
        logger.warning("Ошибка запроса к catfact.ninja: %s", exc)
        return "Не удалось получить факт о кошках."
    return fact or "Сервис фактов о кошках вернул пустой ответ."


def create_cat_fact_tool():
    from haystack.tools import Tool

    return Tool(
        name=TOOL_NAME,
        description=(
            "Возвращает случайный интересный факт о кошках. "
            "Вызывай, когда пользователь просит факт о кошках или что-то интересное про котов. "
            "Факт приходит на английском, переведи его на русский."
        ),
        parameters={"type": "object", "properties": {}, "required": []},
        function=get_cat_fact,
    )
