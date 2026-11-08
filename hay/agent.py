"""Сборка Haystack Agent и запуск с контекстом из памяти."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from hay.memory import format_context
from hay.tools.cat_fact import create_cat_fact_tool
from hay.tools.dog_image import DOG_IMAGE_TOOL, create_dog_analyzer_tool, create_dog_image_tool
from hay.tools.weather import create_weather_tool

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Ты — дружелюбный персональный ассистент в Telegram. Всегда отвечай на русском языке, коротко и по делу.

У тебя есть инструменты:
1. cat_fact — случайный факт о кошках. Вызывай, когда пользователь просит факт о кошках или что-то интересное про котов. Факт приходит на английском: переведи его на русский.
2. dog_image — ссылка на случайную фотографию собаки. Вызывай, когда пользователь просит показать собаку. Бот сам отправит фото, ссылку в ответе не повторяй.
3. dog_image_analyzer — определяет породу собаки по ссылке на фото и кратко рассказывает историю породы. Если пользователь просит показать собаку и определить породу, сначала вызови dog_image, затем dog_image_analyzer с полученной ссылкой.
4. weather — текущая погода в городе. Вызывай, когда спрашивают о погоде, температуре, ветре или как одеться. Если город не назван, спроси его или возьми из контекста памяти, если пользователь там упоминал свой город.

Память:
- Перед сообщением пользователя может стоять блок «Что пользователь писал раньше». Это его прошлые сообщения из долговременной памяти.
- Используй их, чтобы отвечать на вопросы о пользователе: как его зовут, что он любит, что говорил раньше.
- Если в памяти нет ответа, честно скажи, что не знаешь, и не выдумывай факты о пользователе.

Не вызывай инструменты без необходимости. Если инструмент вернул сообщение об ошибке, объясни это пользователю простыми словами."""

URL_RE = re.compile(r"https?://\S+")


@dataclass
class AgentReply:
    text: str
    image_urls: list[str] = field(default_factory=list)


def build_agent(config):
    from haystack.components.agents import Agent
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.utils import Secret

    def make_generator():
        return OpenAIChatGenerator(
            api_key=Secret.from_env_var("OPENAI_API_KEY"),
            model=config.openai_model,
            api_base_url=config.openai_base_url,
        )

    tools = [
        create_cat_fact_tool(),
        create_dog_image_tool(),
        create_dog_analyzer_tool(make_generator()),
        create_weather_tool(),
    ]
    agent = Agent(
        chat_generator=make_generator(),
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        max_agent_steps=10,
    )
    if hasattr(agent, "warm_up"):
        agent.warm_up()
    logger.info(
        "Агент готов: модель=%s, base_url=%s, инструменты=%s",
        config.openai_model,
        config.openai_base_url or "официальный OpenAI",
        [tool.name for tool in tools],
    )
    return agent


def build_user_message(text: str, context_documents: list) -> str:
    context = format_context(context_documents)
    if not context:
        return text
    return f"Что пользователь писал раньше:\n{context}\n\nНовое сообщение пользователя:\n{text}"


def extract_dog_image_urls(messages: list) -> list[str]:
    """Достаёт ссылки на картинки из результатов инструмента dog_image."""
    urls = []
    for message in messages:
        tool_result = getattr(message, "tool_call_result", None)
        if tool_result is None:
            continue
        origin = getattr(tool_result, "origin", None)
        if getattr(origin, "tool_name", None) != DOG_IMAGE_TOOL:
            continue
        if getattr(tool_result, "error", False):
            continue
        urls.extend(URL_RE.findall(str(tool_result.result)))
    return urls


def run_agent(agent, text: str, context_documents: list) -> AgentReply:
    from haystack.dataclasses import ChatMessage

    result = agent.run(messages=[ChatMessage.from_user(build_user_message(text, context_documents))])
    messages = result.get("messages", [])

    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            logger.info("Агент вызвал инструмент %s с аргументами %s", call.tool_name, call.arguments)

    last = result.get("last_message") or (messages[-1] if messages else None)
    answer = (getattr(last, "text", None) or "").strip()
    return AgentReply(
        text=answer or "Не получилось сформулировать ответ. Попробуйте переспросить.",
        image_urls=extract_dog_image_urls(messages),
    )
