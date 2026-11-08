"""Загрузка настроек из .env.local и проверка обязательных переменных."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env.local"

REQUIRED_VARS = (
    "TELEGRAM_BOT_TOKEN",
    "OPENAI_API_KEY",
    "PINECONE_API_KEY",
    "PINECONE_INDEX",
)

# Размерность векторов для известных моделей эмбеддингов OpenAI.
EMBEDDING_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class ConfigError(RuntimeError):
    """Не хватает переменной окружения или она задана неверно."""


@dataclass(frozen=True)
class Config:
    telegram_token: str
    openai_api_key: str
    openai_base_url: str | None
    openai_model: str
    embedding_model: str
    embedding_dimension: int
    pinecone_api_key: str
    pinecone_index: str
    pinecone_namespace: str
    memory_top_k: int


def load_config() -> Config:
    """Читает .env.local из корня проекта и возвращает настройки."""
    load_dotenv(ENV_FILE)

    missing = [name for name in REQUIRED_VARS if not os.getenv(name, "").strip()]
    if missing:
        raise ConfigError(
            "Не заданы обязательные переменные окружения: "
            + ", ".join(missing)
            + f". Заполните их в файле {ENV_FILE} (образец: .env.local.example)."
        )

    embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "").strip() or "text-embedding-3-small"
    dimension_raw = os.getenv("EMBEDDING_DIMENSION", "").strip()
    if dimension_raw:
        try:
            dimension = int(dimension_raw)
        except ValueError as exc:
            raise ConfigError("EMBEDDING_DIMENSION должна быть целым числом.") from exc
    elif embedding_model in EMBEDDING_DIMENSIONS:
        dimension = EMBEDDING_DIMENSIONS[embedding_model]
    else:
        raise ConfigError(
            f"Неизвестна размерность для модели {embedding_model}. "
            "Задайте EMBEDDING_DIMENSION в .env.local."
        )

    try:
        top_k = int(os.getenv("MEMORY_TOP_K", "").strip() or 5)
    except ValueError as exc:
        raise ConfigError("MEMORY_TOP_K должна быть целым числом.") from exc

    return Config(
        telegram_token=os.environ["TELEGRAM_BOT_TOKEN"].strip(),
        openai_api_key=os.environ["OPENAI_API_KEY"].strip(),
        # Пустое значение означает официальный адрес OpenAI.
        openai_base_url=os.getenv("OPENAI_BASE_URL", "").strip() or None,
        openai_model=os.getenv("OPENAI_MODEL", "").strip() or "gpt-4o-mini",
        embedding_model=embedding_model,
        embedding_dimension=dimension,
        pinecone_api_key=os.environ["PINECONE_API_KEY"].strip(),
        pinecone_index=os.environ["PINECONE_INDEX"].strip(),
        pinecone_namespace=os.getenv("PINECONE_NAMESPACE", "").strip() or "default",
        memory_top_k=top_k,
    )
