"""Долговременная память в Pinecone: только текст сообщений пользователя."""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


def should_save_message(message: Any) -> bool:
    """Решает, нужно ли сохранять сообщение Telegram в память.

    Сохраняется только непустой текст от живого пользователя.
    Команды (/start, /help, /clear и любые другие), нетекстовые сообщения,
    пустые строки и сообщения от ботов не сохраняются.
    """
    if getattr(message, "content_type", None) != "text":
        return False
    text = getattr(message, "text", None)
    if not isinstance(text, str) or not text.strip():
        return False
    if text.strip().startswith("/"):
        return False
    sender = getattr(message, "from_user", None)
    if sender is None or getattr(sender, "is_bot", False):
        return False
    return True


def user_filter(user_id: int | str) -> dict:
    """Фильтр Haystack: документы только этого пользователя."""
    return {"field": "meta.user_id", "operator": "==", "value": str(user_id)}


class UserMemory:
    """Сохраняет сообщения пользователя и достаёт релевантные по косинусному сходству."""

    def __init__(self, document_store, document_embedder, text_embedder, retriever, top_k: int = 5):
        self.store = document_store
        self.document_embedder = document_embedder
        self.text_embedder = text_embedder
        self.retriever = retriever
        self.top_k = top_k

    @classmethod
    def from_config(cls, config) -> "UserMemory":
        from haystack.components.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
        from haystack.utils import Secret
        from haystack_integrations.components.retrievers.pinecone import PineconeEmbeddingRetriever
        from haystack_integrations.document_stores.pinecone import PineconeDocumentStore

        store = PineconeDocumentStore(
            api_key=Secret.from_env_var("PINECONE_API_KEY"),
            index=config.pinecone_index,
            namespace=config.pinecone_namespace,
            dimension=config.embedding_dimension,
            metric="cosine",
            spec={"serverless": {"cloud": "aws", "region": "us-east-1"}},
        )
        openai_kwargs = {
            "api_key": Secret.from_env_var("OPENAI_API_KEY"),
            "model": config.embedding_model,
            "api_base_url": config.openai_base_url,
        }
        document_embedder = OpenAIDocumentEmbedder(**openai_kwargs, progress_bar=False)
        text_embedder = OpenAITextEmbedder(**openai_kwargs)
        retriever = PineconeEmbeddingRetriever(document_store=store, top_k=config.memory_top_k)
        logger.info(
            "Память Pinecone: индекс=%s, namespace=%s, модель эмбеддингов=%s, dimension=%s",
            config.pinecone_index,
            config.pinecone_namespace,
            config.embedding_model,
            config.embedding_dimension,
        )
        return cls(store, document_embedder, text_embedder, retriever, top_k=config.memory_top_k)

    def save_user_message(self, message: Any) -> bool:
        """Сохраняет сообщение, если оно проходит фильтр. Возвращает True при сохранении."""
        if not should_save_message(message):
            logger.debug("Сообщение не сохраняется в Pinecone (команда, пустое или не текст)")
            return False

        from haystack import Document
        from haystack.document_stores.types import DuplicatePolicy

        user_id = str(message.from_user.id)
        chat_id = str(message.chat.id)
        document = Document(
            id=f"{chat_id}-{message.message_id}",
            content=message.text.strip(),
            meta={
                "user_id": user_id,
                "chat_id": chat_id,
                "timestamp": int(getattr(message, "date", None) or time.time()),
                "role": "user",
            },
        )
        embedded = self.document_embedder.run(documents=[document])["documents"]
        written = self.store.write_documents(embedded, policy=DuplicatePolicy.OVERWRITE)
        logger.info(
            "Сохранено в Pinecone: user_id=%s, chat_id=%s, id=%s, документов=%s",
            user_id,
            chat_id,
            document.id,
            written,
        )
        return True

    def retrieve(self, user_id: int | str, query: str, top_k: int | None = None) -> list:
        """Возвращает top_k сообщений пользователя, похожих на запрос."""
        if not query or not query.strip():
            return []
        k = top_k or self.top_k
        embedding = self.text_embedder.run(text=query)["embedding"]
        documents = self.retriever.run(
            query_embedding=embedding,
            filters=user_filter(user_id),
            top_k=k,
        )["documents"]
        # Подстраховка: оставляем только записи этого пользователя с ролью user.
        documents = [
            doc
            for doc in documents
            if str(doc.meta.get("user_id")) == str(user_id) and doc.meta.get("role") == "user"
        ]
        logger.info(
            "Retrieval из Pinecone: user_id=%s, top_k=%s, найдено=%s, score=%s",
            user_id,
            k,
            len(documents),
            [round(doc.score, 3) for doc in documents if doc.score is not None],
        )
        return documents

    def clear(self, user_id: int | str) -> int:
        """Удаляет все документы пользователя. Возвращает число удалённых."""
        filters = user_filter(user_id)
        if hasattr(self.store, "delete_by_filter"):
            deleted = self.store.delete_by_filter(filters)
        else:
            # Старые версии pinecone-haystack: ищем документы и удаляем по id порциями.
            deleted = 0
            for _ in range(100):
                ids = [doc.id for doc in self.store.filter_documents(filters=filters)]
                if not ids:
                    break
                self.store.delete_documents(ids)
                deleted += len(ids)
        logger.info("Очищена память пользователя user_id=%s, удалено=%s", user_id, deleted)
        return deleted or 0


def format_context(documents: list) -> str:
    """Собирает найденные сообщения в блок контекста для агента."""
    lines = [f"- {doc.content}" for doc in documents if getattr(doc, "content", None)]
    return "\n".join(lines)
