"""Фильтр сохранения в Pinecone: только текст пользователя. Pinecone замокан."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from hay.memory import UserMemory, should_save_message, user_filter


def make_message(text="Меня зовут Иван", content_type="text", is_bot=False):
    return SimpleNamespace(
        content_type=content_type,
        text=text,
        message_id=42,
        date=1_700_000_000,
        from_user=SimpleNamespace(id=111, is_bot=is_bot),
        chat=SimpleNamespace(id=222),
    )


def make_memory():
    return UserMemory(
        document_store=MagicMock(),
        document_embedder=MagicMock(),
        text_embedder=MagicMock(),
        retriever=MagicMock(),
        top_k=3,
    )


def test_user_text_is_saved():
    assert should_save_message(make_message("Я люблю горы")) is True


@pytest.mark.parametrize("command", ["/start", "/help", "/clear", "/start@my_bot", "  /clear  "])
def test_commands_are_not_saved(command):
    assert should_save_message(make_message(command)) is False


@pytest.mark.parametrize("text", ["", "   ", "\n", None])
def test_empty_messages_are_not_saved(text):
    assert should_save_message(make_message(text)) is False


@pytest.mark.parametrize("content_type", ["photo", "sticker", "voice", "document"])
def test_non_text_messages_are_not_saved(content_type):
    assert should_save_message(make_message("подпись", content_type=content_type)) is False


def test_bot_messages_are_not_saved():
    assert should_save_message(make_message("Ответ бота", is_bot=True)) is False


def test_command_does_not_touch_pinecone():
    memory = make_memory()

    assert memory.save_user_message(make_message("/start")) is False

    memory.document_embedder.run.assert_not_called()
    memory.store.write_documents.assert_not_called()


def test_user_message_written_with_meta():
    pytest.importorskip("haystack")
    memory = make_memory()
    memory.document_embedder.run.side_effect = lambda documents: {"documents": documents}

    assert memory.save_user_message(make_message("Меня зовут Иван")) is True

    written = memory.store.write_documents.call_args.args[0]
    assert len(written) == 1
    doc = written[0]
    assert doc.content == "Меня зовут Иван"
    assert doc.meta == {"user_id": "111", "chat_id": "222", "timestamp": 1_700_000_000, "role": "user"}


def test_retrieve_filters_by_user():
    memory = make_memory()
    memory.text_embedder.run.return_value = {"embedding": [0.1, 0.2]}
    own = SimpleNamespace(content="Меня зовут Иван", meta={"user_id": "111", "role": "user"}, score=0.9)
    foreign = SimpleNamespace(content="чужое", meta={"user_id": "999", "role": "user"}, score=0.8)
    memory.retriever.run.return_value = {"documents": [own, foreign]}

    result = memory.retrieve(111, "Как меня зовут?")

    assert result == [own]
    kwargs = memory.retriever.run.call_args.kwargs
    assert kwargs["filters"] == user_filter(111)
    assert kwargs["top_k"] == 3


def test_clear_deletes_by_user_filter():
    memory = make_memory()
    memory.store.delete_by_filter.return_value = 5

    assert memory.clear(111) == 5
    memory.store.delete_by_filter.assert_called_once_with(
        {"field": "meta.user_id", "operator": "==", "value": "111"}
    )
