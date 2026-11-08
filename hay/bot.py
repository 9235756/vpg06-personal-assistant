"""Telegram-бот «Персональный ассистент» на Haystack Agent и Pinecone.

Запуск из корня проекта:  python -m hay.bot
"""

from __future__ import annotations

import logging
import sys

from hay.config import PROJECT_ROOT, ConfigError, load_config

LOG_DIR = PROJECT_ROOT / "logs"
TELEGRAM_LIMIT = 4096

START_TEXT = """Привет! Я персональный ассистент. Вот что я умею:

• запоминаю, что вы мне пишете, и учитываю это в ответах;
• рассказываю факты о кошках;
• показываю случайную собаку и определяю её породу;
• подсказываю погоду в любом городе.

Команды:
/help — подсказка
/clear — забыть всё, что вы мне писали"""

HELP_TEXT = """Примеры запросов:
• «Меня зовут Иван, я люблю горы» — запомню;
• «Расскажи факт о кошках»;
• «Покажи собаку и определи породу»;
• «Какая погода в Москве?»;
• «Как меня зовут и что я люблю?» — отвечу из памяти.

/clear — удалить всю память о вас."""


def setup_logging() -> None:
    LOG_DIR.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_DIR / "bot.log", encoding="utf-8"),
        ],
    )
    # Сетевые библиотеки слишком болтливы на уровне INFO.
    for noisy in ("httpx", "urllib3", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def split_text(text: str, limit: int = TELEGRAM_LIMIT) -> list[str]:
    return [text[i : i + limit] for i in range(0, len(text), limit)] or [""]


def create_bot(config, agent, memory):
    import telebot

    from hay.agent import run_agent

    logger = logging.getLogger("hay.bot")
    bot = telebot.TeleBot(config.telegram_token, parse_mode=None)

    @bot.message_handler(commands=["start"])
    def handle_start(message):
        bot.send_message(message.chat.id, START_TEXT)

    @bot.message_handler(commands=["help"])
    def handle_help(message):
        bot.send_message(message.chat.id, HELP_TEXT)

    @bot.message_handler(commands=["clear"])
    def handle_clear(message):
        try:
            deleted = memory.clear(message.from_user.id)
        except Exception:
            logger.exception("Не удалось очистить память user_id=%s", message.from_user.id)
            bot.send_message(message.chat.id, "Не получилось очистить память. Попробуйте позже.")
            return
        bot.send_message(message.chat.id, f"Память очищена. Удалено записей: {deleted}.")

    @bot.message_handler(content_types=["text"])
    def handle_text(message):
        user_id = message.from_user.id
        text = (message.text or "").strip()
        if not text:
            return
        if text.startswith("/"):
            bot.send_message(message.chat.id, "Не знаю такой команды. Список команд: /help")
            return

        bot.send_chat_action(message.chat.id, "typing")

        # Сначала ищем контекст, потом сохраняем: текущее сообщение не попадёт в свой же контекст.
        try:
            context = memory.retrieve(user_id, text)
        except Exception:
            logger.exception("Ошибка retrieval из Pinecone для user_id=%s", user_id)
            context = []

        try:
            memory.save_user_message(message)
        except Exception:
            logger.exception("Ошибка сохранения в Pinecone для user_id=%s", user_id)

        try:
            reply = run_agent(agent, text, context)
        except Exception:
            logger.exception("Ошибка агента для user_id=%s", user_id)
            bot.send_message(message.chat.id, "Что-то пошло не так. Попробуйте ещё раз чуть позже.")
            return

        for url in reply.image_urls:
            try:
                bot.send_photo(message.chat.id, url)
            except Exception:
                logger.exception("Не удалось отправить фото %s", url)
                bot.send_message(message.chat.id, url)

        for chunk in split_text(reply.text):
            bot.send_message(message.chat.id, chunk)

    @bot.message_handler(
        content_types=["photo", "sticker", "voice", "audio", "video", "document", "video_note", "location", "contact"]
    )
    def handle_other(message):
        # Нетекстовые сообщения не сохраняем и не отправляем агенту.
        bot.send_message(message.chat.id, "Пока я понимаю только текстовые сообщения.")

    return bot


def main() -> None:
    setup_logging()
    logger = logging.getLogger("hay.bot")
    try:
        config = load_config()
    except ConfigError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    from hay.agent import build_agent
    from hay.memory import UserMemory

    memory = UserMemory.from_config(config)
    agent = build_agent(config)
    bot = create_bot(config, agent, memory)
    logger.info("Бот запущен, ожидаю сообщения")
    bot.infinity_polling(skip_pending=True)


if __name__ == "__main__":
    main()
