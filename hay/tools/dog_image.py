"""DogImageTool и DogImageAnalyzerTool: картинка собаки и определение породы."""

from __future__ import annotations

import logging

import requests

from hay.tools.common import HTTP_TIMEOUT, logged_tool

logger = logging.getLogger(__name__)

DOG_IMAGE_URL = "https://dog.ceo/api/breeds/image/random"
DOG_IMAGE_TOOL = "dog_image"
DOG_ANALYZER_TOOL = "dog_image_analyzer"

ANALYZER_PROMPT = (
    "Определи породу собаки на фотографии. Ответь на русском: "
    "название породы и краткая история породы в 2–3 предложениях. "
    "Если порода неочевидна, назови наиболее вероятные варианты."
)


@logged_tool(DOG_IMAGE_TOOL)
def get_dog_image() -> str:
    """Возвращает ссылку на случайную картинку собаки."""
    try:
        response = requests.get(DOG_IMAGE_URL, timeout=HTTP_TIMEOUT)
        response.raise_for_status()
        data = response.json()
    except requests.Timeout:
        return "Сервис картинок собак не ответил вовремя. Попробуйте позже."
    except (requests.RequestException, ValueError) as exc:
        logger.warning("Ошибка запроса к dog.ceo: %s", exc)
        return "Не удалось получить картинку собаки."
    url = data.get("message", "")
    if data.get("status") != "success" or not str(url).startswith("http"):
        return "Сервис картинок собак вернул некорректный ответ."
    return url


class DogImageAnalyzer:
    """Описывает собаку на картинке через модель OpenAI с поддержкой изображений."""

    def __init__(self, chat_generator):
        self._generator = chat_generator

    def analyze(self, image_url: str) -> str:
        from haystack.dataclasses import ChatMessage, ImageContent

        if not image_url or not image_url.startswith("http"):
            return "Нужна корректная ссылка на картинку."
        try:
            image = ImageContent.from_url(image_url, detail="low")
            message = ChatMessage.from_user(content_parts=[ANALYZER_PROMPT, image])
            result = self._generator.run(messages=[message])
            text = result["replies"][0].text or ""
        except Exception as exc:
            logger.exception("Ошибка анализа картинки %s", image_url)
            return f"Не удалось проанализировать картинку: {exc}"
        return text.strip() or "Модель не смогла описать картинку."


def create_dog_image_tool():
    from haystack.tools import Tool

    return Tool(
        name=DOG_IMAGE_TOOL,
        description=(
            "Возвращает ссылку на случайную фотографию собаки. "
            "Вызывай, когда пользователь просит показать собаку. "
            "Картинку пользователю отправит бот, в ответе ссылку повторять не нужно."
        ),
        parameters={"type": "object", "properties": {}, "required": []},
        function=get_dog_image,
    )


def create_dog_analyzer_tool(chat_generator):
    from haystack.tools import Tool

    analyzer = DogImageAnalyzer(chat_generator)

    @logged_tool(DOG_ANALYZER_TOOL)
    def analyze_dog_image(image_url: str) -> str:
        return analyzer.analyze(image_url)

    return Tool(
        name=DOG_ANALYZER_TOOL,
        description=(
            "Анализирует фотографию собаки по ссылке: определяет породу и кратко рассказывает "
            "её историю. Вызывай после dog_image, если пользователь хочет узнать породу, "
            "или когда пользователь сам прислал ссылку на фото собаки."
        ),
        parameters={
            "type": "object",
            "properties": {
                "image_url": {
                    "type": "string",
                    "description": "Прямая ссылка на изображение собаки (http или https).",
                }
            },
            "required": ["image_url"],
        },
        function=analyze_dog_image,
    )
