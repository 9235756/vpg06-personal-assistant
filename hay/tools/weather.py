"""WeatherTool: текущая погода в городе через бесплатный Open-Meteo (без ключа)."""

from __future__ import annotations

import logging

import requests

from hay.tools.common import HTTP_TIMEOUT, logged_tool

logger = logging.getLogger(__name__)

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
TOOL_NAME = "weather"

# Коды погоды WMO, которые возвращает Open-Meteo.
WEATHER_CODES = {
    0: "ясно",
    1: "преимущественно ясно",
    2: "переменная облачность",
    3: "пасмурно",
    45: "туман",
    48: "туман с изморозью",
    51: "слабая морось",
    53: "морось",
    55: "сильная морось",
    56: "слабая ледяная морось",
    57: "сильная ледяная морось",
    61: "небольшой дождь",
    63: "дождь",
    65: "сильный дождь",
    66: "слабый ледяной дождь",
    67: "сильный ледяной дождь",
    71: "небольшой снег",
    73: "снег",
    75: "сильный снег",
    77: "снежные зёрна",
    80: "небольшой ливень",
    81: "ливень",
    82: "сильный ливень",
    85: "небольшой снегопад",
    86: "сильный снегопад",
    95: "гроза",
    96: "гроза с небольшим градом",
    99: "гроза с сильным градом",
}


class WeatherError(Exception):
    """Ошибка, текст которой можно показать пользователю."""


def describe_weather_code(code) -> str:
    try:
        return WEATHER_CODES.get(int(code), "погодные условия неизвестны")
    except (TypeError, ValueError):
        return "погодные условия неизвестны"


def _get_json(url: str, params: dict) -> dict:
    try:
        response = requests.get(url, params=params, timeout=HTTP_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.Timeout as exc:
        raise WeatherError("Сервис погоды не ответил вовремя. Попробуйте позже.") from exc
    except (requests.RequestException, ValueError) as exc:
        logger.warning("Ошибка запроса к %s: %s", url, exc)
        raise WeatherError("Не удалось получить данные о погоде.") from exc


def geocode_city(city: str) -> dict:
    data = _get_json(
        GEOCODING_URL,
        {"name": city, "count": 1, "language": "ru", "format": "json"},
    )
    results = data.get("results") or []
    if not results:
        raise WeatherError(f"Город «{city}» не найден. Проверьте название.")
    return results[0]


def format_weather(place: dict, current: dict) -> str:
    name = place.get("name", "")
    location = ", ".join(part for part in (name, place.get("country")) if part)
    temperature = current.get("temperature_2m")
    feels_like = current.get("apparent_temperature")
    wind = current.get("wind_speed_10m")
    if temperature is None or feels_like is None or wind is None:
        raise WeatherError("Сервис погоды вернул неполные данные.")
    description = describe_weather_code(current.get("weather_code"))
    return (
        f"Сейчас {location}: {description}, "
        f"температура {round(temperature)}°C, ощущается как {round(feels_like)}°C, "
        f"ветер {round(wind, 1)} м/с."
    )


@logged_tool(TOOL_NAME)
def get_weather(city: str) -> str:
    """Возвращает текущую погоду в городе текстом на русском. Не бросает исключений."""
    city = (city or "").strip()
    if not city:
        return "Не указан город. Уточните, для какого города нужна погода."
    try:
        place = geocode_city(city)
        data = _get_json(
            FORECAST_URL,
            {
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": "temperature_2m,apparent_temperature,wind_speed_10m,weather_code",
                "wind_speed_unit": "ms",
                "timezone": "auto",
            },
        )
        return format_weather(place, data.get("current") or {})
    except WeatherError as exc:
        logger.info("WeatherTool: %s", exc)
        return str(exc)
    except (KeyError, TypeError) as exc:
        logger.warning("Неожиданный формат ответа Open-Meteo: %s", exc)
        return "Сервис погоды вернул ответ в неожиданном формате."


def create_weather_tool():
    from haystack.tools import Tool

    return Tool(
        name=TOOL_NAME,
        description=(
            "Возвращает текущую погоду в указанном городе: описание, температуру, "
            "температуру по ощущениям и скорость ветра. Вызывай, когда пользователь спрашивает "
            "о погоде, температуре, ветре или о том, как одеться в каком-то городе."
        ),
        parameters={
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "Название города, например «Москва» или «Санкт-Петербург».",
                }
            },
            "required": ["city"],
        },
        function=get_weather,
    )
