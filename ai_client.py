import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_BASE_URL = "https://ai.api.cloud.yandex.net/v1"
DEFAULT_TEMPERATURE = 0.4
DEFAULT_MAX_TOKENS = 12000


def _load_settings() -> dict[str, str | int | float]:
    load_dotenv(PROJECT_ROOT / ".env")

    api_key = os.getenv("YANDEX_API_KEY")
    folder_id = os.getenv("YANDEX_FOLDER_ID")
    model_uri = os.getenv("YANDEX_MODEL_URI")
    base_url = os.getenv("YANDEX_BASE_URL", DEFAULT_BASE_URL)
    temperature_raw = os.getenv("YANDEX_TEMPERATURE", str(DEFAULT_TEMPERATURE))
    max_tokens_raw = os.getenv("YANDEX_MAX_TOKENS", str(DEFAULT_MAX_TOKENS))

    missing = [
        name
        for name, value in {
            "YANDEX_API_KEY": api_key,
            "YANDEX_FOLDER_ID": folder_id,
            "YANDEX_MODEL_URI": model_uri,
        }.items()
        if not value
    ]
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(
            f"Не хватает переменных окружения: {joined}. "
            "Создайте локальный .env по .env.example и заполните значения без публикации секретов."
        )

    try:
        temperature = float(temperature_raw)
    except ValueError as exc:
        raise RuntimeError("YANDEX_TEMPERATURE должен быть числом.") from exc

    if not 0 <= temperature <= 1:
        raise RuntimeError("YANDEX_TEMPERATURE должен быть от 0 до 1.")

    try:
        max_tokens = int(max_tokens_raw)
    except ValueError as exc:
        raise RuntimeError("YANDEX_MAX_TOKENS должен быть целым числом.") from exc

    if max_tokens <= 0:
        raise RuntimeError("YANDEX_MAX_TOKENS должен быть больше 0.")

    return {
        "api_key": api_key or "",
        "folder_id": folder_id or "",
        "model_uri": model_uri or "",
        "base_url": base_url,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }


def _build_client(settings: dict[str, str | int | float]) -> OpenAI:
    return OpenAI(
        api_key=str(settings["api_key"]),
        base_url=str(settings["base_url"]),
        project=str(settings["folder_id"]),
        default_headers={"x-data-logging-enabled": "false"},
    )


def call_model(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> Any:
    """Call AI Studio and return the whole assistant message, including tool_calls."""
    settings = _load_settings()
    client = _build_client(settings)

    request: dict[str, Any] = {
        "model": settings["model_uri"],
        "messages": messages,
        "temperature": settings["temperature"],
        "max_tokens": settings["max_tokens"],
    }

    if tools:
        request["tools"] = tools
        request["tool_choice"] = "auto"

    response = client.chat.completions.create(**request)
    return response.choices[0].message