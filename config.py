"""Конфигурация локального AI-бота. Все настройки читаются из .env (python-dotenv)."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

#: Пресеты провайдеров. Любой из них работает по OpenAI-совместимому API,
#: поэтому клиент в ai.py один и тот же для всех.
PROVIDERS: dict[str, dict[str, str]] = {
    # Полностью локально и бесплатно: Ollama должен быть запущен (ollama serve)
    "ollama": {"base_url": "http://localhost:11434/v1", "api_key": "ollama", "model": "qwen3.5:latest"},
    # Локальная модель без GPU-акселерации
    "ollama-light": {"base_url": "http://localhost:11434/v1", "api_key": "ollama", "model": "deepseek-coder:latest"},
    # Бесплатные модели OpenRouter (нужен API-ключ, тарификация не требуется)
    "openrouter": {"base_url": "https://openrouter.ai/api/v1", "api_key": "", "model": "deepseek/deepseek-chat-v3-0324:free"},
    # Платный, но очень дешёвый провайдер из урока
    "proxyapi": {"base_url": "https://api.proxyapi.ru/v1", "api_key": "", "model": "gpt-4.1-mini"},
    "deepseek": {"base_url": "https://api.deepseek.com/v1", "api_key": "", "model": "deepseek-chat"},
}

DEFAULT_SYSTEM_PROMPT = (
    "Ты — локальный AI-ассистент, который помогает с вопросами, кодом и текстами. "
    "Отвечай на русском языке, по делу и без воды. "
    "Если вопрос короткий — отвечай коротко. "
    "Если не хватает данных, задай уточняющий вопрос вместо догадок. "
    "Честно говори, когда чего-то не знаешь."
)


def _get(key: str, default: str | None = None) -> str | None:
    value = os.getenv(key)
    if value is None:
        return default
    value = value.strip()
    return value or default


def _get_int(key: str, default: int) -> int:
    raw = _get(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _get_float(key: str, default: float) -> float:
    raw = _get(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _get_bool(key: str, default: bool = False) -> bool:
    raw = _get(key)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on", "да"}


@dataclass(frozen=True)
class Settings:
    """Все параметры приложения, собранные из .env."""

    provider: str
    base_url: str
    model: str
    api_key: str
    system_prompt: str
    temperature: float
    max_tokens: int
    history_limit: int
    timeout: float
    stream: bool
    echo_user: bool
    show_timing: bool
    save_transcript: bool
    transcript_path: str

    @property
    def requires_key(self) -> bool:
        """Ollama не проверяет ключ, остальным провайдерам он нужен."""
        return not self.base_url.startswith("http://localhost")

    @property
    def title(self) -> str:
        return f"{self.provider} · {self.model}"


def load_settings() -> Settings:
    """Читает .env и возвращает готовые настройки."""

    provider = (_get("AI_PROVIDER", "ollama") or "ollama").lower()
    preset = PROVIDERS.get(provider, {})

    return Settings(
        provider=provider,
        base_url=_get("AI_BASE_URL", preset.get("base_url")) or "",
        model=_get("AI_MODEL", preset.get("model")) or "",
        api_key=_get("AI_API_KEY", preset.get("api_key")) or "",
        system_prompt=_get("SYSTEM_PROMPT") or DEFAULT_SYSTEM_PROMPT,
        temperature=_get_float("TEMPERATURE", 0.7),
        max_tokens=_get_int("MAX_TOKENS", 800),
        history_limit=_get_int("HISTORY_LIMIT", 10),
        timeout=_get_float("TIMEOUT", 120.0),
        stream=_get_bool("STREAM", True),
        echo_user=_get_bool("ECHO_USER", True),
        show_timing=_get_bool("SHOW_TIMING", True),
        save_transcript=_get_bool("SAVE_TRANSCRIPT", True),
        transcript_path=_get("TRANSCRIPT_PATH", "logs/dialog.md") or "logs/dialog.md",
    )
