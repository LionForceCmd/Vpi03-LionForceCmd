"""Клиент нейросети: один OpenAI-совместимый класс для Ollama, OpenRouter, DeepSeek и ProxyAPI."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from dataclasses import dataclass

from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, RateLimitError

from config import Settings


class AIError(RuntimeError):
    """Ошибка обращения к языковой модели."""


@dataclass
class Reply:
    """Ответ модели вместе с замерами — их удобно показать в логе."""

    text: str
    seconds: float
    input_tokens: int = 0
    output_tokens: int = 0


class AIClient:
    """Тонкая обёртка над OpenAI-совместимым chat completions API."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = AsyncOpenAI(
            base_url=settings.base_url,
            api_key=settings.api_key or "not-needed",
            timeout=settings.timeout,
            max_retries=1,
        )

    @property
    def settings(self) -> Settings:
        return self._settings

    def _messages(self, history: list[dict[str, str]], user_message: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self._settings.system_prompt},
            *history,
            {"role": "user", "content": user_message},
        ]

    async def ask(self, history: list[dict[str, str]], user_message: str) -> Reply:
        """Один запрос без стриминга."""

        started = time.perf_counter()
        try:
            response = await self._client.chat.completions.create(
                model=self._settings.model,
                messages=self._messages(history, user_message),
                temperature=self._settings.temperature,
                max_tokens=self._settings.max_tokens,
                stream=False,
            )
        except Exception as error:  # noqa: BLE001 - приводим к понятному тексту
            raise AIError(_describe(error)) from error

        text = (response.choices[0].message.content or "").strip()
        usage = response.usage
        return Reply(
            text=text,
            seconds=time.perf_counter() - started,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )

    async def ask_stream(
        self,
        history: list[dict[str, str]],
        user_message: str,
    ) -> AsyncIterator[str]:
        """Отдаёт текст ответа кусочками по мере генерации."""

        try:
            stream = await self._client.chat.completions.create(
                model=self._settings.model,
                messages=self._messages(history, user_message),
                temperature=self._settings.temperature,
                max_tokens=self._settings.max_tokens,
                stream=True,
            )
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content
        except Exception as error:  # noqa: BLE001
            raise AIError(_describe(error)) from error

    async def ping(self) -> tuple[bool, str]:
        """Быстрая проверка, что провайдер доступен и ключ принят."""

        try:
            models = await self._client.models.list()
        except Exception as error:  # noqa: BLE001
            return False, _describe(error)

        available = [item.id for item in getattr(models, "data", [])]
        if available and self._settings.model not in available:
            return True, f"модель «{self._settings.model}» не найдена, доступно: {', '.join(available[:5])}"
        return True, "провайдер отвечает"

    async def close(self) -> None:
        await self._client.close()


def _describe(error: Exception) -> str:
    """Человекочитаемое описание ошибки вместо трейсбека."""

    if isinstance(error, APITimeoutError):
        return "модель не ответила вовремя. Попробуй уменьшить MAX_TOKENS или сменить модель."
    if isinstance(error, RateLimitError):
        return "провайдер отклонил запрос по лимиту. Проверь ключ или баланс."
    if isinstance(error, APIConnectionError):
        return "не удалось подключиться к провайдеру. Проверь base_url и интернет."
    if isinstance(error, FileNotFoundError):
        return "провайдер недоступен. Для Ollama выполни: ollama serve"
    text = str(error).strip()
    return text or error.__class__.__name__
