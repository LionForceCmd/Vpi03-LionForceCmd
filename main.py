"""Локальный AI-бот в терминале.

Запуск:
    python main.py
    python main.py --model qwen3.5:latest
    python main.py --no-stream
    python main.py "Привет, что ты умеешь?"
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from dataclasses import replace

from ai import AIClient, AIError
from config import Settings, load_settings
from conversation import Conversation, Transcript

# Windows-консоль по умолчанию отдаёт cp1251 и роняет вывод на символах вроде ↳.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

BANNER = r"""
  ___  ___  _    _   ____
 / _ \/ _ \| |  | | / ___|
| |  | | | | |__| | \___ \
| |__| |_| |____  |  ___) |
 \____\___/|_|  |_| |____/
       local ai assistant
"""

HELP = """
Команды:
  /help              эта справка
  /reset             очистить историю диалога
  /history           показать, сколько реплик помнит модель
  /model             текущая модель
  /system            текущий системный промпт
  /exit, /quit       выйти
  Ctrl+C             выйти

Любой другой текст уходит в нейросеть.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Локальный AI-бот в терминале")
    parser.add_argument("prompt", nargs="*", help="разовый вопрос вместо интерактивного режима")
    parser.add_argument("--model", help="переопределить модель")
    parser.add_argument("--provider", help="переопределить провайдера из .env")
    parser.add_argument("--no-stream", action="store_true", help="не печатать ответ по мере генерации")
    parser.add_argument("--no-history", action="store_true", help="ответить один раз и выйти")
    parser.add_argument("--system", help="свой системный промпт")
    return parser.parse_args()


def build_settings(args: argparse.Namespace) -> Settings:
    settings = load_settings()
    changes: dict[str, object] = {}

    if args.provider:
        changes["provider"] = args.provider
    if args.model:
        changes["model"] = args.model
    if args.system:
        changes["system_prompt"] = args.system
    if args.no_stream:
        changes["stream"] = False

    return replace(settings, **changes) if changes else settings


ENV_OVERRIDES: dict[str, str] = {
    "AI_PROVIDER": "provider",
    "AI_BASE_URL": "base_url",
    "AI_MODEL": "model",
    "AI_API_KEY": "api_key",
    "SYSTEM_PROMPT": "system_prompt",
}


def apply_env_overrides(settings: Settings) -> Settings:
    """Позволяет переопределить модель из окружения, не правя .env."""

    changes: dict[str, object] = {}
    for env_key, field_name in ENV_OVERRIDES.items():
        value = os.getenv(env_key)
        if value:
            changes[field_name] = value
    return replace(settings, **changes) if changes else settings


class ConsoleBot:
    """Небольшой REPL поверх AIClient."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._ai = AIClient(settings)
        self._chat = Conversation(limit=settings.history_limit)
        self._log = Transcript(settings.transcript_path) if settings.save_transcript else None

    async def start(self) -> None:
        print(BANNER)
        print(f"  провайдер : {self._settings.provider}")
        print(f"  endpoint  : {self._settings.base_url}")
        print(f"  модель    : {self._settings.model}")
        if self._settings.requires_key and not self._settings.api_key:
            print("  ВНИМАНИЕ  : AI_API_KEY не задан в .env — провайдер его отклонит.")
        print("  /help — список команд\n")

        ok, detail = await self._ai.ping()
        print(f"  {'провайдер отвечает' if ok else 'провайдер недоступен'}: {detail}\n")
        if self._log:
            self._log.note(f"Модель: {self._settings.model} · endpoint: {self._settings.base_url}")

    async def run(self, one_shot: str | None = None) -> None:
        await self.start()

        if one_shot:
            await self.turn(one_shot)
            return

        while True:
            try:
                user_input = input("\033[36mТы ›\033[0m ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n  Пока!")
                return

            if not user_input:
                continue

            if user_input.startswith("/"):
                if not self._command(user_input):
                    return
                continue

            await self.turn(user_input)

    async def turn(self, user_input: str) -> None:
        if self._log:
            self._log.write("Пользователь", user_input)

        started = time.perf_counter()
        try:
            if self._settings.stream:
                answer = await self._stream_answer(user_input)
                stats = f"↳ {time.perf_counter() - started:.1f} с · контекст: {len(self._chat.user_turns())} реплик"
            else:
                reply = await self._ai.ask(self._chat.messages, user_input)
                print(f"\033[32mБот ›\033[0m {reply.text}")
                answer = reply.text
                stats = (
                    f"↳ {reply.seconds:.1f} с · токены: {reply.input_tokens}→{reply.output_tokens}"
                    f" · контекст: {len(self._chat.user_turns())} реплик"
                )
        except AIError as error:
            print(f"\033[31m  Ошибка: {error}\033[0m")
            return
        except KeyboardInterrupt:
            print("\n  Генерация прервана.")
            return

        self._chat.add("user", user_input)
        self._chat.add("assistant", answer)

        if self._log:
            self._log.write("Бот", answer)

        if self._settings.show_timing:
            print(f"\033[90m  {stats}\033[0m")

    async def _stream_answer(self, user_input: str) -> str:
        chunks: list[str] = []
        print("\033[32mБот ›\033[0m ", end="", flush=True)

        async for piece in self._ai.ask_stream(self._chat.messages, user_input):
            chunks.append(piece)
            print(piece, end="", flush=True)

        print()
        return "".join(chunks).strip()

    def _command(self, raw: str) -> bool:
        """Обрабатывает служебные команды. False — пора выходить."""

        command, _, _argument = raw.partition(" ")
        command = command.lower()

        if command in {"/exit", "/quit", "/q"}:
            print("  Пока!")
            return False

        if command == "/help":
            print(HELP)

        elif command == "/reset":
            self._chat.reset()
            print("  История очищена.")

        elif command == "/history":
            turns = self._chat.user_turns()
            print(f"  В контексте {len(turns)} реплик (лимит {self._settings.history_limit}).")
            for index, item in enumerate(turns, start=1):
                print(f"   {index}. {item[:70]}")

        elif command == "/model":
            print(f"  {self._settings.model} через {self._settings.base_url}")
            print("  Сменить: переменная AI_MODEL в .env или флаг --model при запуске.")

        elif command == "/system":
            print(f"  {self._settings.system_prompt}")

        else:
            print("  Неизвестная команда. Список команд: /help")

        return True

    async def close(self) -> None:
        await self._ai.close()


async def amain() -> int:
    args = parse_args()
    settings = apply_env_overrides(build_settings(args))
    bot = ConsoleBot(settings)

    try:
        await bot.run(one_shot=" ".join(args.prompt) if args.prompt else None)
    except KeyboardInterrupt:
        print("\n  Пока!")
    finally:
        await bot.close()

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(amain()))
