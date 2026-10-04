"""История диалога и журнал переписки."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class Conversation:
    """Кольцевой буфер последних реплик: у модели есть контекст."""

    limit: int
    messages: list[dict[str, str]] = field(default_factory=list)

    def add(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
        if len(self.messages) > self.limit:
            del self.messages[: len(self.messages) - self.limit]

    def reset(self) -> None:
        self.messages.clear()

    def user_turns(self) -> list[str]:
        return [m["content"] for m in self.messages if m["role"] == "user"]


class Transcript:
    """Пишет диалог в Markdown-файл — из него удобно собрать скриншот для ДЗ."""

    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._session_started = False

    def write(self, role: str, content: str) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)

        if not self._session_started:
            self._path.write_text(
                f"# Лог диалога\n\n_Создан {datetime.now():%d.%m.%Y %H:%M:%S}_\n\n",
                encoding="utf-8",
            )
            self._session_started = True

        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(f"**{role}:** {content}\n\n")

    def note(self, text: str) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(f"_{text}_\n\n")
