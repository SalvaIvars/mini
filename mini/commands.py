from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from prompt_toolkit.completion import Completer, Completion

if TYPE_CHECKING:
    from .display import Display


@dataclass
class Command:
    name: str
    description: str
    category: str = "General"


CommandHandler = Callable[[str], None]


class CommandRegistry:
    """Single source of truth for slash commands and their handlers."""

    def __init__(self, display: Display | None = None):
        self._commands: list[Command] = []
        self._handlers: dict[str, CommandHandler] = {}
        self._display = display

    def register(self, command: Command, handler: CommandHandler):
        self._commands.append(command)
        self._handlers[f"/{command.name}"] = handler

    def get_command(self, raw: str) -> str | None:
        stripped = raw.strip()
        return stripped.split()[0] if " " in stripped else stripped

    def handle(self, raw: str) -> bool:
        cmd_name = self.get_command(raw)
        if cmd_name in ("/exit", "/quit"):
            return False
        handler = self._handlers.get(cmd_name)
        if handler:
            handler(raw)
        elif self._display:
            self._display.unknown_command(raw)
        else:
            print(f"  Unknown command: {raw}. Type /help for available commands.")
        return True

    @property
    def commands(self) -> list[Command]:
        return list(self._commands)

    @property
    def names(self) -> frozenset[str]:
        return frozenset(self._handlers.keys())


class SlashCommandCompleter(Completer):
    def __init__(self, registry: CommandRegistry | None = None):
        self.registry = registry

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if not text.startswith("/"):
            return
        query = text[1:]
        commands = self.registry.commands if self.registry else []
        if not query:
            for cmd in commands:
                yield Completion(
                    f"/{cmd.name}",
                    start_position=-len(text),
                    display=f"/{cmd.name}",
                    display_meta=cmd.description,
                )
        else:
            for cmd in commands:
                score = fuzzy_match(query, cmd.name)
                if score > 0:
                    yield Completion(
                        f"/{cmd.name}",
                        start_position=-len(text),
                        display=f"/{cmd.name}",
                        display_meta=cmd.description,
                    )


def fuzzy_match(query: str, target: str) -> int:
    if not query:
        return 0
    query = query.lower()
    target = target.lower()
    if query in target:
        return 100 + len(query)
    qi = 0
    score = 0
    for ch in target:
        if qi < len(query) and ch == query[qi]:
            qi += 1
            score += 10
    return score if qi == len(query) else 0


# Legacy global list kept for backward compatibility during migration.
COMMANDS: list[Command] = [
    Command("cost", "Show token usage and cost for this session", "Info"),
    Command("calls", "Show tool call count for this session", "Info"),
    Command("show-interior", "Toggle verbose interior mode", "Debug"),
    Command("show-reasoning", "Show model reasoning", "Debug"),
    Command("hide-reasoning", "Hide model reasoning", "Debug"),
    Command("help", "Show all available commands", "Help"),
    Command("summarization", "Force context summarization", "Session"),
    Command("clear", "Clear conversation memory", "Session"),
    Command("exit", "Quit the application", "Session"),
]

COMMAND_NAMES = frozenset(f"/{c.name}" for c in COMMANDS)
