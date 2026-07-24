from unittest.mock import MagicMock

import pytest

from mini.commands import Command, CommandRegistry, SlashCommandCompleter


class TestCommandRegistry:
    def test_register_and_handle(self):
        registry = CommandRegistry()
        handler = MagicMock()
        registry.register(Command("test", "A test command"), handler)

        continuing = registry.handle("/test arg1 arg2")

        assert continuing is True
        handler.assert_called_once_with("/test arg1 arg2")

    def test_handle_unknown_command(self, capsys):
        registry = CommandRegistry()

        continuing = registry.handle("/unknown")

        assert continuing is True
        captured = capsys.readouterr()
        assert "Unknown command" in captured.out

    def test_exit_command_returns_false(self):
        registry = CommandRegistry()
        registry.register(Command("exit", "Quit"), lambda raw: None)

        continuing = registry.handle("/exit")

        assert continuing is False

    def test_get_command_parses_name_and_args(self):
        registry = CommandRegistry()
        assert registry.get_command("/foo bar") == "/foo"
        assert registry.get_command("/foo") == "/foo"

    def test_commands_property_returns_registered(self):
        registry = CommandRegistry()
        registry.register(Command("a", "first"), lambda raw: None)
        registry.register(Command("b", "second"), lambda raw: None)
        names = [c.name for c in registry.commands]
        assert names == ["a", "b"]


class TestSlashCommandCompleter:
    def test_completions_with_registry(self):
        registry = CommandRegistry()
        registry.register(Command("help", "Show help"), lambda raw: None)
        registry.register(Command("clear", "Clear memory"), lambda raw: None)
        completer = SlashCommandCompleter(registry)

        document = MagicMock()
        document.text_before_cursor = "/"
        completions = list(completer.get_completions(document, None))

        assert len(completions) == 2
        assert {c.display_text for c in completions} == {"/help", "/clear"}

    def test_completions_filter_by_query(self):
        registry = CommandRegistry()
        registry.register(Command("help", "Show help"), lambda raw: None)
        registry.register(Command("clear", "Clear memory"), lambda raw: None)
        completer = SlashCommandCompleter(registry)

        document = MagicMock()
        document.text_before_cursor = "/cl"
        completions = list(completer.get_completions(document, None))

        assert len(completions) == 1
        assert completions[0].display_text == "/clear"

    def test_no_completions_outside_slash(self):
        completer = SlashCommandCompleter(CommandRegistry())
        document = MagicMock()
        document.text_before_cursor = "hello"
        completions = list(completer.get_completions(document, None))
        assert completions == []
