from io import StringIO
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from mini.core import Mini
from mini.display import Display


def _fake_stream_response(stream, on_usage=None, force_hide_reasoning=False):
    message, actions, usage = None, [], None
    for event in stream:
        if event["type"] == "done":
            message = event["message"]
            actions = event["actions"]
            usage = event.get("usage")
            if on_usage and usage:
                on_usage(usage)
    return message, actions, usage


class TestMini:
    def _make_mini(self, max_context_tokens=96000, keep_turns=2):
        model = MagicMock()
        env = MagicMock()
        env.execute.return_value = {"output": "ok", "returncode": 0}
        display = Display(console=Console(file=StringIO()))
        display.stream_response = _fake_stream_response
        mini = Mini(
            model,
            env,
            display,
            max_context_tokens=max_context_tokens,
            keep_turns=keep_turns,
        )
        return mini, model, env

    def test_run_turn_without_actions(self):
        mini, model, env = self._make_mini()
        mini.messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ]
        model.stream.return_value = [
            {"type": "content", "delta": "Hello!"},
            {
                "type": "done",
                "message": {"role": "assistant", "content": "Hello!"},
                "actions": [],
                "usage": None,
            },
        ]

        mini._run_turn()

        assert len(mini.messages) == 3
        assert mini.messages[-1]["role"] == "assistant"
        assert mini.messages[-1]["content"] == "Hello!"
        env.execute.assert_not_called()

    def test_run_turn_with_tool_call(self):
        mini, model, env = self._make_mini()
        mini.messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "run ls"},
        ]
        tool_call = {
            "id": "call_1",
            "type": "function",
            "function": {"name": "bash", "arguments": '{"command": "ls"}'},
        }
        message = {"role": "assistant", "content": "", "tool_calls": [tool_call]}
        model.format_observation_messages.return_value = [
            {"role": "tool", "content": "ok", "tool_call_id": "call_1"}
        ]

        # First stream returns tool call; second returns final text answer.
        model.stream.side_effect = [
            [
                {
                    "type": "done",
                    "message": message,
                    "actions": [
                        {
                            "tool_name": "bash",
                            "arguments": {"command": "ls"},
                            "tool_call_id": "call_1",
                        }
                    ],
                    "usage": None,
                },
            ],
            [
                {"type": "content", "delta": "Done"},
                {
                    "type": "done",
                    "message": {"role": "assistant", "content": "Done"},
                    "actions": [],
                    "usage": None,
                },
            ],
        ]

        mini._run_turn()

        assert env.execute.call_count == 1
        # system + user + assistant(tool) + tool result + assistant(final)
        assert len(mini.messages) == 5
        model.format_observation_messages.assert_called_once()

    def test_run_turn_tracks_cost(self):
        mini, model, env = self._make_mini()
        mini.messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ]
        model.stream.return_value = [
            {"type": "content", "delta": "Hi"},
            {
                "type": "done",
                "message": {"role": "assistant", "content": "Hi"},
                "actions": [],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        ]
        mini.tracker.input_cost_per_token = 0.001
        mini.tracker.output_cost_per_token = 0.002

        mini._run_turn()

        assert mini.tracker.total_uncached_tokens == 10
        assert mini.tracker.total_completion_tokens == 5
        assert mini.tracker.cost == pytest.approx(0.02)

    def test_clear_command_resets_state(self):
        mini, model, env = self._make_mini()
        mini.messages = [{"role": "user", "content": "hi"}]
        mini.n_tool_calls = 3
        mini.tracker.update({"prompt_tokens": 10, "completion_tokens": 5})
        mini.context.log_event({"type": "skip", "total_tokens": 10})

        mini._cmd_clear("/clear")

        assert mini.messages == []
        assert mini.n_tool_calls == 0
        assert mini.tracker.cost == 0
        assert mini.context.get_events() == []
        assert mini.context.interior_mode is False

    def test_help_command_lists_commands(self):
        mini, model, env = self._make_mini()
        mini._cmd_help("/help")
        output = mini.display.console.file.getvalue()
        assert "/cost" in output
        assert "/help" in output

    def test_cost_command_shows_report(self):
        mini, model, env = self._make_mini()
        mini.tracker.input_cost_per_token = 0.001
        mini.tracker.update({"prompt_tokens": 100, "completion_tokens": 50})
        mini._cmd_cost("/cost")
        output = mini.display.console.file.getvalue()
        assert "Session cost" in output
        assert "$0.100000" in output

    def test_show_interior_toggle(self):
        mini, model, env = self._make_mini()
        mini.messages = [{"role": "user", "content": "hi"}]
        mini._cmd_show_interior("/show-interior")
        assert mini.context.interior_mode is True
        output = mini.display.console.file.getvalue()
        assert "Interior mode: ON" in output

        mini._cmd_show_interior("/show-interior off")
        assert mini.context.interior_mode is False

    def test_start_exits_on_exit_text(self):
        mini, model, env = self._make_mini()
        with patch.object(mini.display, "prompt", side_effect=["exit"]):
            mini.start()
        # start() initializes messages with the system prompt before exiting.
        assert len(mini.messages) == 1
        assert mini.messages[0]["role"] == "system"

    def test_start_exits_on_slash_exit(self):
        mini, model, env = self._make_mini()
        with patch.object(mini.display, "prompt", side_effect=["/exit"]):
            mini.start()
        assert len(mini.messages) == 1
        assert mini.messages[0]["role"] == "system"
