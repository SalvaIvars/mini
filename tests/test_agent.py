from io import StringIO
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from mini_agent.agent import DefaultAgent


class TestDefaultAgent:
    def _make_agent(self, max_context_tokens=96000, keep_turns=2):
        model = MagicMock()
        env = MagicMock()
        env.execute.return_value = {"output": "ok", "returncode": 0}
        agent = DefaultAgent(
            model,
            env,
            max_context_tokens=max_context_tokens,
            keep_turns=keep_turns,
        )
        agent.display.console = Console(file=StringIO())
        return agent, model, env

    def test_run_turn_without_actions(self):
        agent, model, env = self._make_agent()
        agent.messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
        model.stream.return_value = [
            {"type": "content", "delta": "Hello!"},
            {"type": "done", "message": {"role": "assistant", "content": "Hello!"}, "actions": [], "usage": None},
        ]

        agent._run_turn()

        assert len(agent.messages) == 3
        assert agent.messages[-1]["role"] == "assistant"
        assert agent.messages[-1]["content"] == "Hello!"
        env.execute.assert_not_called()

    def test_run_turn_with_tool_call(self):
        agent, model, env = self._make_agent()
        agent.messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "run ls"}]
        tool_call = {
            "id": "call_1",
            "type": "function",
            "function": {"name": "bash", "arguments": '{"command": "ls"}'},
        }
        message = {"role": "assistant", "content": "", "tool_calls": [tool_call]}
        model.format_observation_messages.return_value = [{"role": "tool", "content": "ok", "tool_call_id": "call_1"}]

        # First stream returns tool call; second returns final text answer.
        model.stream.side_effect = [
            [
                {"type": "done", "message": message, "actions": [{"command": "ls", "tool_call_id": "call_1"}], "usage": None},
            ],
            [
                {"type": "content", "delta": "Done"},
                {"type": "done", "message": {"role": "assistant", "content": "Done"}, "actions": [], "usage": None},
            ],
        ]

        agent._run_turn()

        assert env.execute.call_count == 1
        # system + user + assistant(tool) + tool result + assistant(final)
        assert len(agent.messages) == 5
        model.format_observation_messages.assert_called_once()

    def test_run_turn_tracks_cost(self):
        agent, model, env = self._make_agent()
        agent.messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
        model.stream.return_value = [
            {"type": "content", "delta": "Hi"},
            {
                "type": "done",
                "message": {"role": "assistant", "content": "Hi"},
                "actions": [],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        ]
        agent.tracker.input_cost_per_token = 0.001
        agent.tracker.output_cost_per_token = 0.002

        agent._run_turn()

        assert agent.tracker.total_uncached_tokens == 10
        assert agent.tracker.total_completion_tokens == 5
        assert agent.tracker.cost == pytest.approx(0.02)

    def test_clear_command_resets_state(self):
        agent, model, env = self._make_agent()
        agent.messages = [{"role": "user", "content": "hi"}]
        agent.n_tool_calls = 3
        agent.tracker.update({"prompt_tokens": 10, "completion_tokens": 5})
        agent.context.log_event({"type": "skip", "total_tokens": 10})

        agent._cmd_clear("/clear")

        assert agent.messages == []
        assert agent.n_tool_calls == 0
        assert agent.tracker.cost == 0
        assert agent.context.get_events() == []
        assert agent.context.interior_mode is False

    def test_help_command_lists_commands(self):
        agent, model, env = self._make_agent()
        agent._cmd_help("/help")
        output = agent.display.console.file.getvalue()
        assert "/cost" in output
        assert "/help" in output

    def test_cost_command_shows_report(self):
        agent, model, env = self._make_agent()
        agent.tracker.input_cost_per_token = 0.001
        agent.tracker.update({"prompt_tokens": 100, "completion_tokens": 50})
        agent._cmd_cost("/cost")
        output = agent.display.console.file.getvalue()
        assert "Session cost" in output
        assert "$0.100000" in output

    def test_show_interior_toggle(self):
        agent, model, env = self._make_agent()
        agent.messages = [{"role": "user", "content": "hi"}]
        agent._cmd_show_interior("/show-interior")
        assert agent.context.interior_mode is True
        output = agent.display.console.file.getvalue()
        assert "Interior mode: ON" in output

        agent._cmd_show_interior("/show-interior off")
        assert agent.context.interior_mode is False

    def test_start_exits_on_exit_text(self):
        agent, model, env = self._make_agent()
        with patch.object(agent.display, "prompt", side_effect=["exit"]):
            agent.start()
        # start() initializes messages with the system prompt before exiting.
        assert len(agent.messages) == 1
        assert agent.messages[0]["role"] == "system"

    def test_start_exits_on_slash_exit(self):
        agent, model, env = self._make_agent()
        with patch.object(agent.display, "prompt", side_effect=["/exit"]):
            agent.start()
        assert len(agent.messages) == 1
        assert agent.messages[0]["role"] == "system"

    def test_command_registry_matches_legacy_commands(self):
        from mini_agent.commands import COMMANDS
        agent, model, env = self._make_agent()
        registered_names = {c.name for c in agent.commands.commands}
        legacy_names = {c.name for c in COMMANDS}
        assert registered_names == legacy_names
