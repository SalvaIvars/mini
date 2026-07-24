import json
from unittest.mock import MagicMock

from mini.context import ContextWindow


def make_tool_call(id_: str, command: str) -> dict:
    return {
        "id": id_,
        "type": "function",
        "function": {"name": "bash", "arguments": json.dumps({"command": command})},
    }


def make_tool_result(content: str, tool_call_id: str = "call_1") -> dict:
    return {"role": "tool", "content": content, "tool_call_id": tool_call_id}


SMALL_OUTPUT = "<returncode>0</returncode>\n<output>\nhello\nworld\n</output>"
LARGE_OUTPUT = (
    "<returncode>0</returncode>\n<output>\n"
    + "\n".join(f"line_{i}_" + "x" * 50 for i in range(100))
    + "\n</output>"
)
ERROR_OUTPUT = "<returncode>1</returncode>\n<output>\ncommand not found\n</output>"


# ============================================================
# count_tokens
# ============================================================


class TestCountTokens:
    def test_empty_list(self):
        assert ContextWindow.count_tokens([]) == 0

    def test_simple_messages(self):
        msgs = [
            {"role": "user", "content": "hello world"},
            {"role": "assistant", "content": "hi there"},
        ]
        count = ContextWindow.count_tokens(msgs)
        assert count > 0
        # same content should give same count
        assert ContextWindow.count_tokens(msgs) == count

    def test_with_tool_calls(self):
        msgs = [
            {"role": "assistant", "content": "", "tool_calls": [make_tool_call("c1", "ls -la")]},
            make_tool_result("output here"),
        ]
        count = ContextWindow.count_tokens(msgs)
        assert count > 0


# ============================================================
# strip_reasoning
# ============================================================


class TestStripReasoning:
    def test_removes_reasoning_content(self):
        msgs = [
            {"role": "assistant", "content": "hello", "reasoning_content": "thinking..."},
            {"role": "user", "content": "hi"},
        ]
        ContextWindow.strip_reasoning(msgs)
        assert "reasoning_content" not in msgs[0]
        assert msgs[0]["content"] == "hello"

    def test_no_reasoning_is_noop(self):
        msgs = [{"role": "user", "content": "hi"}]
        before = dict(msgs[0])
        ContextWindow.strip_reasoning(msgs)
        assert msgs[0] == before


# ============================================================
# group_into_turns
# ============================================================


class TestGroupIntoTurns:
    def test_single_user_message(self):
        msgs = [{"role": "user", "content": "hi"}]
        turns = ContextWindow.group_into_turns(msgs)
        assert len(turns) == 1
        assert len(turns[0]) == 1

    def test_user_assistant_pair(self):
        msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        turns = ContextWindow.group_into_turns(msgs)
        assert len(turns) == 1
        assert [m["role"] for m in turns[0]] == ["user", "assistant"]

    def test_tool_use_sequence(self):
        msgs = [
            {"role": "user", "content": "find the bug"},
            {"role": "assistant", "content": "", "tool_calls": [make_tool_call("c1", "grep error")]},
            make_tool_result("error in main.py"),
            {"role": "assistant", "content": "Found it in main.py"},
            {"role": "user", "content": "fix it"},
            {"role": "assistant", "content": "Fixed"},
        ]
        turns = ContextWindow.group_into_turns(msgs)
        assert len(turns) == 2
        assert [m["role"] for m in turns[0]] == ["user", "assistant", "tool", "assistant"]
        assert [m["role"] for m in turns[1]] == ["user", "assistant"]

    def test_empty_list(self):
        assert ContextWindow.group_into_turns([]) == []

    def test_only_assistant_messages(self):
        msgs = [{"role": "assistant", "content": "hi"}]
        turns = ContextWindow.group_into_turns(msgs)
        assert len(turns) == 1
        assert turns[0] == msgs


# ============================================================
# clear_tool_result
# ============================================================


class TestClearToolResult:
    def test_non_tool_message_unchanged(self):
        msg = {"role": "user", "content": "hello"}
        result = ContextWindow.clear_tool_result(msg)
        assert result is msg

    def test_small_output_unchanged(self):
        msg = make_tool_result(SMALL_OUTPUT)
        result = ContextWindow.clear_tool_result(msg)
        assert result is msg

    def test_error_output_unchanged(self):
        msg = make_tool_result(ERROR_OUTPUT)
        result = ContextWindow.clear_tool_result(msg)
        assert result is msg

    def test_large_output_truncated(self):
        msg = make_tool_result(LARGE_OUTPUT)
        result = ContextWindow.clear_tool_result(msg)
        assert result is not msg
        original_lines = len(msg["content"].splitlines())
        new_lines = len(result["content"].splitlines())
        assert new_lines < original_lines
        assert "lines /" in result["content"] or "bytes omitted" in result["content"]
        assert result["tool_call_id"] == "call_1"

    def test_custom_max_lines(self):
        msg = make_tool_result(LARGE_OUTPUT)
        result = ContextWindow.clear_tool_result(msg, max_lines=3)
        # head 1 + omitted + tail 1 = 3 lines (with very small max_lines)
        # Actually: kept = lines[:3] + omitted + lines[-3:] = 7 min
        assert result is not msg


# ============================================================
# prepare (integration)
# ============================================================


class TestPrepareMessages:
    def _make_context(self, max_context_tokens=96000, keep_turns=2):
        model = MagicMock()
        model.query.return_value = {"content": "- Fixed login bug\n- Added tests"}
        return ContextWindow(
            model,
            max_context_tokens=max_context_tokens,
            keep_turns=keep_turns,
        )

    def test_under_budget_returns_all(self):
        context = self._make_context(max_context_tokens=999999)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        result = context.prepare(messages)
        assert result is messages

    def test_tool_result_clearing_phase(self):
        # Budget high enough that clearing alone is sufficient
        context = self._make_context(max_context_tokens=150, keep_turns=1)
        big_content = "<returncode>0</returncode>\n<output>\n" + "\n".join(
            f"line_{i}_" + "x" * 200 for i in range(100)
        ) + "\n</output>"
        big = make_tool_result(big_content)
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "run command"},
            {"role": "assistant", "content": "", "tool_calls": [make_tool_call("c1", "big command")]},
            big,
            {"role": "assistant", "content": "done"},
            {"role": "user", "content": "next"},
            {"role": "assistant", "content": "ok"},
        ]
        result = context.prepare(messages)
        assert any("lines /" in m.get("content", "") or "bytes omitted" in m.get("content", "") for m in result)
        # Should NOT contain Session History (clearing was enough)
        assert not any("Session History" in m.get("content", "") for m in result)

    def test_over_budget_triggers_summarization(self):
        context = self._make_context(max_context_tokens=80, keep_turns=1)
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "do thing 1 that is very long and detailed " * 50},
            {"role": "assistant",
             "content": "ok let me do that with a lot of text " * 30,
             "tool_calls": [make_tool_call("c1", "cmd 1 that is very long " * 20)]},
            make_tool_result("output 1 " * 100),
            {"role": "assistant", "content": "done with that task " * 30},
            {"role": "user", "content": "do thing 2 a different thing here"},
            {"role": "assistant", "content": "ok here we go"},
        ]
        result = context.prepare(messages)
        contents = [m.get("content", "") for m in result]
        assert any("Session History" in c for c in contents), f"No summary found in: {contents}"
        # Recent turn should be preserved
        assert any("do thing 2" in m.get("content", "") for m in result)

    def test_summarization_fails_falls_back(self):
        context = self._make_context(max_context_tokens=100, keep_turns=1)
        context.model.query.return_value = {}  # empty response
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "do thing 1"},
            {"role": "assistant", "content": "ok", "tool_calls": [make_tool_call("c1", "cmd 1")]},
            make_tool_result("output 1"),
            {"role": "assistant", "content": "done"},
            {"role": "user", "content": "do thing 2"},
            {"role": "assistant", "content": "ok"},
        ]
        result = context.prepare(messages)
        # Should not contain Session History (summary was empty)
        contents = [m.get("content", "") for m in result]
        assert not any("Session History" in c for c in contents)

    def test_recent_alone_over_budget_aggressive(self):
        context = self._make_context(max_context_tokens=30, keep_turns=1)
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "do thing 1"},
            {"role": "assistant", "content": "ok", "tool_calls": [make_tool_call("c1", "cmd 1")]},
            make_tool_result("output " + "x" * 1000),
            {"role": "assistant", "content": "done"},
            {"role": "user", "content": "do thing 2"},
            {"role": "assistant", "content": "ok"},
        ]
        result = context.prepare(messages)
        # Should survive without crash and contain system + at least recent user
        contents = [m.get("content", "") for m in result]
        assert any(m["role"] == "system" for m in result)

    def test_no_reasoning_in_output(self):
        context = self._make_context(max_context_tokens=999999)
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi", "reasoning_content": "thinking..."},
        ]
        # reasoning should be stripped before returning
        result = context.prepare(messages)
        for m in result:
            assert "reasoning_content" not in m


# ============================================================
# summarize_turns
# ============================================================


class TestSummarizeTurns:
    def test_calls_model_stream(self):
        model = MagicMock()
        model.stream.return_value = iter([
            {"type": "content", "delta": "- Fixed the bug"},
            {"type": "done", "message": {"role": "assistant", "content": "- Fixed the bug"}, "actions": [], "usage": None},
        ])
        context = ContextWindow(model, max_context_tokens=100, keep_turns=1)
        turns = [
            [
                {"role": "user", "content": "find the bug"},
                {"role": "assistant", "content": "", "tool_calls": [make_tool_call("c1", "grep error")]},
                make_tool_result("<returncode>0</returncode>\n<output>\nerror found\n</output>"),
                {"role": "assistant", "content": "Found it"},
            ]
        ]
        input_lines, stream = context._summarize_turns(turns)
        assert input_lines == ['User: find the bug', '  → bash: grep error', '  ↷ (ok) error found...', '  → Found it']
        assert model.stream.called

    def test_with_tool_calls_in_summary(self):
        model = MagicMock()
        model.stream.return_value = iter([
            {"type": "content", "delta": "- Ran grep"},
            {"type": "done", "message": {"role": "assistant", "content": "- Ran grep"}, "actions": [], "usage": None},
        ])
        context = ContextWindow(model, max_context_tokens=100, keep_turns=1)
        turns = [
            [
                {"role": "user", "content": "search codebase"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [make_tool_call("c1", "grep -r 'TODO' src/")],
                },
                make_tool_result("TODO in main.py"),
                {"role": "assistant", "content": "Found TODOs"},
            ]
        ]
        input_lines, stream = context._summarize_turns(turns)
        assert input_lines
        prompt_sent = model.stream.call_args[0][0][0]["content"]
        assert "grep" in prompt_sent

    def test_tool_result_with_large_output(self):
        model = MagicMock()
        model.stream.return_value = iter([
            {"type": "content", "delta": "- summary"},
            {"type": "done", "message": {"role": "assistant", "content": "- summary"}, "actions": [], "usage": None},
        ])
        context = ContextWindow(model, max_context_tokens=100, keep_turns=1)
        large_output = "A" * 150
        turns = [
            [
                {"role": "user", "content": "test"},
                {"role": "assistant", "content": "", "tool_calls": [make_tool_call("c1", "cmd")]},
                make_tool_result(f"<returncode>0</returncode>\n<output>\n{large_output}\n</output>"),
            ]
        ]
        input_lines, stream = context._summarize_turns(turns)
        assert len(input_lines) == 3
        assert input_lines[2].startswith('  ↷ (ok) ')
        assert input_lines[2].endswith('...')
        preview_part = input_lines[2][len('  ↷ (ok) '):-len('...')]
        assert len(preview_part) == 100

    def test_tool_result_with_newlines(self):
        model = MagicMock()
        model.stream.return_value = iter([
            {"type": "content", "delta": "- summary"},
            {"type": "done", "message": {"role": "assistant", "content": "- summary"}, "actions": [], "usage": None},
        ])
        context = ContextWindow(model, max_context_tokens=100, keep_turns=1)
        output_with_newlines = "line1\nline2\nline3"
        turns = [
            [
                {"role": "user", "content": "test"},
                {"role": "assistant", "content": "", "tool_calls": [make_tool_call("c1", "cmd")]},
                make_tool_result(f"<returncode>0</returncode>\n<output>\n{output_with_newlines}\n</output>"),
            ]
        ]
        input_lines, stream = context._summarize_turns(turns)
        assert '\n' not in input_lines[2]
        assert 'line1 line2 line3' in input_lines[2]

    def test_tool_result_without_output_tags(self):
        model = MagicMock()
        model.stream.return_value = iter([
            {"type": "content", "delta": "- summary"},
            {"type": "done", "message": {"role": "assistant", "content": "- summary"}, "actions": [], "usage": None},
        ])
        context = ContextWindow(model, max_context_tokens=100, keep_turns=1)
        turns = [
            [
                {"role": "user", "content": "test"},
                {"role": "assistant", "content": "", "tool_calls": [make_tool_call("c1", "cmd")]},
                {"role": "tool", "content": "plain text without tags"},
            ]
        ]
        input_lines, stream = context._summarize_turns(turns)
        assert input_lines[2] == '  ↷ (ok)'


# ============================================================
# summarize
# ============================================================


class TestSummarize:
    def test_summarize_clears_recent_tool_outputs(self):
        model = MagicMock()
        model.stream.return_value = iter([
            {"type": "content", "delta": "- summary"},
            {"type": "done", "message": {"role": "assistant", "content": "- summary"}, "actions": [], "usage": None},
        ])
        context = ContextWindow(model, max_context_tokens=100, keep_turns=1)
        
        large_output = "line\n" * 100
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "old request"},
            {"role": "assistant", "content": "", "tool_calls": [make_tool_call("c1", "old_cmd")]},
            make_tool_result("<returncode>0</returncode>\n<output>\nold output\n</output>"),
            {"role": "user", "content": "recent request"},
            {"role": "assistant", "content": "", "tool_calls": [make_tool_call("c2", "recent_cmd")]},
            make_tool_result(f"<returncode>0</returncode>\n<output>\n{large_output}\n</output>"),
        ]
        
        info = context.summarize(messages)
        assert info is not None
        
        recent = info["recent"]
        tool_result = next(m for m in recent if m.get("role") == "tool")
        
        lines = tool_result["content"].split("\n")
        assert len(lines) <= 15

    def test_summarize_reduces_context_significantly(self):
        model = MagicMock()
        model.stream.return_value = iter([
            {"type": "content", "delta": "- summary"},
            {"type": "done", "message": {"role": "assistant", "content": "- summary"}, "actions": [], "usage": None},
        ])
        context = ContextWindow(model, max_context_tokens=100000, keep_turns=1)
        
        large_output = "line\n" * 10000
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "old request 1"},
            {"role": "assistant", "content": "old response 1"},
            {"role": "user", "content": "old request 2"},
            {"role": "assistant", "content": "", "tool_calls": [make_tool_call("c1", "heavy_cmd")]},
            make_tool_result(f"<returncode>0</returncode>\n<output>\n{large_output}\n</output>"),
        ]
        
        tokens_before = context.count_tokens(messages)
        info = context.summarize(messages)
        assert info is not None
        
        messages_after = info["system"] + [{"role": "system", "content": "summary"}] + info["recent"]
        tokens_after = context.count_tokens(messages_after)
        
        reduction = (tokens_before - tokens_after) / tokens_before
        assert reduction >= 0.5

    def test_summarize_preserves_small_tool_outputs(self):
        model = MagicMock()
        model.stream.return_value = iter([
            {"type": "content", "delta": "- summary"},
            {"type": "done", "message": {"role": "assistant", "content": "- summary"}, "actions": [], "usage": None},
        ])
        context = ContextWindow(model, max_context_tokens=100, keep_turns=1)
        
        small_output = "small output"
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "old request"},
            {"role": "assistant", "content": "old response"},
            {"role": "user", "content": "recent request"},
            {"role": "assistant", "content": "", "tool_calls": [make_tool_call("c1", "cmd")]},
            make_tool_result(f"<returncode>0</returncode>\n<output>\n{small_output}\n</output>"),
        ]
        
        info = context.summarize(messages)
        assert info is not None
        
        recent = info["recent"]
        tool_result = next(m for m in recent if m.get("role") == "tool")
        
        assert small_output in tool_result["content"]

    def test_summarize_with_multiple_recent_turns(self):
        model = MagicMock()
        model.stream.return_value = iter([
            {"type": "content", "delta": "- summary"},
            {"type": "done", "message": {"role": "assistant", "content": "- summary"}, "actions": [], "usage": None},
        ])
        context = ContextWindow(model, max_context_tokens=100, keep_turns=2)
        
        large_output = "line\n" * 100
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "old request"},
            {"role": "assistant", "content": "old response"},
            {"role": "user", "content": "recent request 1"},
            {"role": "assistant", "content": "", "tool_calls": [make_tool_call("c1", "cmd1")]},
            make_tool_result(f"<returncode>0</returncode>\n<output>\n{large_output}\n</output>"),
            {"role": "user", "content": "recent request 2"},
            {"role": "assistant", "content": "", "tool_calls": [make_tool_call("c2", "cmd2")]},
            make_tool_result(f"<returncode>0</returncode>\n<output>\n{large_output}\n</output>"),
        ]
        
        info = context.summarize(messages)
        assert info is not None
        
        recent = info["recent"]
        tool_results = [m for m in recent if m.get("role") == "tool"]
        
        for tool_result in tool_results:
            lines = tool_result["content"].split("\n")
            assert len(lines) <= 15
