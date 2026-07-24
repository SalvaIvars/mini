import json
from typing import Iterator

import tiktoken


class ContextWindow:
    """Manages conversation context compression and token accounting."""

    def __init__(self, model, max_context_tokens: int = 96000, keep_turns: int = 2):
        self.model = model
        self.max_context_tokens = max_context_tokens
        self.keep_turns = keep_turns
        self._interior_mode: bool = False
        self._compression_events: list[dict] = []

    def reset(self):
        self._interior_mode = False
        self._compression_events.clear()

    @property
    def interior_mode(self) -> bool:
        return self._interior_mode

    def set_interior_mode(self, value: bool):
        self._interior_mode = value

    @staticmethod
    def count_tokens(messages: list[dict]) -> int:
        enc = tiktoken.get_encoding("cl100k_base")
        text = ""
        for m in messages:
            text += m.get("role", "") + " "
            content = m.get("content")
            if content:
                text += content + " "
            if m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    fn = tc.get("function", {})
                    text += fn.get("name", "") + " "
                    text += fn.get("arguments", "") + " "
        return len(enc.encode(text, disallowed_special=()))

    @staticmethod
    def strip_reasoning(messages: list[dict]):
        for m in messages:
            m.pop("reasoning_content", None)

    @staticmethod
    def group_into_turns(messages: list[dict]) -> list[list[dict]]:
        turns = []
        current = []
        for m in messages:
            if m["role"] == "user":
                if current:
                    turns.append(current)
                current = [m]
            else:
                current.append(m)
        if current:
            turns.append(current)
        return turns

    @staticmethod
    def clear_tool_result(msg: dict, max_lines: int = 15, max_preview: int = 500) -> dict:
        if msg.get("role") != "tool":
            return msg
        content = msg.get("content", "")
        is_error = (
            "<returncode>" in content
            and "0" not in content.split("<returncode>")[1].split("</returncode>")[0].strip()
        )
        if is_error or len(content) < max_preview:
            return msg
        lines = content.split("\n")
        if len(lines) <= max_lines:
            return msg
        kept = lines[:3] + [f"... ({len(lines)-6} lines / {len(content)} bytes omitted)"] + lines[-3:]
        out = dict(msg)
        out["content"] = "\n".join(kept)
        return out

    def _summarize_turns(self, turns: list[list[dict]]) -> tuple[list[str], Iterator]:
        lines = []
        for turn in turns:
            user = next((m["content"] for m in turn if m["role"] == "user"), "")
            lines.append(f"User: {user[:300]}")
            for m in turn:
                if m["role"] == "assistant":
                    if tc := m.get("tool_calls"):
                        for t in tc:
                            try:
                                args = json.loads(t["function"]["arguments"])
                                lines.append(f"  \u2192 bash: {args.get('command', '')[:200]}")
                            except json.JSONDecodeError:
                                pass
                    elif content := m.get("content"):
                        lines.append(f"  \u2192 {content[:200]}")
                elif m["role"] == "tool":
                    is_err = (
                        "<returncode>" in (m.get("content") or "")
                        and "0" not in m["content"].split("<returncode>")[1].split("</returncode>")[0].strip()
                    )
                    content = m.get("content", "")
                    if "<output>" in content and "</output>" in content:
                        output = content.split("<output>")[1].split("</output>")[0]
                        preview = output.strip().replace("\n", " ")[:100]
                        lines.append(f"  \u21b7 ({'error' if is_err else 'ok'}) {preview}...")
                    else:
                        lines.append(f"  \u21b7 ({'error' if is_err else 'ok'})")
        prompt = (
            "Compress this conversation into a concise memory summary.\n"
            "Keep: user requests, commands executed, what was found, decisions made, files modified.\n"
            "Discard: exact command output, stack traces, intermediate back-and-forth.\n"
            "Output 3-8 bullet points. Use technical language.\n\n"
            f"{chr(10).join(lines)}\n\n"
            "Summary:"
        )
        stream = self.model.stream([{"role": "user", "content": prompt}])
        return lines, stream

    def log_event(self, event: dict):
        self._compression_events.append(event)

    def get_events(self) -> list[dict]:
        return list(self._compression_events)

    def last_event(self) -> dict | None:
        return self._compression_events[-1] if self._compression_events else None

    def prepare(self, messages: list[dict]) -> list[dict]:
        self.strip_reasoning(messages)

        total = self.count_tokens(messages)
        if total <= self.max_context_tokens * 0.75:
            self.log_event({"type": "skip", "total_tokens": total})
            return messages

        if messages and messages[0]["role"] == "system":
            system = [messages[0]]
            rest = messages[1:]
        else:
            system = []
            rest = messages

        turns = self.group_into_turns(rest)
        keep_n = min(self.keep_turns, len(turns))
        kept = turns[-keep_n:]
        old = turns[:-keep_n]

        if not old:
            self.log_event({"type": "skip", "total_tokens": total})
            return messages

        old_cleared = [self.clear_tool_result(m) for turn in old for m in turn]
        recent = [m for turn in kept for m in turn]
        candidate = system + old_cleared + recent

        cleared_count = 0
        orig_lines_total = 0
        new_lines_total = 0
        for orig, cleared in zip(
            [m for turn in old for m in turn],
            old_cleared,
        ):
            if orig.get("role") == "tool" and cleared is not orig:
                cleared_count += 1
                orig_lines_total += len(orig["content"].split("\n"))
                new_lines_total += len(cleared["content"].split("\n"))

        if self.count_tokens(candidate) <= self.max_context_tokens:
            self.log_event({
                "type": "clear",
                "total_tokens": self.count_tokens(candidate),
                "count": cleared_count,
                "original_lines": orig_lines_total,
                "new_lines": new_lines_total,
            })
            return candidate

        orig_tok_before = self.count_tokens([m for turn in old for m in turn])
        _input_lines, summary = self._summarize_turns(old)
        if summary:
            summary_msg = {"role": "system", "content": f"## Session History\n{summary}"}
            summary_tok = self.count_tokens([summary_msg])
            candidate = system + [summary_msg] + recent
            self.log_event({
                "type": "summarize",
                "total_tokens": self.count_tokens(candidate),
                "old_turns": len(old),
                "original_tokens": orig_tok_before,
                "summary_tokens": summary_tok,
                "summary_preview": summary[:80],
            })
        else:
            candidate = system + recent

        if self.count_tokens(candidate) > self.max_context_tokens:
            had_summary = bool(summary)
            reason = "dropped summary" if had_summary else "clearing insufficient, dropping old turns"
            self.log_event({
                "type": "aggressive",
                "total_tokens": self.count_tokens(system + recent),
                "reason": reason,
            })
            candidate = system + recent
            if self.count_tokens(candidate) > self.max_context_tokens:
                recent_cleared = [self.clear_tool_result(m, max_lines=8, max_preview=300) for m in recent]
                self.log_event({
                    "type": "aggressive",
                    "total_tokens": self.count_tokens(system + recent_cleared),
                    "reason": "clearing recent too",
                })
                candidate = system + recent_cleared

        return candidate

    def summarize(self, messages: list[dict]) -> dict | None:
        self.strip_reasoning(messages)

        if messages and messages[0]["role"] == "system":
            system = [messages[0]]
            rest = messages[1:]
        else:
            system = []
            rest = messages

        turns = self.group_into_turns(rest)

        if len(turns) <= 1:
            self.log_event({"type": "skip_summarize", "total_tokens": self.count_tokens(messages)})
            return None

        kept = turns[-1:]
        old = turns[:-1]

        old_turns_cleared = [[self.clear_tool_result(m) for m in turn] for turn in old]
        recent = [self.clear_tool_result(m) for turn in kept for m in turn]

        input_lines, stream = self._summarize_turns(old_turns_cleared)

        return {
            "input_lines": input_lines,
            "stream": stream,
            "old_turns": len(old),
            "original_tokens": self.count_tokens([m for turn in old for m in turn]),
            "total_tokens": self.count_tokens(messages),
            "recent": recent,
            "system": system,
        }
