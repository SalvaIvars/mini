from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style
from rich.console import Console, Group
from rich.text import Text
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich import box

_MENU_STYLE = Style.from_dict(
    {
        "completion-menu": "bg:#1e1e1e #d4d4d4",
        "completion-menu.completion": "bg:#252526 #d4d4d4",
        "completion-menu.completion.current": "bg:#094771 #ffffff",
        "completion-menu.meta": "bg:#1e1e1e #808080",
    }
)


class Display:
    """Handles all terminal output: streaming, tool output, and status."""

    def __init__(self, console: Console | None = None):
        self.console = console or Console()
        self.session = PromptSession(style=_MENU_STYLE)
        self.show_reasoning = True

    def prompt(self, completer):
        return self.session.prompt(
            "\n> ",
            completer=completer,
            complete_while_typing=True,
        )

    def welcome(self, model_name: str):
        self.console.print(f"Using model: {model_name}", style="bold")
        self.console.print("Type 'exit' or press Ctrl+C to quit.")

    def goodbye(self):
        self.console.print("Goodbye.", style="dim")

    def unknown_command(self, cmd: str):
        self.console.print(
            f"Unknown command: {cmd}. Type /help for available commands.",
            style="yellow",
        )

    def print_error(self, message: str):
        self.console.print(Text(message, style="bold red"))

    def stream_response(self, stream, *, on_usage=None, force_hide_reasoning=False):
        message = None
        actions = []
        usage = None
        reasoning_ended = False

        for event in stream:
            if event["type"] == "reasoning":
                if self.show_reasoning and not force_hide_reasoning:
                    self.console.print(event["delta"], style="dim", end="")
            elif event["type"] == "content":
                if not reasoning_ended:
                    reasoning_ended = True
                    self.console.print()
                    self.console.print()
                self.console.print(event["delta"], end="")
            elif event["type"] == "done":
                message = event["message"]
                actions = event["actions"]
                usage = event.get("usage")

        self.console.print()
        if usage and on_usage:
            on_usage(usage)

        return message, actions, usage

    def print_tool_outputs(self, actions: list[dict], outputs: list[dict]):
        for action, output in zip(actions, outputs):
            tool_name = action.get("tool_name", "")
            arguments = action.get("arguments", {})
            cmd = (
                arguments.get("command", "") if tool_name == "bash" else str(arguments)
            )
            rc = output["returncode"]
            out = (output.get("output") or "").rstrip()

            elements: list = [Syntax(cmd, "bash", theme="ansi_dark")]
            if out:
                elements.append(Text(""))
                elements.append(Text(out))

            subtitle = f"exit code: {rc}" if rc else "(0)"
            border_style = "green" if rc == 0 else "red"

            self.console.print(
                Panel(
                    Group(*elements) if len(elements) > 1 else elements[0],
                    subtitle=subtitle,
                    border_style=border_style,
                    padding=(0, 1),
                    title="Bash",
                )
            )

    def print_cost_report(self, report: dict[str, float | int]):
        table = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
        table.add_column(style="bold", justify="left")
        table.add_column(justify="right")
        table.add_row("Session cost", f"${report['cost']:.6f}")
        table.add_row("Calls", str(report["calls"]))
        table.add_row(
            "Uncached in",
            f"{report['uncached_tokens']:>8,} tok  ${report['uncached_cost']:.6f}",
        )
        table.add_row(
            "Cached in",
            f"{report['cached_tokens']:>8,} tok  ${report['cached_cost']:.6f}",
        )
        table.add_row(
            "Output",
            f"{report['completion_tokens']:>8,} tok  ${report['completion_cost']:.6f}",
        )
        self.console.print(table)

    def print_tool_call_count(self, n_tool_calls: int):
        self.console.print(f"Tool calls: {n_tool_calls}", style="dim")

    def print_help(self, commands: list):
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column(style="bold cyan", justify="left")
        table.add_column(justify="left")
        for cmd in commands:
            table.add_row(f"/{cmd.name}", cmd.description)
        self.console.print(table)

    def print_clear_confirmation(self):
        self.console.print("Conversation cleared.", style="green")

    def print_summarization_result(self, final_tokens: int, max_tokens: int):
        pct = final_tokens / max_tokens * 100
        self.console.print(
            f"Context: {final_tokens:,} / {max_tokens:,} tok ({pct:.1f}%)",
            style="green",
        )

    def print_summarization_skip(self):
        self.console.print("Nothing to summarize yet.", style="dim")

    def print_summarization_input(self, info: dict, max_context_tokens: int):
        old_turns = info.get("old_turns", 0)
        original_tokens = info.get("original_tokens", 0)
        input_lines = info.get("input_lines", [])
        total_tokens = info.get("total_tokens", 0)

        pct = total_tokens / max_context_tokens * 100
        self.console.print(
            f"Context: {total_tokens:,} / {max_context_tokens:,} tok ({pct:.1f}%)",
            style="green",
        )
        self.console.print(
            f"Summarizing {old_turns} turn(s) ({original_tokens:,} tok):", style="bold"
        )
        for line in input_lines:
            self.console.print(f"  {line}", style="dim")
        self.console.print()
        self.console.print("Generating summary...", style="bold green")
        self.console.print()

    def print_interior_status(self, context, messages: list[dict], total_messages: int):
        events = context.get_events()
        max_ctx = context.max_context_tokens

        lines: list = [Text("Context Window Log", style="bold underline")]

        if not events:
            lines.append(Text("  (no events yet)", style="dim"))

        for i, ev in enumerate(events, 1):
            t = ev["total_tokens"]
            pct = t / max_ctx * 100

            if ev["type"] == "skip":
                lines.append(
                    Text(
                        f"  [{i}] skip — {t:,} / {max_ctx:,} tok ({pct:.1f}%) — no compression"
                    )
                )
            elif ev["type"] == "skip_summarize":
                lines.append(
                    Text(
                        f"  [{i}] skip — {t:,} / {max_ctx:,} tok ({pct:.1f}%) — nothing to summarize"
                    )
                )
            elif ev["type"] == "clear":
                lines.append(
                    Text(
                        f"  [{i}] clear — {t:,} tok — cleared {ev['count']} tool result(s): {ev['original_lines']}→{ev['new_lines']} lines"
                    )
                )
            elif ev["type"] == "summarize":
                prev = ev.get("summary_preview", "")
                lines.append(
                    Text(
                        f'  [{i}] summarize — {t:,} tok — {ev["old_turns"]} turns ({ev["original_tokens"]:,}→{ev["summary_tokens"]:,} tok) "{prev[:60]}"'
                    )
                )
            elif ev["type"] == "aggressive":
                lines.append(Text(f"  [{i}] aggressive — {t:,} tok — {ev['reason']}"))

        lines.append(Text(""))
        last = events[-1] if events else {}
        last_tok = last.get("total_tokens", context.count_tokens(messages))
        pct = last_tok / max_ctx * 100
        lines.append(
            Text(f"  Mode: {'verbose ON' if context.interior_mode else 'verbose OFF'}")
        )
        lines.append(Text(f"  Messages: {total_messages}  |  Events: {len(events)}"))
        lines.append(Text(f"  Last ctx: {last_tok:,} / {max_ctx:,} tok ({pct:.1f}%)"))

        self.console.print(Panel(Group(*lines), border_style="dim"))

    def print_interior_event(self, event: dict, max_context_tokens: int):
        t = event["total_tokens"]
        pct = t / max_context_tokens * 100
        line = f"  — ctx: {t:>6,} / {max_context_tokens:>6,} tok ({pct:.1f}%)"

        if event["type"] == "skip":
            line += " — no compression needed —"
        elif event["type"] == "skip_summarize":
            line += " — nothing to summarize —"
        elif event["type"] == "clear":
            line += f" — cleared {event['count']} tool result(s): {event['original_lines']}→{event['new_lines']} lines —"
        elif event["type"] == "summarize":
            line += f" — summarized {event['old_turns']} turns: {event['original_tokens']:,}→{event['summary_tokens']:,} tok —"
        elif event["type"] == "aggressive":
            line += f" — {event['reason']} —"

        self.console.print(Text(line, style="dim"))

    def print_limits_exceeded(self, messages: list[dict]):
        for msg in messages:
            self.console.print(Text(f"\n{msg.get('content', '')}", style="bold red"))

    def print_interior_toggle(self, on: bool):
        if on:
            self.console.print("Interior mode: ON", style="bold yellow")
        else:
            self.console.print("Interior mode: OFF", style="dim")

    def print_reasoning_toggle(self, on: bool):
        if on:
            self.console.print("Reasoning: ON", style="bold yellow")
        else:
            self.console.print("Reasoning: OFF", style="dim")

    def print_unknown_subcommand(self, arg: str):
        self.console.print(
            Text(f"Unknown: {arg}. Use: /show-interior [on|off]", style="yellow")
        )

    def print_retry_warning(self, attempt: int, wait_time: int, error: str):
        self.console.print(
            Text(
                f"  ⚠ API error (attempt {attempt}/3). Retrying in {wait_time}s: {error}",
                style="yellow",
            )
        )

    def print_cancelled(self):
        self.console.print()
