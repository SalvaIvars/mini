import time

from openai import APIError, RateLimitError

from ._types import Environment, Model
from .commands import Command, CommandRegistry, SlashCommandCompleter
from .context import ContextWindow
from .cost_tracker import CostTracker
from .display import Display
from .exceptions import InterruptFlow, LimitsExceeded


class Mini:
    def __init__(
        self,
        model: Model,
        env: Environment,
        display: Display,
        *,
        call_limit: int = 50,
        cost_limit: float = 0,
        input_cost_per_token: float = 0,
        output_cost_per_token: float = 0,
        cached_input_cost_per_token: float = 0,
        max_context_tokens: int = 96000,
        keep_turns: int = 2,
    ):
        self.model = model
        self.env = env
        self.messages: list[dict] = []
        self.n_tool_calls = 0

        self.display = display
        self.tracker = CostTracker(
            call_limit=call_limit,
            cost_limit=cost_limit,
            input_cost_per_token=input_cost_per_token,
            output_cost_per_token=output_cost_per_token,
            cached_input_cost_per_token=cached_input_cost_per_token,
        )
        self.context = ContextWindow(
            model,
            max_context_tokens=max_context_tokens,
            keep_turns=keep_turns,
        )
        self.commands = CommandRegistry(self.display)
        self._register_commands()

    def _register_commands(self):
        self.commands.register(
            Command("cost", "Show token usage and cost for this session", "Info"),
            self._cmd_cost,
        )
        self.commands.register(
            Command("calls", "Show tool call count for this session", "Info"),
            self._cmd_calls,
        )
        self.commands.register(
            Command("show-interior", "Toggle verbose interior mode", "Debug"),
            self._cmd_show_interior,
        )
        self.commands.register(
            Command("show-reasoning", "Show model reasoning", "Debug"),
            self._cmd_show_reasoning,
        )
        self.commands.register(
            Command("hide-reasoning", "Hide model reasoning", "Debug"),
            self._cmd_hide_reasoning,
        )
        self.commands.register(
            Command("help", "Show all available commands", "Help"),
            self._cmd_help,
        )
        self.commands.register(
            Command("summarization", "Force context summarization", "Session"),
            self._cmd_summarization,
        )
        self.commands.register(
            Command("clear", "Clear conversation memory", "Session"),
            self._cmd_clear,
        )
        self.commands.register(
            Command("exit", "Quit the application", "Session"),
            lambda raw: None,
        )

    def start(self):
        self.messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful coding assistant. "
                    "You have access to a bash shell. Use it when needed, or just respond conversationally."
                ),
            },
        ]
        self.n_tool_calls = 0
        self.tracker.reset()
        self.context.reset()

        self.display.welcome(self.model.model_name)

        while True:
            try:
                user_input = self.display.prompt(SlashCommandCompleter(self.commands))
            except (EOFError, KeyboardInterrupt):
                self.display.print_cancelled()
                break

            if user_input.lower() in ("exit", "quit"):
                break

            if user_input.startswith("/"):
                continuing = self.commands.handle(user_input)
                if not continuing:
                    self.display.goodbye()
                    break
                continue

            self.messages.append({"role": "user", "content": user_input})
            try:
                self._run_turn()
            except InterruptFlow as e:
                if isinstance(e, LimitsExceeded):
                    self.display.print_limits_exceeded(e.messages)
                break

    def _run_turn(self):
        while True:
            self.tracker.check_limits()
            self.tracker.record_call()

            messages = self.context.prepare(self.messages)
            if self.context.interior_mode:
                last_event = self.context.last_event()
                if last_event:
                    self.display.print_interior_event(
                        last_event, self.context.max_context_tokens
                    )

            max_retries = 3
            message, actions = None, []
            for attempt in range(max_retries):
                try:
                    message, actions, _usage = self.display.stream_response(
                        self.model.stream(messages),
                        on_usage=self.tracker.update,
                    )
                    break
                except (RateLimitError, APIError) as e:
                    if attempt == max_retries - 1:
                        raise
                    wait_time = 2**attempt
                    self.display.print_retry_warning(attempt + 1, wait_time, str(e))
                    time.sleep(wait_time)

            if message is None:
                continue

            self.messages.append(message)

            if actions:
                self.n_tool_calls += len(actions)
                outputs = [self.env.execute(action) for action in actions]
                self.display.print_tool_outputs(actions, outputs)
                obs = self.model.format_observation_messages(message, outputs)
                self.messages.extend(obs)
            else:
                break

    def _cmd_cost(self, raw: str):
        self.display.print_cost_report(self.tracker.report())

    def _cmd_calls(self, raw: str):
        self.display.print_tool_call_count(self.n_tool_calls)

    def _cmd_show_interior(self, raw: str):
        parts = raw.split()
        if len(parts) == 1:
            self.context.set_interior_mode(not self.context.interior_mode)
            self.display.print_interior_toggle(self.context.interior_mode)
            if self.context.interior_mode:
                self.display.print_interior_status(
                    self.context, self.messages, len(self.messages)
                )
        elif len(parts) == 2:
            arg = parts[1]
            if arg == "on":
                self.context.set_interior_mode(True)
                self.display.print_interior_toggle(True)
                self.display.print_interior_status(
                    self.context, self.messages, len(self.messages)
                )
            elif arg == "off":
                self.context.set_interior_mode(False)
                self.display.print_interior_toggle(False)
            else:
                self.display.print_unknown_subcommand(arg)

    def _cmd_help(self, raw: str):
        self.display.print_help(self.commands.commands)

    def _cmd_show_reasoning(self, raw: str):
        self.display.show_reasoning = True
        self.display.print_reasoning_toggle(True)

    def _cmd_hide_reasoning(self, raw: str):
        self.display.show_reasoning = False
        self.display.print_reasoning_toggle(False)

    def _cmd_clear(self, raw: str):
        self.messages.clear()
        self.n_tool_calls = 0
        self.tracker.reset()
        self.context.reset()
        self.display.print_clear_confirmation()

    def _cmd_summarization(self, raw: str):
        info = self.context.summarize(self.messages)

        if info is None:
            self.display.print_summarization_skip()
            return

        self.display.print_summarization_input(info, self.context.max_context_tokens)

        message, _actions, _usage = self.display.stream_response(
            info["stream"],
            force_hide_reasoning=True,
        )
        summary = message.get("content", "") if message else ""

        if summary:
            summary_msg = {
                "role": "system",
                "content": f"## Session History\n{summary}",
            }
            self.messages = info["system"] + [summary_msg] + info["recent"]

            summary_tok = self.context.count_tokens([summary_msg])
            final_tokens = self.context.count_tokens(self.messages)
            self.context.log_event(
                {
                    "type": "summarize",
                    "total_tokens": final_tokens,
                    "old_turns": info["old_turns"],
                    "original_tokens": info["original_tokens"],
                    "summary_tokens": summary_tok,
                    "summary_preview": summary[:80],
                    "input_lines": info["input_lines"],
                    "summary_full": summary,
                }
            )

            self.display.print_summarization_result(
                final_tokens, self.context.max_context_tokens
            )
        else:
            self.display.print_summarization_skip()
