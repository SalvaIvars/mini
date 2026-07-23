from .commands import Command, CommandRegistry, SlashCommandCompleter
from .context import ContextWindow
from .cost_tracker import CostTracker
from .display import Display
from .exceptions import InterruptAgentFlow, LimitsExceeded


class DefaultAgent:
    def __init__(
        self,
        model,
        env,
        *,
        step_limit: int = 50,
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

        self.display = Display()
        self.tracker = CostTracker(
            step_limit=step_limit,
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
        self.commands = CommandRegistry()
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
            Command("help", "Show all available commands", "Help"),
            self._cmd_help,
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
            except InterruptAgentFlow as e:
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
                    self.display.print_interior_event(last_event, self.context.max_context_tokens)

            message, actions, _usage = self.display.stream_response(
                self.model.stream(messages),
                on_usage=self.tracker.update,
            )

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
                self.display.print_interior_status(self.context, self.messages, len(self.messages))
        elif len(parts) == 2:
            arg = parts[1]
            if arg == "on":
                self.context.set_interior_mode(True)
                self.display.print_interior_toggle(True)
                self.display.print_interior_status(self.context, self.messages, len(self.messages))
            elif arg == "off":
                self.context.set_interior_mode(False)
                self.display.print_interior_toggle(False)
            else:
                self.display.print_unknown_subcommand(arg)

    def _cmd_help(self, raw: str):
        self.display.print_help(self.commands.commands)

    def _cmd_clear(self, raw: str):
        self.messages.clear()
        self.n_tool_calls = 0
        self.tracker.reset()
        self.context.reset()
        self.display.print_clear_confirmation()
