# mini

A lightweight, modular AI coding agent that runs in the terminal. Connects to any OpenAI-compatible API, exposes a bash tool, and provides full visibility into every LLM call via an inspectable context window.

## Features

- **Streaming REPL** with syntax-highlighted tool output, reasoning visibility, and tab-completion for slash commands
- **Modular architecture** based on protocol-driven dependency injection -- swap models or execution environments without touching the core loop
- **Transparent context management** -- interior mode shows token counts, compression decisions, and event history turn by turn
- **Reasoning visibility** -- toggle model reasoning on/off with `/show-reasoning` and `/hide-reasoning`
- **On-demand summarization** -- manually compress context with `/summarization`
- **Three-level context compression** (clear, summarize, aggressive) keeps long sessions under budget without losing critical information
- **Automatic retry** with exponential backoff for transient API errors (429, 500, 503)
- **Extensible tool system** via ToolRegistry -- add new tools without modifying the core loop
- **Cost tracking** with per-token pricing and hard step/cost limits
- **Single binary entry point** via `mini` CLI command

## Architecture

```
cli.py           -- Typer CLI entry point, wires dependencies
core.py          -- Mini orchestrator: main loop, turn logic, command dispatch
model.py         -- OpenAIModel: streaming and non-streaming LLM API calls
environment.py   -- LocalEnvironment: subprocess-based bash execution
context.py       -- ContextWindow: token counting, compression, interior logging
cost_tracker.py  -- CostTracker: token/cost accounting, limit enforcement
display.py       -- Display: all terminal I/O (Rich + prompt-toolkit)
commands.py      -- Command / CommandRegistry / SlashCommandCompleter
exceptions.py    -- InterruptFlow / LimitsExceeded / FormatError
_types.py        -- Model, Environment, Summarizer, Tool protocols and ToolRegistry for extensible tool system
```

The `Model`, `Environment`, `Summarizer`, and `Tool` protocols in `_types.py` allow swapping implementations. The `ToolRegistry` enables adding new tools without modifying the core loop. The `Mini` class receives any object conforming to these protocols.

## Extensible tools

The tool system is extensible via `ToolRegistry`. To add a new tool:
1. Create a class implementing the `Tool` protocol (with `name`, `description`, `parameters`)
2. Register it in `cli.py` with `tool_registry.register(YourTool())`
3. Handle execution in `LocalEnvironment.execute()`

The agent will automatically expose the tool to the LLM.

## Installation

Requires Python >= 3.11.

```bash
pip install mini
```

Or with uv:

```bash
uv pip install mini
```

## Usage

```bash
mini --model "gpt-4o" --api-key "sk-..."
```

### CLI options

| Flag | Env var | Default | Description |
|------|---------|---------|-------------|
| `-m`, `--model` | `MODEL_NAME` | `deepseek-v4-flash` | Model name |
| `--api-base` | `API_BASE` | `https://opencode.ai/zen/go/v1` | API base URL |
| `--api-key` | `API_KEY` | `""` | API key |
| `--max-ctx` | | `96000` | Max context tokens before compression |
| `--keep-turns` | | `2` | Full-fidelity turns to preserve in context |

Pricing is configured via environment variables: `INPUT_PRICE`, `OUTPUT_PRICE`, `CACHED_PRICE`.

### Slash commands

| Command | Description |
|---------|-------------|
| `/cost` | Show token usage and cost for this session |
| `/calls` | Show tool call count |
| `/show-interior [on\|off]` | Toggle verbose interior mode |
| `/show-reasoning` | Show model reasoning |
| `/hide-reasoning` | Hide model reasoning |
| `/summarization` | Force context summarization |
| `/help` | List all commands |
| `/clear` | Clear conversation memory |
| `/exit` | Quit |

## Context compression

The context window compresses messages when token count exceeds 75% of the limit. Compression runs in three escalating phases:

1. **Clear** -- truncates large tool outputs (keeps first 3 and last 3 lines)
2. **Summarize** -- sends old turns to the LLM for a bullet-point summary, injected as a system message
3. **Aggressive** -- drops the summary and, if needed, aggressively truncates recent tool outputs

All compression events are logged and visible via interior mode.

## Retry logic

The agent automatically retries API calls on transient errors (429, 500, 503) with exponential backoff (3 attempts max: 1s, 2s, 4s). Retries are visible in the terminal output.

## Interior mode

Interior mode (`/show-interior`) prints a one-line summary after every compression pass showing token totals, compression type, and savings. A full event log is available with `/show-interior` (toggle) or by running the command again to see the panel.

This gives visibility into what the model sees at each step, how much context is consumed, and when/how compression is applied.

## Development

```bash
uv sync
uv run pytest
```

The test suite covers all core modules with mocked model and environment interfaces.

## License

MIT
