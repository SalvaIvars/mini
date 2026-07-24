# mini

A lightweight, modular AI coding agent that runs in the terminal. Connects to any OpenAI-compatible API, exposes a bash tool, and provides full visibility into every LLM call via an inspectable context window.

## Features

- **Streaming REPL** with syntax-highlighted tool output, reasoning visibility, and tab-completion for slash commands
- **Modular architecture** based on protocol-driven dependency injection -- swap models or execution environments without touching the core loop
- **Transparent context management** -- interior mode shows token counts, compression decisions, and event history turn by turn
- **Three-level context compression** (clear, summarize, aggressive) keeps long sessions under budget without losing critical information
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
_types.py        -- Model and Environment protocols for DI and testability
```

The `Model` and `Environment` protocols in `_types.py` allow swapping implementations. The `Mini` class receives any object conforming to these protocols.

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
| `/help` | List all commands |
| `/clear` | Clear conversation memory |
| `/exit` | Quit |

## Context compression

The context window compresses messages when token count exceeds 75% of the limit. Compression runs in three escalating phases:

1. **Clear** -- truncates large tool outputs (keeps first 3 and last 3 lines)
2. **Summarize** -- sends old turns to the LLM for a bullet-point summary, injected as a system message
3. **Aggressive** -- drops the summary and, if needed, aggressively truncates recent tool outputs

All compression events are logged and visible via interior mode.

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
