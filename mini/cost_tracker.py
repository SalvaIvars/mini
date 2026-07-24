from dataclasses import dataclass, field

from .exceptions import LimitsExceeded


@dataclass
class CostTracker:
    """Tracks token usage, cost, and enforces spend/call limits."""

    step_limit: int = 50
    cost_limit: float = 0.0
    input_cost_per_token: float = 0.0
    output_cost_per_token: float = 0.0
    cached_input_cost_per_token: float = 0.0

    cost: float = field(default=0.0, init=False)
    n_calls: int = field(default=0, init=False)
    total_uncached_tokens: int = field(default=0, init=False)
    total_cached_tokens: int = field(default=0, init=False)
    total_completion_tokens: int = field(default=0, init=False)

    def reset(self):
        self.cost = 0.0
        self.n_calls = 0
        self.total_uncached_tokens = 0
        self.total_cached_tokens = 0
        self.total_completion_tokens = 0

    def record_call(self):
        """Increment the LLM call counter."""
        self.n_calls += 1

    def check_limits(self):
        if 0 < self.step_limit <= self.n_calls:
            raise LimitsExceeded({
                "role": "exit",
                "content": "LimitsExceeded",
                "extra": {"exit_status": "LimitsExceeded", "submission": ""},
            })
        if 0 < self.cost_limit <= self.cost:
            raise LimitsExceeded({
                "role": "exit",
                "content": "CostLimitExceeded",
                "extra": {"exit_status": "CostLimitExceeded", "submission": ""},
            })

    def update(self, usage: dict):
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        cached_tokens = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
        uncached = prompt_tokens - cached_tokens

        self.cost += uncached * self.input_cost_per_token
        self.cost += cached_tokens * self.cached_input_cost_per_token
        self.cost += completion_tokens * self.output_cost_per_token

        self.total_uncached_tokens += uncached
        self.total_cached_tokens += cached_tokens
        self.total_completion_tokens += completion_tokens

    def report(self) -> dict[str, float | int]:
        return {
            "cost": self.cost,
            "calls": self.n_calls,
            "uncached_tokens": self.total_uncached_tokens,
            "cached_tokens": self.total_cached_tokens,
            "completion_tokens": self.total_completion_tokens,
            "uncached_cost": self.total_uncached_tokens * self.input_cost_per_token,
            "cached_cost": self.total_cached_tokens * self.cached_input_cost_per_token,
            "completion_cost": self.total_completion_tokens * self.output_cost_per_token,
        }
