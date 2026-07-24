from typing import Any, Protocol


class Model(Protocol):
    def query(self, messages: list[dict], **kwargs) -> dict: ...
    def format_message(self, **kwargs) -> dict: ...
    def format_observation_messages(self, message: dict, outputs: list[dict], template_vars: dict | None = None) -> list[dict]: ...


class Environment(Protocol):
    def execute(self, action: dict, **kwargs) -> dict[str, Any]: ...



