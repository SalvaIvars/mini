from typing import Any, Iterator, Protocol


class Model(Protocol):
    def query(self, messages: list[dict], **kwargs) -> dict: ...
    def stream(self, messages: list[dict], **kwargs) -> Iterator[dict]: ...
    def format_message(self, **kwargs) -> dict: ...
    def format_observation_messages(
        self, message: dict, outputs: list[dict], template_vars: dict | None = None
    ) -> list[dict]: ...


class Environment(Protocol):
    def execute(self, action: dict, **kwargs) -> dict[str, Any]: ...


class Summarizer(Protocol):
    def stream(self, messages: list[dict], **kwargs) -> Iterator[dict]: ...


class Tool(Protocol):
    name: str
    description: str
    parameters: dict


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool):
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def get_schemas(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self._tools.values()
        ]
