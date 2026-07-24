import json
import time

from openai import OpenAI

from .exceptions import FormatError

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash command",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "The bash command to execute"}},
            "required": ["command"],
        },
    },
}


class OpenAIModel:
    def __init__(self, model_name: str, api_key: str = "", api_base: str = ""):
        self.model_name = model_name
        kwargs = {}
        if api_key:
            kwargs["api_key"] = api_key
        if api_base:
            kwargs["base_url"] = api_base
        self.client = OpenAI(**kwargs)

    def query(self, messages: list[dict], **kwargs) -> dict:
        cleaned = [{k: v for k, v in m.items() if k != "extra"} for m in messages]
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=cleaned,
            tools=[BASH_TOOL],
            **kwargs,
        )
        actions = self._parse_actions(response)
        message = response.choices[0].message.model_dump()
        message["extra"] = {
            "actions": actions,
            "response": response.model_dump(),
            "timestamp": time.time(),
        }
        return message

    def stream(self, messages):
        cleaned = [{k: v for k, v in m.items() if k != "extra"} for m in messages]
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=cleaned,
            tools=[BASH_TOOL],
            stream=True,
            stream_options={"include_usage": True},
            extra_body={"thinking": {"type": "enabled"}},
        )
        content = ""
        reasoning = ""
        tool_calls_acc = {}
        usage = None
        for chunk in response:
            if getattr(chunk, "usage", None):
                usage = chunk.usage.model_dump()
            choice = chunk.choices[0] if chunk.choices else None
            if not choice:
                continue
            delta = choice.delta
            rc = getattr(delta, "reasoning_content", None)
            if rc:
                reasoning += rc
                yield {"type": "reasoning", "delta": rc}
            if delta.content:
                content += delta.content
                yield {"type": "content", "delta": delta.content}
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls_acc:
                        tool_calls_acc[idx] = {"id": "", "name": "", "arguments": ""}
                    if tc.id:
                        tool_calls_acc[idx]["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            tool_calls_acc[idx]["name"] = tc.function.name
                        if tc.function.arguments:
                            tool_calls_acc[idx]["arguments"] += tc.function.arguments
            if choice.finish_reason:
                for remaining in response:
                    if getattr(remaining, "usage", None):
                        usage = remaining.usage.model_dump()
                break
        message = {"role": "assistant", "content": content}
        if reasoning:
            message["reasoning_content"] = reasoning
        actions = []
        if tool_calls_acc:
            sorted_items = sorted(tool_calls_acc.items(), key=lambda x: x[0])
            tcs = []
            for _, data in sorted_items:
                tc = {
                    "id": data["id"],
                    "type": "function",
                    "function": {"name": data["name"], "arguments": data["arguments"]},
                }
                tcs.append(tc)
                if data["name"] == "bash":
                    args = json.loads(data["arguments"])
                    actions.append({"command": args["command"], "tool_call_id": data["id"]})
                else:
                    raise FormatError({"role": "user", "content": f"Unknown tool '{data['name']}'. Only 'bash' is allowed."})
            message["tool_calls"] = tcs
        message["extra"] = {"actions": actions, "timestamp": time.time(), "usage": usage}
        yield {"type": "done", "message": message, "actions": actions, "usage": usage}

    def _parse_actions(self, response) -> list[dict]:
        choice = response.choices[0]
        if not choice.message.tool_calls:
            return []
        actions = []
        for tc in choice.message.tool_calls:
            if tc.function.name != "bash":
                raise FormatError({"role": "user", "content": f"Unknown tool '{tc.function.name}'. Only 'bash' is allowed."})
            args = json.loads(tc.function.arguments)
            if "command" not in args:
                raise FormatError({"role": "user", "content": "Missing 'command' argument in bash tool call."})
            actions.append({"command": args["command"], "tool_call_id": tc.id})
        return actions

    def format_message(self, **kwargs) -> dict:
        return kwargs

    def format_observation_messages(self, message: dict, outputs: list[dict], template_vars: dict | None = None) -> list[dict]:
        actions = message.get("extra", {}).get("actions", [])
        padded = outputs + [{"output": "", "returncode": -1}] * (len(actions) - len(outputs))
        results = []
        for action, output in zip(actions, padded):
            msg = {
                "role": "tool",
                "content": f"<returncode>{output['returncode']}</returncode>\n<output>\n{output['output']}</output>",
                "tool_call_id": action["tool_call_id"],
            }
            results.append(msg)
        return results
