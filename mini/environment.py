import os
import signal
import subprocess
from typing import Any


class LocalEnvironment:
    def __init__(self, cwd: str = "", timeout: int = 30):
        self.cwd = cwd or os.getcwd()
        self.timeout = timeout

    def execute(self, action: dict, **kwargs) -> dict[str, Any]:
        tool_name = action.get("tool_name", "")
        arguments = action.get("arguments", {})
        if tool_name == "bash":
            command = arguments.get("command", "")
            return self._run(command)
        return {"output": f"Unknown tool: {tool_name}", "returncode": -1}

    def _run(self, command: str) -> dict[str, Any]:
        try:
            result = subprocess.run(
                command,
                shell=True,
                text=True,
                cwd=self.cwd,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=os.name == "posix",
                timeout=self.timeout,
            )
            return {"output": result.stdout, "returncode": result.returncode}
        except subprocess.TimeoutExpired as e:
            return {
                "output": e.output or "",
                "returncode": -1,
                "error": f"Command timed out after {self.timeout}s",
            }
        except Exception as e:
            return {"output": str(e), "returncode": -1, "error": str(e)}
