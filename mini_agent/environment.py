import os
import signal
import subprocess
from typing import Any


class LocalEnvironment:
    def __init__(self, cwd: str = "", timeout: int = 30):
        self.cwd = cwd or os.getcwd()
        self.timeout = timeout

    def execute(self, action: dict, **kwargs) -> dict[str, Any]:
        command = action.get("command", "")
        return self._run(command)

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
            return {"output": e.output or "", "returncode": -1, "error": f"Command timed out after {self.timeout}s"}
        except Exception as e:
            return {"output": str(e), "returncode": -1, "error": str(e)}
