"""
Actors — Output execution layer.

Each actor executes a specific type of action.
Actions are dicts with at minimum a "type" field.
"""
import subprocess
import json
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


class ActorSuite:
    """Routes actions to the appropriate actor."""

    def execute(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an action dict.
        Expected fields:
          - type: str  (shell | apple_script | write_file | gui_click)
          - ... other fields vary by actor type
        Returns a result dict with at least "status": "ok" or "error"
        """
        action_type = action.get("type", "unknown")

        actor_map = {
            "shell": ShellActor(),
            "apple_script": AppleScriptActor(),
            "write_file": WriteFileActor(),
            "gui_click": GUIClickActor(),
        }
        actor = actor_map.get(action_type)

        if not actor:
            return {"status": "error", "error": f"Unknown action type: {action_type}"}

        try:
            return actor.execute(action)
        except Exception as e:
            logger.error(f"Actor {action_type} raised: {e}")
            return {"status": "error", "error": str(e)}


class ShellActor:
    """Executes a shell command."""

    def execute(self, action: Dict[str, Any]) -> Dict[str, Any]:
        cmd = action.get("command")
        if not cmd:
            return {"status": "error", "error": "No command provided"}

        timeout = action.get("timeout", 30)

        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return {
                "status": "ok",
                "stdout": result.stdout[:2000],
                "stderr": result.stderr[:500],
                "returncode": result.returncode,
            }
        except subprocess.TimeoutExpired:
            return {"status": "error", "error": f"Command timed out after {timeout}s"}
        except Exception as e:
            return {"status": "error", "error": str(e)}


class AppleScriptActor:
    """Executes an AppleScript command via osascript."""

    def execute(self, action: Dict[str, Any]) -> Dict[str, Any]:
        script = action.get("script")
        if not script:
            return {"status": "error", "error": "No AppleScript provided"}

        try:
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=action.get("timeout", 15),
            )
            return {
                "status": "ok" if result.returncode == 0 else "error",
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip()[:500],
                "returncode": result.returncode,
            }
        except subprocess.TimeoutExpired:
            return {"status": "error", "error": "AppleScript timed out"}
        except Exception as e:
            return {"status": "error", "error": str(e)}


class WriteFileActor:
    """Writes content to a file."""

    def execute(self, action: Dict[str, Any]) -> Dict[str, Any]:
        path = action.get("path")
        content = action.get("content", "")

        if not path:
            return {"status": "error", "error": "No path provided"}

        try:
            file_path = Path(path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content)
            return {
                "status": "ok",
                "path": str(file_path),
                "bytes_written": len(content),
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}


class GUIClickActor:
    """
    Clicks at screen coordinates using AppleScript.
    Requires `cliclick` binary (brew install cliclick).
    """
    def execute(self, action: Dict[str, Any]) -> Dict[str, Any]:
        x = action.get("x")
        y = action.get("y")
        click_type = action.get("click", "c")  # c=click, dc=double-click

        if x is None or y is None:
            return {"status": "error", "error": "No x/y coordinates provided"}

        try:
            result = subprocess.run(
                ["cliclick", f"{click_type}:{x},{y}"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return {
                "status": "ok" if result.returncode == 0 else "error",
                "stdout": result.stdout.strip(),
            }
        except FileNotFoundError:
            return {"status": "error", "error": "cliclick not installed (brew install cliclick)"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
