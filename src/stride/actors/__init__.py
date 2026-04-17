"""Actors — Output execution layer."""
import subprocess
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


class ShellActor:
    def execute(self, action: Dict[str, Any]) -> Dict[str, Any]:
        cmd = action.get("command")
        if not cmd:
            return {"status": "error", "error": "No command provided"}
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=action.get("timeout", 30))
            return {"status": "ok", "stdout": result.stdout[:2000], "stderr": result.stderr[:500], "returncode": result.returncode}
        except subprocess.TimeoutExpired:
            return {"status": "error", "error": f"Command timed out"}
        except Exception as e:
            return {"status": "error", "error": str(e)}


class AppleScriptActor:
    def execute(self, action: Dict[str, Any]) -> Dict[str, Any]:
        script = action.get("script")
        if not script:
            return {"status": "error", "error": "No AppleScript provided"}
        try:
            result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=action.get("timeout", 15))
            return {"status": "ok" if result.returncode == 0 else "error", "stdout": result.stdout.strip(), "stderr": result.stderr.strip()[:500]}
        except subprocess.TimeoutExpired:
            return {"status": "error", "error": "AppleScript timed out"}
        except Exception as e:
            return {"status": "error", "error": str(e)}


class WriteFileActor:
    def execute(self, action: Dict[str, Any]) -> Dict[str, Any]:
        path = action.get("path")
        if not path:
            return {"status": "error", "error": "No path provided"}
        try:
            file_path = Path(path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(action.get("content", ""))
            return {"status": "ok", "path": str(file_path)}
        except Exception as e:
            return {"status": "error", "error": str(e)}


class GUIClickActor:
    def execute(self, action: Dict[str, Any]) -> Dict[str, Any]:
        x, y = action.get("x"), action.get("y")
        if x is None or y is None:
            return {"status": "error", "error": "No x/y coordinates"}
        try:
            result = subprocess.run(["cliclick", f"c:{x},{y}"], capture_output=True, text=True, timeout=5)
            return {"status": "ok" if result.returncode == 0 else "error"}
        except FileNotFoundError:
            return {"status": "error", "error": "cliclick not installed"}
        except Exception as e:
            return {"status": "error", "error": str(e)}


class ActorSuite:
    def execute(self, action: Dict[str, Any]) -> Dict[str, Any]:
        actor_map = {
            "shell": ShellActor(),
            "apple_script": AppleScriptActor(),
            "write_file": WriteFileActor(),
            "gui_click": GUIClickActor(),
        }
        actor = actor_map.get(action.get("type", ""))
        if not actor:
            return {"status": "error", "error": f"Unknown action type: {action.get('type')}"}
        try:
            return actor.execute(action)
        except Exception as e:
            logger.error(f"Actor error: {e}")
            return {"status": "error", "error": str(e)}
