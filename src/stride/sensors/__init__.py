"""
Sensors — Input collection layer.

Each sensor collects data from a source and returns a dict.
Sensors are called during the 20s collect phase.
"""
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class SensorSuite:
    """Collects all sensor data."""

    def collect_all(self) -> Dict[str, Any]:
        """Collect all sensor inputs. Returns a flat dict."""
        results = {}
        results["system"] = SystemSensor().read()
        results["gui"] = GUISensor().read()
        results["cli"] = CLISensor().read()
        return results


class SystemSensor:
    """Captures OS-level state: CPU, memory, disk."""

    def read(self) -> Dict[str, Any]:
        data = {}

        # Disk usage
        try:
            df = subprocess.run(
                ["df", "-h", "/"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            lines = df.stdout.strip().split("\n")
            if len(lines) >= 2:
                parts = lines[1].split()
                data["disk_total"] = parts[1]
                data["disk_used"] = parts[2]
                data["disk_avail"] = parts[3]
                data["disk_use_pct"] = parts[4]
        except Exception as e:
            logger.warning(f"SystemSensor df failed: {e}")

        # CPU + memory via top
        try:
            top = subprocess.run(
                ["top", "-l", "1", "-n", "5"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            # Parse CPU idle from first ~5 lines of output
            for line in top.stdout.split("\n")[:8]:
                if "CPU usage" in line:
                    # e.g. "  12% user,  3% sys,  84% idle"
                    data["cpu_raw"] = line.strip()
                if "PhysMem" in line:
                    # e.g. "PhysMem: 8192M used (3214M wired), 8192M unused."
                    data["mem_raw"] = line.strip()
        except Exception as e:
            logger.warning(f"SystemSensor top failed: {e}")

        return data


class GUISensor:
    """
    Captures GUI state via macOS screenshot.
    Uses `screencapture` to take a screenshot and stores the path.
    """
    SCREENSHOT_DIR = Path.home() / ".stride" / "screenshots"

    def __init__(self):
        self.SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

    def read(self) -> Dict[str, Any]:
        data = {"screenshots": []}

        try:
            import time
            ts = int(time.time())
            screenshot_path = self.SCREENSHOT_DIR / f"screenshot-{ts}.png"

            subprocess.run(
                ["screencapture", "-x", str(screenshot_path)],
                check=True,
                capture_output=True,
                timeout=10,
            )
            if screenshot_path.exists():
                size_kb = screenshot_path.stat().st_size // 1024
                data["screenshots"].append({
                    "path": str(screenshot_path),
                    "size_kb": size_kb,
                    "timestamp": ts,
                })
                data["latest"] = str(screenshot_path)
                logger.debug(f"Screenshot saved: {screenshot_path} ({size_kb}KB)")
        except subprocess.TimeoutExpired:
            logger.warning("GUISensor screencapture timed out")
        except Exception as e:
            logger.warning(f"GUISensor screencapture failed: {e}")

        return data


class CLISensor:
    """
    Reads CLI commands from a FIFO pipe at ~/.stride/cli_input.fifo
    If no FIFO exists, reads from a plain log file (~/.stride/cli_log.json)
    """
    FIFO_PATH = Path.home() / ".stride" / "cli_input.fifo"
    LOG_PATH = Path.home() / ".stride" / "cli_log.json"

    def read(self) -> Dict[str, Any]:
        data = {"pending_commands": [], "log_entries": []}

        # Try reading from FIFO (non-blocking)
        try:
            import select
            if self.FIFO_PATH.exists():
                # Read available data without blocking
                try:
                    ready, _, _ = select.select([open(self.FIFO_PATH)], [], [], 0.1)
                    if ready:
                        fifo_file = open(self.FIFO_PATH)
                        raw = fifo_file.read()
                        fifo_file.close()
                        for line in raw.strip().split("\n"):
                            if line.strip():
                                data["pending_commands"].append(line.strip())
                        # Clear FIFO after reading
                        open(self.FIFO_PATH, "w").close()
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"CLISensor FIFO read failed: {e}")

        # Fallback: read from log file
        if not data["pending_commands"] and self.LOG_PATH.exists():
            try:
                import json
                with open(self.LOG_PATH) as f:
                    log = json.load(f)
                    data["log_entries"] = log[-10:]  # last 10 entries
            except Exception as e:
                logger.warning(f"CLISensor log read failed: {e}")

        return data
