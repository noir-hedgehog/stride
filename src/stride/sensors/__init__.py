import subprocess, platform, time, logging, json, select
from pathlib import Path

logger = logging.getLogger(__name__)

class SensorSuite:
    def collect_all(self):
        r = {}
        r["system"] = SystemSensor().read()
        r["gui"] = GUISensor().read()
        r["cli"] = CLISensor().read()
        return r

class SystemSensor:
    def read(self):
        data = {}
        try:
            r = subprocess.run(["df","-h","/"], capture_output=True, text=True, timeout=5)
            p = r.stdout.strip().split("\n")[1].split()
            data["disk_total"] = p[1]; data["disk_used"] = p[2]; data["disk_avail"] = p[3]; data["disk_use_pct"] = p[4]
        except Exception as e:
            logger.warning(f"SystemSensor df: {e}")
        plat = platform.system()
        if plat == "Linux":
            try:
                r = subprocess.run(["top","-b","-n","1"], capture_output=True, text=True, timeout=5)
                for ln in r.stdout.split("\n")[:8]:
                    if "Cpu(s)" in ln or "%Cpu" in ln: data["cpu_raw"] = ln.strip()
                    if "Mem:" in ln: data["mem_raw"] = ln.strip()
            except Exception as e:
                logger.warning(f"SystemSensor top: {e}")
        else:
            try:
                r = subprocess.run(["top","-l","1","-n","5"], capture_output=True, text=True, timeout=5)
                for ln in r.stdout.split("\n")[:8]:
                    if "CPU usage" in ln: data["cpu_raw"] = ln.strip()
                    if "PhysMem" in ln: data["mem_raw"] = ln.strip()
            except Exception as e:
                logger.warning(f"SystemSensor top: {e}")
        return data

class GUISensor:
    SCREENSHOT_DIR = Path.home() / ".stride" / "screenshots"
    def __init__(self):
        self.SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    def read(self):
        data = {"screenshots": [], "platform": platform.system()}
        cmd = self._cmd()
        ts = int(time.time())
        path = self.SCREENSHOT_DIR / f"screenshot-{ts}.png"
        if not cmd:
            data["error"] = f"No screenshot tool on {platform.system()}"
            return data
        try:
            subprocess.run(cmd + [str(path)], check=True, capture_output=True, timeout=10)
            if path.exists():
                data["screenshots"].append({"path": str(path), "size_kb": path.stat().st_size//1024, "timestamp": ts})
                data["latest"] = str(path)
        except Exception as e:
            data["error"] = str(e)
            logger.warning(f"GUISensor: {e}")
        return data
    def _cmd(self):
        if platform.system() == "Darwin": return ["screencapture", "-x"]
        if platform.system() == "Linux":
            for c in ["gnome-screenshot", "scrot"]:
                try:
                    subprocess.run(["which", c], capture_output=True, check=True, timeout=3)
                    return [c, "-f"] if c == "gnome-screenshot" else [c]
                except: pass
        return None

class CLISensor:
    FIFO = Path.home() / ".stride" / "cli_input.fifo"
    def read(self):
        data = {"pending_commands": [], "log_entries": []}
        try:
            if self.FIFO.exists():
                ready,_,_ = select.select([open(self.FIFO)],[],[], 0.1)
                if ready:
                    raw = open(self.FIFO).read(); open(self.FIFO,"w").close()
                    data["pending_commands"] = [l for l in raw.strip().split("\n") if l.strip()]
        except: pass
        return data
