"""
World Model — Shared state store for all pipelines.
Stored as JSON at ~/.stride/world_model.json
"""
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

WORLD_MODEL_PATH = Path.home() / ".stride" / "world_model.json"


class WorldModel:
    """
    Thread-safe(ish) JSON world model. All pipelines read/write through here.

    The world model is the single source of truth for the agent's view of the world.
    Each frame reads the previous state and writes updates at the end.
    """

    def __init__(self, path: Optional[Path] = None):
        self.path = path or WORLD_MODEL_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._state: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        """Load world model from disk, return empty dict if missing."""
        if not self.path.exists():
            logger.info(f"World model not found at {self.path}, starting fresh")
            return self._default_state()
        try:
            with open(self.path) as f:
                state = json.load(f)
            logger.debug(f"World model loaded from {self.path}")
            return state
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Could not load world model: {e}, starting fresh")
            return self._default_state()

    def _default_state(self) -> Dict[str, Any]:
        return {
            "version": "0.1",
            "frame_number": 0,
            "started_at": None,
            "sensors": {},
            "last_frame_at": None,
            "last_decision": {},
            "last_actions": [],
            "last_errors": [],
            "notes": [],
        }

    def snapshot(self) -> Dict[str, Any]:
        """Return a copy of the current state for reading."""
        return dict(self._state)

    def update(self, updates: Dict[str, Any]) -> None:
        """Merge updates into the current state."""
        if "started_at" not in self._state and "started_at" not in updates:
            updates["started_at"] = updates.get("started_at") or self._state.get("started_at")
        self._state.update(updates)

    def save(self) -> None:
        """Write current state to disk atomically."""
        tmp = self.path.with_suffix(".tmp")
        try:
            with open(tmp, "w") as f:
                json.dump(self._state, f, indent=2, default=str)
            tmp.rename(self.path)
            logger.debug(f"World model saved to {self.path}")
        except IOError as e:
            logger.error(f"Failed to save world model: {e}")

    def read_key(self, key: str, default: Any = None) -> Any:
        """Read a single key from the state."""
        return self._state.get(key, default)

    def append_note(self, text: str) -> None:
        """Append a note to the notes list (useful for milestones)."""
        notes = self._state.get("notes", [])
        notes.append({"text": text, "at": self._state.get("last_frame_at", "unknown")})
        self._state["notes"] = notes
