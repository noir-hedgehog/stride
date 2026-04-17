"""
Stride Loop Tests
Run with: python -m pytest tests/ -v
"""
import pytest
import time
import threading
from unittest.mock import MagicMock, patch

from stride.loop import FrameLoop, FrameResult
from stride.world_model import WorldModel
from stride.sensors import SensorSuite
from stride.actors import ActorSuite
from stride.ai import AIClient


class DummySensorSuite:
    def collect_all(self):
        return {
            "system": {"cpu_raw": "10% user", "mem_raw": "8GB used"},
            "gui": {"screenshots": []},
            "cli": {"pending_commands": []},
        }


class DummyActorSuite:
    def execute(self, action):
        return {"status": "ok", "action": action.get("type")}


class DummyAIClient:
    def decide(self, frame_context):
        return {
            "reasoning": "test decision",
            "actions": [{"type": "shell", "command": "echo test"}],
            "observations": "test observation",
        }


@pytest.fixture
def short_frame_loop(tmp_path):
    from stride.world_model import WorldModel
    loop = FrameLoop(
        world_model=WorldModel(path=tmp_path / "world_model.json"),
        sensor_suite=DummySensorSuite(),
        actor_suite=DummyActorSuite(),
        ai_client=DummyAIClient(),
    )
    loop.FRAME_DURATION = 2
    loop.PHASE_DURATIONS = {"collect": 0.3, "decide": 0.3, "act": 0.3}
    return loop


def test_frame_loop_single_frame(short_frame_loop):
    result = short_frame_loop.run_frame()
    assert isinstance(result, FrameResult)
    assert result.frame_number == 1
    assert result.phase_timings["total"] < 2
    assert result.world_model_snapshot["frame_number"] == 1


def test_world_model_save_and_load(tmp_path):
    wm1 = WorldModel(path=tmp_path / "test_wm.json")
    wm1.update({"frame_number": 5, "notes": ["test note"]})
    wm1.save()
    wm2 = WorldModel(path=tmp_path / "test_wm.json")
    assert wm2.read_key("frame_number") == 5


def test_actor_suite_unknown_action():
    actors = ActorSuite()
    result = actors.execute({"type": "unknown_type"})
    assert result["status"] == "error"


def test_actor_suite_shell_action():
    actors = ActorSuite()
    result = actors.execute({"type": "shell", "command": "echo hello"})
    assert result["status"] == "ok"
    assert "hello" in result.get("stdout", "")


def test_ai_client_no_api_key():
    import os
    old_key = os.environ.pop("ANTHROPIC_API_KEY", None)
    try:
        ai = AIClient(api_key=None)
        decision = ai.decide({"frame_number": 1})
        assert decision["actions"] == []
        assert "not set" in decision["reasoning"].lower()
    finally:
        if old_key:
            os.environ["ANTHROPIC_API_KEY"] = old_key
