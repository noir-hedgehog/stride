"""
Stride Frame Loop

Main loop: runs at fixed 60s per frame, each frame has 3 phases:
- Input Collection:  20s
- Analysis/Decision:  20s
- Output Execution:   20s
"""
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from datetime import datetime

from .world_model import WorldModel
from .sensors import SensorSuite
from .actors import ActorSuite
from .ai import AIClient

logger = logging.getLogger(__name__)


@dataclass
class FrameResult:
    frame_number: int
    phase_timings: Dict[str, float]
    world_model_snapshot: Dict[str, Any]
    actions_taken: list = field(default_factory=list)
    errors: list = field(default_factory=list)


class FrameLoop:
    """
    Fixed-frame loop. Each frame runs: collect → decide → act → persist.
    """

    FRAME_DURATION = 60  # seconds
    PHASE_DURATIONS = {
        "collect": 20,
        "decide": 20,
        "act": 20,
    }

    def __init__(
        self,
        world_model: Optional[WorldModel] = None,
        sensor_suite: Optional[SensorSuite] = None,
        actor_suite: Optional[ActorSuite] = None,
        ai_client: Optional[AIClient] = None,
    ):
        self.world = world_model or WorldModel()
        self.sensors = sensor_suite or SensorSuite()
        self.actors = actor_suite or ActorSuite()
        self.ai = ai_client or AIClient()

        self.frame_number = 0
        self.running = False

    def run_frame(self) -> FrameResult:
        """Run one complete frame (collect → decide → act)."""
        self.frame_number += 1
        start_time = time.time()
        errors = []
        actions_taken = []

        logger.info(f"[Frame {self.frame_number}] Starting frame at {datetime.now().isoformat()}")

        # ── Phase 1: Input Collection ──────────────────────────────────────────
        collect_start = time.time()
        try:
            sensor_data = self.sensors.collect_all()
            self.world.update({"sensors": sensor_data, "last_collect_at": datetime.now().isoformat()})
        except Exception as e:
            logger.error(f"[Frame {self.frame_number}] Collection error: {e}")
            errors.append(f"collect: {e}")
            sensor_data = {}
        collect_duration = time.time() - collect_start
        logger.info(f"[Frame {self.frame_number}] Collection done in {collect_duration:.1f}s")

        # Sleep to fill remaining collect time
        self._sleep_remaining(collect_start, "collect")

        # ── Phase 2: Analysis & Decision ──────────────────────────────────────
        decide_start = time.time()
        decision = {}
        try:
            decision = self.ai.decide(frame_context=self.world.snapshot())
        except Exception as e:
            logger.error(f"[Frame {self.frame_number}] Decision error: {e}")
            errors.append(f"decide: {e}")
            decision = {}
        decide_duration = time.time() - decide_start
        logger.info(f"[Frame {self.frame_number}] Decision done in {decide_duration:.1f}s")

        # Sleep to fill remaining decide time
        self._sleep_remaining(decide_start, "decide")

        # ── Phase 3: Output Execution ──────────────────────────────────────────
        act_start = time.time()
        try:
            if decision.get("actions"):
                for action in decision["actions"]:
                    result = self.actors.execute(action)
                    actions_taken.append({"action": action, "result": result})
                    logger.info(f"[Frame {self.frame_number}] Action: {action.get('type')} → {result.get('status')}")
        except Exception as e:
            logger.error(f"[Frame {self.frame_number}] Execution error: {e}")
            errors.append(f"act: {e}")
        act_duration = time.time() - act_start
        logger.info(f"[Frame {self.frame_number}] Execution done in {act_duration:.1f}s")

        # Sleep to fill remaining act time
        self._sleep_remaining(act_start, "act")

        # ── Persist world model ────────────────────────────────────────────────
        self.world.update({
            "last_frame_at": datetime.now().isoformat(),
            "last_decision": decision,
            "last_actions": actions_taken,
            "last_errors": errors,
            "frame_number": self.frame_number,
        })
        self.world.save()

        total_duration = time.time() - start_time
        logger.info(f"[Frame {self.frame_number}] Frame complete in {total_duration:.1f}s")

        return FrameResult(
            frame_number=self.frame_number,
            phase_timings={
                "collect": collect_duration,
                "decide": decide_duration,
                "act": act_duration,
                "total": total_duration,
            },
            world_model_snapshot=self.world.snapshot(),
            actions_taken=actions_taken,
            errors=errors,
        )

    def _sleep_remaining(self, phase_start: float, phase_name: str):
        """Sleep to fill the remaining time for a phase."""
        elapsed = time.time() - phase_start
        remaining = self.PHASE_DURATIONS[phase_name] - elapsed
        if remaining > 0:
            logger.debug(f"[Frame {self.frame_number}] {phase_name}: sleeping {remaining:.1f}s")
            time.sleep(remaining)

    def run(self, max_frames: Optional[int] = None):
        """
        Run the loop indefinitely (or for max_frames).
        Each frame waits until FRAME_DURATION from the previous frame start.
        """
        self.running = True
        logger.info(f"Strider loop starting (max_frames={max_frames})")

        while self.running:
            frame_start = time.time()

            result = self.run_frame()

            if max_frames and self.frame_number >= max_frames:
                logger.info(f"Reached max_frames={max_frames}, stopping")
                self.running = False
                break

            # Wait for next frame boundary
            elapsed = time.time() - frame_start
            if elapsed < self.FRAME_DURATION:
                sleep_time = self.FRAME_DURATION - elapsed
                logger.debug(f"Frame {self.frame_number} done early, sleeping {sleep_time:.1f}s until next frame")
                time.sleep(sleep_time)

    def stop(self):
        self.running = False
        logger.info("Strider loop stopping")
