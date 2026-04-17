"""
Stride Main Entry Point

Usage:
    python -m stride.main                    # Run indefinitely
    python -m stride.main --max-frames 3     # Run 3 frames only
    python -m stride.main --frame-duration 30 # 30s per frame (overrides default)
"""
import argparse
import logging
import os
import sys

from stride import __version__
from stride.loop import FrameLoop
from stride.world_model import WorldModel
from stride.sensors import SensorSuite
from stride.actors import ActorSuite
from stride.ai import AIClient


def setup_logging(level: int = logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Stride — Real-Time Frame-Based Agent")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N frames")
    parser.add_argument("--frame-duration", type=int, default=60, help="Frame duration in seconds")
    parser.add_argument("--api-key", type=str, default=None, help="Anthropic API key (or set ANTHROPIC_API_KEY env)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--version", action="store_true", help="Print version and exit")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.version:
        print(f"Stride v{__version__}")
        return

    setup_logging(level=logging.DEBUG if args.debug else logging.INFO)
    logger = logging.getLogger(__name__)

    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")

    if not api_key:
        logger.warning("ANTHROPIC_API_KEY not set — AI decisions will be no-ops")

    # Wire up components
    logger.info(f"Stride v{__version__} starting")
    logger.info(f"Frame duration: {args.frame_duration}s")

    world_model = WorldModel()
    sensors = SensorSuite()
    actors = ActorSuite()
    ai = AIClient(api_key=api_key)

    loop = FrameLoop(
        world_model=world_model,
        sensor_suite=sensors,
        actor_suite=actors,
        ai_client=ai,
    )

    # Override frame duration if specified
    if args.frame_duration != 60:
        loop.FRAME_DURATION = args.frame_duration

    try:
        loop.run(max_frames=args.max_frames)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        loop.stop()


if __name__ == "__main__":
    main()
