"""
Stride CLI — Command-line interface for Stride.

Usage:
    python -m stride.cli run [--max-frames N] [--debug]
    python -m stride.cli status
    python -m stride.cli sensors
    python -m stride.cli browse
"""
import argparse
import json
import sys
from pathlib import Path

from stride.loop import FrameLoop
from stride.world_model import WorldModel
from stride.sensors import SensorSuite
from stride.actors import ActorSuite
from stride.ai import AIClient


def cmd_run(args):
    import logging
    import os

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    provider = args.provider or "anthropic"
    api_key_env = f"{provider.upper()}_API_KEY"
    api_key = args.api_key or os.environ.get(api_key_env)
    if not api_key:
        logger.warning(f"{api_key_env} not set — AI will return no-ops")

    world = WorldModel()
    sensors = SensorSuite()
    actors = ActorSuite()
    ai = AIClient(api_key=api_key, provider=provider)

    loop = FrameLoop(world_model=world, sensor_suite=sensors, actor_suite=actors, ai_client=ai)
    loop.FRAME_DURATION = args.frame_duration or 60

    print(f"Stride starting — provider={provider}, {args.max_frames or 'unlimited'} frames, {loop.FRAME_DURATION}s/frame")
    try:
        loop.run(max_frames=args.max_frames)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        loop.stop()


def cmd_status(args):
    wm = WorldModel()
    state = wm.snapshot()

    print("=== Stride Status ===")
    print(f"Frame:        #{state.get('frame_number', 0)}")
    print(f"Started:      {state.get('started_at', 'never')}")
    print(f"Last frame:   {state.get('last_frame_at', 'never')}")
    print(f"Last errors:  {state.get('last_errors', [])}")
    print()

    last_actions = state.get("last_actions", [])
    if last_actions:
        print("Last actions:")
        for a in last_actions[-3:]:
            print(f"  - {a.get('action', {}).get('type', '?')}: {a.get('result', {}).get('status', '?')}")
    else:
        print("No actions taken yet.")

    notes = state.get("notes", [])
    if notes:
        print(f"\nNotes ({len(notes)}):")
        for n in notes[-5:]:
            print(f"  [{n.get('at', '?')}] {n.get('text', '')[:80]}")


def cmd_sensors(args):
    sensors = SensorSuite()
    print("=== Sensor Output ===\n")
    data = sensors.collect_all()

    for name, section in data.items():
        print(f"--- {name} ---")
        if isinstance(section, dict):
            for k, v in section.items():
                print(f"  {k}: {v}")
        elif isinstance(section, list):
            for item in section:
                print(f"  - {item}")
        else:
            print(f"  {section}")
        print()


def cmd_browse(args):
    wm = WorldModel()
    state = wm.snapshot()
    print(json.dumps(state, indent=2, default=str))


def main():
    parser = argparse.ArgumentParser(prog="stride")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_parser = sub.add_parser("run", help="Run the agent loop")
    run_parser.add_argument("--max-frames", type=int, default=None)
    run_parser.add_argument("--frame-duration", type=int, default=None)
    run_parser.add_argument("--debug", action="store_true")
    run_parser.add_argument("--api-key", type=str, default=None)
    run_parser.add_argument("--provider", type=str, default=None, choices=["anthropic", "minimax"], help="AI provider (default: anthropic)")
    run_parser.set_defaults(func=cmd_run)

    sub.add_parser("status", help="Show world model status").set_defaults(func=cmd_status)
    sub.add_parser("sensors", help="Run all sensors and print output").set_defaults(func=cmd_sensors)
    sub.add_parser("browse", help="Print the full world model JSON").set_defaults(func=cmd_browse)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
