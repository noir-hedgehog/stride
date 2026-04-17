"""
AI Client — Anthropic Claude API integration.

The decide() method takes the current world model snapshot and returns
a JSON action plan.
"""
import os
import json
import logging
from typing import Dict, Any, Optional

import anthropic

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-4-5"


class AIClient:
    """
    Calls Anthropic Claude API with frame context.
    Returns a structured decision dict.
    """

    SYSTEM_PROMPT = """You are the decision engine for Stride, a real-time frame-based agent.

The agent runs in a continuous loop. For each frame you receive:
- sensor_data: current state from system, GUI, and CLI inputs
- frame_number: how many frames have run
- recent_actions: what actions were taken in the previous frame
- recent_errors: any errors from the previous frame

Your job: Decide what actions to take this frame.

Respond ONLY with a valid JSON object matching this schema:
{
  "reasoning": "brief explanation of your decision (1-2 sentences)",
  "actions": [
    {
      "type": "shell|apple_script|write_file|gui_click",
      "description": "what this action does",
      ... other action-specific fields
    }
  ],
  "observations": "what you notice in the environment (1-2 sentences)"
}

Rules:
- Keep actions minimal (max 3) per frame
- Prefer read-only actions unless something needs changing
- Use absolute paths for file operations
- If nothing needs doing, return {"reasoning": "...", "actions": [], "observations": "..."}
"""


    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_MODEL):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model

        if not self.api_key:
            logger.warning("ANTHROPIC_API_KEY not set — AI client will return no-op decisions")

        self._client = None

    def _get_client(self):
        if not self._client and self.api_key:
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def decide(self, frame_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point. Takes the world model snapshot and returns
        a structured decision dict.
        """
        if not self.api_key:
            logger.debug("No API key — returning no-op decision")
            return {
                "reasoning": "No AI client configured (ANTHROPIC_API_KEY not set)",
                "actions": [],
                "observations": "",
            }

        # Build the user message from frame context
        user_msg = self._build_message(frame_context)

        try:
            client = self._get_client()
            response = client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=self.SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )

            response_text = response.content[0].text.strip()

            # Parse JSON from response
            # Claude sometimes wraps JSON in ```json blocks
            if response_text.startswith("```"):
                # Strip markdown code fences
                lines = response_text.split("\n")
                response_text = "\n".join(lines[1:-1])  # Remove first and last line

            decision = json.loads(response_text)
            logger.info(f"AI decision: {decision.get('reasoning', '')[:80]}")
            return decision

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse AI JSON response: {e}\nRaw: {response_text[:500]}")
            return {
                "reasoning": f"Failed to parse AI response: {e}",
                "actions": [],
                "observations": "",
            }
        except Exception as e:
            logger.error(f"AI client error: {e}")
            return {
                "reasoning": f"AI client error: {e}",
                "actions": [],
                "observations": "",
            }

    def _build_message(self, ctx: Dict[str, Any]) -> str:
        """Build the user message from frame context."""
        sensors = ctx.get("sensors", {})
        system = sensors.get("system", {})
        gui = sensors.get("gui", {})
        cli = sensors.get("cli", {})

        lines = [
            f"Frame #{ctx.get('frame_number', 0)}",
            "",
            "System state:",
            f"  {system.get('cpu_raw', 'N/A')}",
            f"  {system.get('mem_raw', 'N/A')}",
            f"  Disk: {system.get('disk_use_pct', 'N/A')} used",
            "",
            "GUI state:",
            f"  Latest screenshot: {gui.get('latest', 'none')}",
            f"  Screenshot count this session: {len(gui.get('screenshots', []))}",
            "",
            "CLI input:",
            f"  Pending commands: {cli.get('pending_commands', [])}",
            f"  Recent log entries: {cli.get('log_entries', [])[-3:]}",
            "",
            "Recent actions (last frame):",
        ]

        last_actions = ctx.get("last_actions", [])
        if last_actions:
            for a in last_actions[-3:]:
                lines.append(f"  - {a}")
        else:
            lines.append("  (none yet)")

        last_errors = ctx.get("last_errors", [])
        if last_errors:
            lines.append(f"\nRecent errors: {last_errors[-3:]}")

        return "\n".join(lines)
