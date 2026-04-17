"""
AI Client — Anthropic Claude API integration.

Improvements over v0.1:
- Native tool-calling instead of fragile JSON-parsing
- Exponential backoff retry on transient API errors
- Prompt caching on the system prompt (reused every frame)
"""
import os
import time
import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional

import anthropic

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-4-6"


@dataclass
class RetryConfig:
    """Exponential backoff configuration for API calls."""
    max_attempts: int = 4
    base_delay: float = 1.0    # seconds before first retry
    max_delay: float = 60.0    # cap on any single sleep
    backoff_factor: float = 2.0


# Tool definitions that mirror ActorSuite's action types.
# Claude calls these tools; we translate the call into an action dict.
STRIDE_TOOLS = [
    {
        "name": "shell",
        "description": "Run a shell command and capture its output.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute (e.g. 'df -h /')",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (default 30)",
                    "default": 30,
                },
                "description": {
                    "type": "string",
                    "description": "Human-readable summary of what this command does",
                },
            },
            "required": ["command", "description"],
        },
    },
    {
        "name": "write_file",
        "description": "Write text content to a file on disk.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute file path to write",
                },
                "content": {
                    "type": "string",
                    "description": "Content to write",
                },
                "description": {
                    "type": "string",
                    "description": "Human-readable summary of what this write does",
                },
            },
            "required": ["path", "content", "description"],
        },
    },
    {
        "name": "apple_script",
        "description": "Run an AppleScript on macOS to control the GUI or apps.",
        "input_schema": {
            "type": "object",
            "properties": {
                "script": {
                    "type": "string",
                    "description": "AppleScript source code to execute",
                },
                "description": {
                    "type": "string",
                    "description": "Human-readable summary of what this script does",
                },
            },
            "required": ["script", "description"],
        },
    },
    {
        "name": "gui_click",
        "description": "Click at a specific (x, y) pixel coordinate on screen.",
        "input_schema": {
            "type": "object",
            "properties": {
                "x": {"type": "integer", "description": "X coordinate"},
                "y": {"type": "integer", "description": "Y coordinate"},
                "description": {
                    "type": "string",
                    "description": "Human-readable summary of what this click does",
                },
            },
            "required": ["x", "y", "description"],
        },
    },
    {
        "name": "no_op",
        "description": (
            "Take no action this frame. Use when the environment is stable "
            "and no intervention is needed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "observations": {
                    "type": "string",
                    "description": "What you observe about the current state",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Why no action is needed this frame",
                },
            },
            "required": ["observations", "reasoning"],
        },
    },
]


class AIClient:
    """
    Calls Anthropic Claude with:
    - Native tool-calling for reliable structured output
    - Exponential backoff retry on transient API errors
    - Prompt caching so the system prompt isn't billed on every frame
    """

    SYSTEM_PROMPT = """You are the decision engine for Stride, a real-time frame-based agent.

The agent runs in a continuous 60-second loop. Each frame you receive:
- sensor_data: current state from system, GUI, and CLI inputs
- frame_number: how many frames have run
- recent_actions: what the agent did last frame
- recent_errors: any errors from the previous frame

Your job: call one or more of the provided tools to act this frame.

Rules:
- Call at most 3 action tools per frame
- Prefer read-only shell commands unless something truly needs changing
- Use absolute paths for all file operations
- When uncertain about the environment, observe first (shell ls/cat) before writing
- If nothing needs doing, call no_op"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        retry: Optional[RetryConfig] = None,
    ):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model
        self.retry = retry or RetryConfig()

        if not self.api_key:
            logger.warning("ANTHROPIC_API_KEY not set — AI client will return no-op decisions")

        self._client: Optional[anthropic.Anthropic] = None

    def _get_client(self) -> anthropic.Anthropic:
        if not self._client:
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    # ── Retry logic ────────────────────────────────────────────────────────────

    def _is_retryable(self, exc: Exception) -> bool:
        """Return True for transient errors that warrant a retry."""
        return isinstance(
            exc,
            (
                anthropic.RateLimitError,
                anthropic.InternalServerError,
                anthropic.APIConnectionError,
                anthropic.APITimeoutError,
            ),
        )

    def _call_with_retry(self, **kwargs) -> anthropic.types.Message:
        """
        Call messages.create with exponential backoff.
        Raises RuntimeError after max_attempts, or re-raises non-retryable errors immediately.
        """
        cfg = self.retry
        attempt = 0
        last_exc: Optional[Exception] = None

        while attempt < cfg.max_attempts:
            try:
                return self._get_client().messages.create(**kwargs)
            except Exception as exc:
                if not self._is_retryable(exc):
                    raise
                last_exc = exc
                attempt += 1
                if attempt >= cfg.max_attempts:
                    break
                delay = min(cfg.base_delay * (cfg.backoff_factor ** (attempt - 1)), cfg.max_delay)
                logger.warning(
                    f"API error (attempt {attempt}/{cfg.max_attempts}): "
                    f"{type(exc).__name__} — retrying in {delay:.1f}s"
                )
                time.sleep(delay)

        raise RuntimeError(
            f"API call failed after {cfg.max_attempts} attempts"
        ) from last_exc

    # ── Main decision entry point ──────────────────────────────────────────────

    def decide(self, frame_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Take the world model snapshot and return a structured decision dict.
        Uses Claude's tool-calling API for reliable, schema-validated output.
        """
        if not self.api_key:
            logger.debug("No API key — returning no-op decision")
            return {
                "reasoning": "No AI client configured (ANTHROPIC_API_KEY not set)",
                "actions": [],
                "observations": "",
            }

        user_msg = self._build_message(frame_context)

        try:
            response = self._call_with_retry(
                model=self.model,
                max_tokens=1024,
                # System as a list enables prompt caching (ephemeral = 5-min TTL).
                # The large, static system prompt is cached; only the user message
                # is re-billed each frame.
                system=[
                    {
                        "type": "text",
                        "text": self.SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                tools=STRIDE_TOOLS,
                messages=[{"role": "user", "content": user_msg}],
            )

            decision = self._parse_tool_response(response)
            logger.info(
                f"AI decision: {decision['reasoning'][:80]} "
                f"({len(decision['actions'])} action(s))"
            )
            return decision

        except Exception as e:
            logger.error(f"AI client error: {e}")
            return {
                "reasoning": f"AI client error: {e}",
                "actions": [],
                "observations": "",
            }

    # ── Response parsing ───────────────────────────────────────────────────────

    def _parse_tool_response(self, response: anthropic.types.Message) -> Dict[str, Any]:
        """
        Extract tool calls from the API response and translate them into
        action dicts that ActorSuite understands.
        """
        actions = []
        reasoning_parts = []
        observations = ""

        for block in response.content:
            if block.type == "text":
                text = block.text.strip()
                if text:
                    reasoning_parts.append(text)
            elif block.type == "tool_use":
                tool_name = block.name
                inputs: Dict[str, Any] = block.input  # type: ignore[assignment]

                if tool_name == "no_op":
                    observations = inputs.get("observations", "")
                    reasoning_parts.append(inputs.get("reasoning", ""))
                else:
                    # Build an action dict compatible with ActorSuite.execute()
                    action = {"type": tool_name}
                    action.update(inputs)
                    actions.append(action)

        reasoning = " ".join(reasoning_parts).strip() or "No reasoning provided"
        return {
            "reasoning": reasoning,
            "actions": actions,
            "observations": observations,
        }

    # ── Message builder ────────────────────────────────────────────────────────

    def _build_message(self, ctx: Dict[str, Any]) -> str:
        """Serialise frame context into the user turn sent to Claude."""
        sensors = ctx.get("sensors", {})
        system = sensors.get("system", {})
        gui = sensors.get("gui", {})
        cli = sensors.get("cli", {})

        lines = [
            f"Frame #{ctx.get('frame_number', 0)}",
            "",
            "System state:",
            f"  CPU: {system.get('cpu_raw', 'N/A')}",
            f"  Mem: {system.get('mem_raw', 'N/A')}",
            f"  Disk: {system.get('disk_use_pct', 'N/A')} used",
            "",
            "GUI state:",
            f"  Latest screenshot: {gui.get('latest', 'none')}",
            f"  Screenshots this session: {len(gui.get('screenshots', []))}",
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

        lines.append("\nWhat should the agent do this frame? Call the appropriate tool(s).")
        return "\n".join(lines)
