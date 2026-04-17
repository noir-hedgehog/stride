"""
AI Client — Multi-provider Claude API integration (Anthropic + MiniMax).

Usage:
    ai = AIClient(provider="minimax")           # uses MINIMAX_API_KEY env
    ai = AIClient(provider="anthropic")         # uses ANTHROPIC_API_KEY env
    ai = AIClient(provider="minimax", model="MiniMax-M2.7")
"""
import os
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Literal

import anthropic

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-4-6"
MINIMAX_MODELS = ["MiniMax-M2.7", "MiniMax-M2", "MiniMax-Text-01"]


@dataclass
class RetryConfig:
    max_attempts: int = 4
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_factor: float = 2.0


STRIDE_TOOLS = [
    {
        "name": "shell",
        "description": "Run a shell command and capture its output.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to execute"},
                "timeout": {"type": "integer", "description": "Timeout in seconds (default 30)", "default": 30},
                "description": {"type": "string", "description": "Human-readable summary"},
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
                "path": {"type": "string", "description": "Absolute file path"},
                "content": {"type": "string", "description": "Content to write"},
                "description": {"type": "string", "description": "Human-readable summary"},
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
                "script": {"type": "string", "description": "AppleScript source code"},
                "description": {"type": "string", "description": "Human-readable summary"},
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
                "x": {"type": "integer"},
                "y": {"type": "integer"},
                "description": {"type": "string", "description": "Human-readable summary"},
            },
            "required": ["x", "y", "description"],
        },
    },
    {
        "name": "no_op",
        "description": "Take no action this frame. Use when the environment is stable.",
        "input_schema": {
            "type": "object",
            "properties": {
                "observations": {"type": "string"},
                "reasoning": {"type": "string"},
            },
            "required": ["observations", "reasoning"],
        },
    },
]


SYSTEM_PROMPT = """You are the decision engine for Stride, a real-time frame-based agent.

The agent runs in a continuous 60-second loop. Each frame you receive:
- sensor_data: current state from system, GUI, and CLI inputs
- frame_number: how many frames have run
- recent_actions: what the agent did last frame
- recent_errors: any errors from the previous frame

Your job: decide what to do this frame and respond with ONLY valid JSON.

Output format — respond with this exact JSON structure, no other text:
{"reasoning": "why you chose these actions", "actions": [{"type": "shell", "command": "ls /tmp", "description": "check temp dir"}], "observations": "what you noticed"}

Action types:
- shell: run a shell command (add 'command' and 'description')
- write_file: write a file (add 'path', 'content', 'description')
- apple_script: run AppleScript on macOS (add 'script', 'description')
- gui_click: click screen coordinate (add 'x', 'y', 'description')
- no_op: do nothing this frame (add 'reasoning' and 'observations')

Rules:
- Respond with ONLY JSON — no markdown, no explanations, no thinking text
- Call at most 3 actions per frame
- Prefer read-only shell commands unless something needs changing
- Use absolute paths for all file operations
- When uncertain, observe first (shell ls/cat) before writing
- If nothing needs doing, use no_op"""


class AIClient:
    """
    Multi-provider AI client. Supports:
    - anthropic: Anthropic's Claude API (default model: claude-sonnet-4-6)
    - minimax: MiniMax's OpenAI-compatible API (default model: MiniMax-M2.7)
    """

    PROVIDERS = {"anthropic", "minimax"}

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        provider: Literal["anthropic", "minimax"] = "anthropic",
        retry: Optional[RetryConfig] = None,
    ):
        self.provider = provider
        self.retry = retry or RetryConfig()

        if api_key:
            self._api_key = api_key
        elif provider == "minimax":
            self._api_key = os.environ.get("MINIMAX_API_KEY")
        else:
            self._api_key = os.environ.get("ANTHROPIC_API_KEY")

        if not self._api_key:
            logger.warning(f"[AIClient] {provider.upper()}_API_KEY not set — returning no-op decisions")

        # Resolve model
        if model:
            self.model = model
        elif provider == "minimax":
            self.model = "MiniMax-M2.7"
        else:
            self.model = DEFAULT_MODEL

        self._anthropic: Optional[anthropic.Anthropic] = None
        self._openai_client = None

    # ── Test compatibility aliases ────────────────────────────────────────────

    @property
    def _client(self):
        """Alias for _anthropic for backward test compatibility."""
        return self._anthropic

    @_client.setter
    def _client(self, value):
        """Allow setting _anthropic via _client for test compatibility."""
        self._anthropic = value

    def _is_retryable(self, exc: Exception) -> bool:
        """Alias for _is_retryable_anthropic for test compatibility."""
        return self._is_retryable_anthropic(exc)

    def _call_with_retry(self, **kwargs):
        """Alias for _call_anthropic for test compatibility."""
        return self._call_anthropic(**kwargs)

    def _parse_tool_response(self, response):
        """Alias for _parse_anthropic_response for test compatibility."""
        return self._parse_anthropic_response(response)

    # ── Anthropic ─────────────────────────────────────────────────────────────

    def _is_retryable_anthropic(self, exc: Exception) -> bool:
        return isinstance(exc, (
            anthropic.RateLimitError,
            anthropic.InternalServerError,
            anthropic.APIConnectionError,
            anthropic.APITimeoutError,
        ))

    def _call_anthropic(self, **kwargs):
        """Call Anthropic with retry, return Message."""
        cfg = self.retry
        attempt = 0
        last_exc: Optional[Exception] = None

        while attempt < cfg.max_attempts:
            try:
                client = self._get_anthropic()
                return client.messages.create(**kwargs)
            except Exception as exc:
                if not self._is_retryable_anthropic(exc):
                    raise
                last_exc = exc
                attempt += 1
                if attempt >= cfg.max_attempts:
                    break
                delay = min(cfg.base_delay * (cfg.backoff_factor ** (attempt - 1)), cfg.max_delay)
                logger.warning(f"[Anthropic] {type(exc).__name__} — retry {attempt}/{cfg.max_attempts} in {delay:.1f}s")
                time.sleep(delay)
        raise RuntimeError(f"Anthropic API failed after {cfg.max_attempts} attempts") from last_exc

    def _get_anthropic(self):
        if not self._anthropic:
            self._anthropic = anthropic.Anthropic(api_key=self._api_key)
        return self._anthropic

    # ── MiniMax (OpenAI-compatible) ───────────────────────────────────────────

    def _get_openai(self):
        """Lazy-init OpenAI client for MiniMax."""
        if self._openai_client is None:
            try:
                import openai
            except ImportError:
                raise RuntimeError("MiniMax provider requires `pip install openai`")
            self._openai_client = openai.OpenAI(
                api_key=self._api_key,
                base_url=os.environ.get("MINIMAX_BASE_URL", "https://api.minimaxi.com/v1"),
            )
        return self._openai_client

    def _tools_to_functions(self):
        """Convert STRIDE_TOOLS (OpenAI tool format) to functions format for MiniMax."""
        return [
            {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            }
            for t in STRIDE_TOOLS
        ]

    def _call_minimax(self, system: str, user_msg: str):
        """Call MiniMax API. Function calling doesn't work on this API key, so we
        rely on JSON-in-content parsing as the output format."""
        cfg = self.retry
        attempt = 0
        last_exc: Optional[Exception] = None

        while attempt < cfg.max_attempts:
            try:
                client = self._get_openai()
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system + "\n\nIMPORTANT: Respond ONLY with valid JSON. No explanations, no thinking text, no markdown. Your entire response must be parseable as a JSON object."},
                        {"role": "user", "content": user_msg},
                    ],
                )
                return response
            except Exception as exc:
                retryable = False
                try:
                    import openai as _oa
                    retryable = isinstance(exc, (_oa.RateLimitError, _oa.APIConnectionError, _oa.APITimeoutError))
                except ImportError:
                    pass

                if not retryable:
                    raise
                last_exc = exc
                attempt += 1
                if attempt >= cfg.max_attempts:
                    break
                delay = min(cfg.base_delay * (cfg.backoff_factor ** (attempt - 1)), cfg.max_delay)
                logger.warning(f"[MiniMax] {type(exc).__name__} — retry {attempt}/{cfg.max_attempts} in {delay:.1f}s")
                time.sleep(delay)
        raise RuntimeError(f"MiniMax API failed after {cfg.max_attempts} attempts") from last_exc

    # ── decide() — main entry point ────────────────────────────────────────────

    def decide(self, frame_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Take world model snapshot, call AI, return structured decision.
        """
        if not self._api_key:
            return {
                "reasoning": f"API key not set for {self.provider} — no-op",
                "actions": [],
                "observations": "",
            }

        user_msg = self._build_message(frame_context)

        try:
            if self.provider == "minimax":
                response = self._call_minimax(system=SYSTEM_PROMPT, user_msg=user_msg)
                decision = self._parse_openai_response(response)
            else:
                response = self._call_anthropic(
                    model=self.model,
                    max_tokens=1024,
                    system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
                    tools=STRIDE_TOOLS,
                    messages=[{"role": "user", "content": user_msg}],
                )
                decision = self._parse_anthropic_response(response)

            logger.info(f"[{self.provider}] decision: {decision['reasoning'][:60]} ({len(decision['actions'])} actions)")
            return decision

        except Exception as e:
            logger.error(f"[{self.provider}] decide error: {e}")
            return {"reasoning": f"Error: {e}", "actions": [], "observations": ""}

    # ── Response parsing ─────────────────────────────────────────────────────

    def _parse_anthropic_response(self, response) -> Dict[str, Any]:
        actions, reasoning_parts, observations = [], [], ""
        for block in response.content:
            if block.type == "text":
                if block.text.strip():
                    reasoning_parts.append(block.text.strip())
            elif block.type == "tool_use":
                inputs = block.input
                if block.name == "no_op":
                    observations = inputs.get("observations", "")
                    reasoning_parts.append(inputs.get("reasoning", ""))
                else:
                    action = {"type": block.name}
                    action.update(inputs)
                    actions.append(action)
        return {
            "reasoning": " ".join(reasoning_parts).strip() or "No reasoning provided",
            "actions": actions,
            "observations": observations,
        }

    def _strip_thinking(self, text: str) -> str:
        """
        Strip MiniMax's thinking block from content.
        MiniMax always puts the actual response at the END of content,
        after a reasoning/thinking block. The thinking block ends with various
        phrases ("So comply.", "Thus we respond with JSON.", etc.).

        Strategy: find the last '{' and last '}' and extract JSON from there.
        Fallback: strip known thinking suffixes.
        """
        text = text.strip()

        # Strategy 1: find the last JSON object (last '{' to last '}')
        last_brace = text.rfind('}')
        first_brace = text.rfind('{')
        if last_brace > first_brace and first_brace != -1:
            candidate = text[first_brace:last_brace+1]
            # Verify it looks like JSON (has common JSON keys)
            import json
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict) and any(k in parsed for k in ("reasoning", "actions", "observations")):
                    return candidate
            except Exception:
                pass

        # Strategy 2: strip thinking suffixes
        suffixes = [
            "So comply.", "So, comply.", "Thus, comply.",
            "Thus we respond with JSON.", "Thus final output:",
            "Therefore, comply.", "Hence, comply.",
        ]
        last_pos = -1
        for s in suffixes:
            idx = text.rfind(s)
            if idx != -1 and idx > last_pos:
                last_pos = idx + len(s)
        if last_pos > 0:
            text = text[last_pos:].strip()

        return text

    def _extract_json_from_content(self, content: str) -> str:
        """
        Extract the first complete JSON object from content using brace matching.
        Handles nested objects and strings with escaped characters.
        """
        depth = 0
        start = -1
        in_string = False
        escape_next = False

        for i, c in enumerate(content):
            if escape_next:
                escape_next = False
                continue
            if c == '\\' and in_string:
                escape_next = True
                continue
            if c == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == '{':
                if start == -1:
                    start = i
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0 and start != -1:
                    return content[start:i+1]

        # Fallback: return content stripped
        return content.strip()

    def _parse_openai_response(self, response) -> Dict[str, Any]:
        """
        Parse MiniMax/OpenAI response.
        Handles both OpenAI tool_calls format and MiniMax function_call format.
        MiniMax prepends a thinking block to content — strip it first.
        """
        actions, reasoning_parts, observations = [], [], ""
        import json

        for choice in response.choices:
            msg = choice.message
            raw_content = getattr(msg, "content", "") or ""

            # Handle MiniMax function_call format (functions API)
            fc = getattr(msg, "function_call", None)
            if fc:
                name = fc.name if hasattr(fc, "name") else (fc.get("name") if isinstance(fc, dict) else "")
                arguments = fc.arguments if hasattr(fc, "arguments") else (fc.get("arguments") if isinstance(fc, dict) else "{}")
                try:
                    args = json.loads(arguments) if isinstance(arguments, str) else arguments
                except Exception:
                    args = {}
                if name == "no_op":
                    observations = args.get("observations", "")
                    reasoning_parts.append(args.get("reasoning", ""))
                elif name:
                    action = {"type": name}
                    action.update(args)
                    actions.append(action)

            # Handle OpenAI tool_calls format
            tcs = getattr(msg, "tool_calls", None)
            if tcs:
                for tc in tcs:
                    fn = tc.function if hasattr(tc, "function") else tc
                    name = fn.name if hasattr(fn, "name") else ""
                    arguments = fn.arguments if hasattr(fn, "arguments") else "{}"
                    try:
                        args = json.loads(arguments) if isinstance(arguments, str) else arguments
                    except Exception:
                        args = {}
                    if name == "no_op":
                        observations = args.get("observations", "")
                        reasoning_parts.append(args.get("reasoning", ""))
                    elif name:
                        action = {"type": name}
                        action.update(args)
                        actions.append(action)

            # No function calls — parse content as JSON or free text
            has_calls = (fc is not None) or (tcs is not None)
            if not has_calls and raw_content:
                cleaned = self._strip_thinking(raw_content)
                json_str = self._extract_json_from_content(cleaned)
                try:
                    parsed = json.loads(json_str)
                    # Check if it looks like a Stride decision response
                    if isinstance(parsed, dict):
                        # Try Stride schema first
                        if "reasoning" in parsed or "actions" in parsed:
                            reasoning_parts.append(str(parsed.get("reasoning", "")))
                            for a in parsed.get("actions", []):
                                if isinstance(a, dict) and "type" in a:
                                    actions.append(a)
                            observations = str(parsed.get("observations", observations) or observations)
                        else:
                            # MiniMax returned non-Stride JSON — treat entire JSON as reasoning
                            # and look for any action-like keys
                            reasoning_parts.append(json_str[:500])
                            for k, v in parsed.items():
                                if k in ("action", "cmd", "command", "type") and v:
                                    if isinstance(v, str):
                                        actions.append({"type": "shell", "command": v, "description": k})
                                    elif isinstance(v, dict):
                                        actions.append({"type": k, **v})
                    else:
                        reasoning_parts.append(str(parsed)[:200])
                except Exception:
                    if cleaned.strip():
                        reasoning_parts.append(cleaned.strip()[:500])

        return {
            "reasoning": " ".join(reasoning_parts).strip() or "No reasoning provided",
            "actions": actions,
            "observations": observations,
        }

    # ── Message builder ───────────────────────────────────────────────────────

    def _build_message(self, ctx: Dict[str, Any]) -> str:
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

        for a in ctx.get("last_actions", [])[-3:]:
            lines.append(f"  - {a}")
        if not ctx.get("last_actions"):
            lines.append("  (none yet)")

        errors = ctx.get("last_errors", [])
        if errors:
            lines.append(f"\nRecent errors: {errors[-3:]}")

        lines.append("\nWhat should the agent do this frame? Call the appropriate tool(s).")
        return "\n".join(lines)
