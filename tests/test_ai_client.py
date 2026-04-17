"""
Tests for the improved AI client:
- Tool-calling response parsing
- Exponential backoff retry logic
- Tool schema structure validation
- No-API-key fallback
"""
import os
import pytest
from unittest.mock import MagicMock, patch

import anthropic

from stride.ai import AIClient, RetryConfig, STRIDE_TOOLS


# ─── Mock helpers ─────────────────────────────────────────────────────────────

def _tool_block(name: str, inputs: dict):
    """Build a fake ToolUseBlock."""
    block = MagicMock()
    block.type = "tool_use"
    block.name = name
    block.input = inputs
    return block


def _text_block(text: str):
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _response(*blocks):
    """Wrap blocks in a fake Message."""
    msg = MagicMock()
    msg.content = list(blocks)
    return msg


def _make_ai(max_attempts=3, base_delay=0.01):
    """Return an AIClient wired with a fast retry config."""
    return AIClient(
        api_key="dummy-key",
        retry=RetryConfig(max_attempts=max_attempts, base_delay=base_delay, backoff_factor=2.0),
    )


def _inject_mock_client(ai: AIClient, side_effects):
    """
    Replace the Anthropic client inside `ai` with a mock whose
    messages.create raises/returns `side_effects` in sequence.
    """
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = side_effects
    ai._client = mock_client
    return mock_client


# ─── _parse_tool_response ─────────────────────────────────────────────────────

class TestParseToolResponse:
    def setup_method(self):
        self.ai = _make_ai()

    def test_single_shell_action(self):
        resp = _response(_tool_block("shell", {"command": "ls /", "description": "list root"}))
        decision = self.ai._parse_tool_response(resp)
        assert decision["actions"] == [
            {"type": "shell", "command": "ls /", "description": "list root"}
        ]

    def test_no_op_produces_empty_actions(self):
        resp = _response(
            _tool_block("no_op", {"observations": "all quiet", "reasoning": "nothing to do"})
        )
        decision = self.ai._parse_tool_response(resp)
        assert decision["actions"] == []
        assert decision["observations"] == "all quiet"
        assert "nothing to do" in decision["reasoning"]

    def test_multiple_actions_preserved_in_order(self):
        resp = _response(
            _text_block("Checking disk then logging result."),
            _tool_block("shell", {"command": "df -h", "description": "disk usage"}),
            _tool_block(
                "write_file",
                {"path": "/tmp/note.txt", "content": "ok", "description": "save note"},
            ),
        )
        decision = self.ai._parse_tool_response(resp)
        assert len(decision["actions"]) == 2
        assert decision["actions"][0]["type"] == "shell"
        assert decision["actions"][1]["type"] == "write_file"
        assert "Checking disk" in decision["reasoning"]

    def test_text_only_response_goes_to_reasoning(self):
        resp = _response(_text_block("The system appears idle."))
        decision = self.ai._parse_tool_response(resp)
        assert decision["actions"] == []
        assert "idle" in decision["reasoning"]

    def test_empty_response_gives_fallback_reasoning(self):
        resp = _response()
        decision = self.ai._parse_tool_response(resp)
        assert decision["actions"] == []
        assert decision["reasoning"] == "No reasoning provided"

    def test_action_dict_includes_all_tool_inputs(self):
        resp = _response(
            _tool_block(
                "write_file",
                {"path": "/tmp/x.txt", "content": "hello", "description": "write x"},
            )
        )
        action = self.ai._parse_tool_response(resp)["actions"][0]
        assert action["type"] == "write_file"
        assert action["path"] == "/tmp/x.txt"
        assert action["content"] == "hello"
        assert action["description"] == "write x"

    def test_apple_script_action(self):
        resp = _response(
            _tool_block("apple_script", {"script": 'say "hello"', "description": "greet"})
        )
        decision = self.ai._parse_tool_response(resp)
        assert decision["actions"][0]["type"] == "apple_script"

    def test_gui_click_action(self):
        resp = _response(
            _tool_block("gui_click", {"x": 100, "y": 200, "description": "click button"})
        )
        decision = self.ai._parse_tool_response(resp)
        assert decision["actions"][0]["x"] == 100
        assert decision["actions"][0]["y"] == 200


# ─── _call_with_retry ─────────────────────────────────────────────────────────

class TestRetryLogic:
    def setup_method(self):
        self.ai = _make_ai(max_attempts=3, base_delay=0.01)

    def _rate_limit_error(self):
        return anthropic.RateLimitError(
            message="rate limited",
            response=MagicMock(status_code=429),
            body={},
        )

    def _internal_error(self):
        return anthropic.InternalServerError(
            message="server error",
            response=MagicMock(status_code=500),
            body={},
        )

    def _auth_error(self):
        return anthropic.AuthenticationError(
            message="bad key",
            response=MagicMock(status_code=401),
            body={},
        )

    def _connection_error(self):
        return anthropic.APIConnectionError(request=MagicMock())

    def test_succeeds_on_first_attempt(self):
        success = _response(_tool_block("no_op", {"observations": "ok", "reasoning": "fine"}))
        _inject_mock_client(self.ai, [success])
        result = self.ai._call_with_retry(
            model="m", max_tokens=10, system="s", tools=[], messages=[]
        )
        assert result is success

    def test_retries_on_rate_limit_then_succeeds(self):
        success = _response(_tool_block("no_op", {"observations": "ok", "reasoning": "fine"}))
        mock = _inject_mock_client(self.ai, [self._rate_limit_error(), success])
        result = self.ai._call_with_retry(
            model="m", max_tokens=10, system="s", tools=[], messages=[]
        )
        assert result is success
        assert mock.messages.create.call_count == 2

    def test_retries_on_internal_server_error(self):
        success = _response(_tool_block("no_op", {"observations": "ok", "reasoning": "fine"}))
        mock = _inject_mock_client(self.ai, [self._internal_error(), success])
        result = self.ai._call_with_retry(
            model="m", max_tokens=10, system="s", tools=[], messages=[]
        )
        assert result is success
        assert mock.messages.create.call_count == 2

    def test_retries_on_connection_error(self):
        success = _response(_tool_block("no_op", {"observations": "ok", "reasoning": "fine"}))
        mock = _inject_mock_client(self.ai, [self._connection_error(), success])
        result = self.ai._call_with_retry(
            model="m", max_tokens=10, system="s", tools=[], messages=[]
        )
        assert result is success
        assert mock.messages.create.call_count == 2

    def test_raises_runtime_error_after_all_attempts_exhausted(self):
        _inject_mock_client(
            self.ai,
            [self._rate_limit_error(), self._rate_limit_error(), self._rate_limit_error()],
        )
        with pytest.raises(RuntimeError, match="failed after 3 attempts"):
            self.ai._call_with_retry(
                model="m", max_tokens=10, system="s", tools=[], messages=[]
            )

    def test_does_not_retry_auth_error(self):
        mock = _inject_mock_client(self.ai, [self._auth_error()])
        with pytest.raises(anthropic.AuthenticationError):
            self.ai._call_with_retry(
                model="m", max_tokens=10, system="s", tools=[], messages=[]
            )
        # Only one attempt — auth errors are not retryable
        assert mock.messages.create.call_count == 1

    def test_does_not_retry_bad_request(self):
        bad_req = anthropic.BadRequestError(
            message="bad request",
            response=MagicMock(status_code=400),
            body={},
        )
        mock = _inject_mock_client(self.ai, [bad_req])
        with pytest.raises(anthropic.BadRequestError):
            self.ai._call_with_retry(
                model="m", max_tokens=10, system="s", tools=[], messages=[]
            )
        assert mock.messages.create.call_count == 1

    def test_decide_returns_error_dict_when_all_retries_fail(self):
        """decide() should catch exhausted retries and return a safe error dict."""
        _inject_mock_client(
            self.ai,
            [self._rate_limit_error(), self._rate_limit_error(), self._rate_limit_error()],
        )
        decision = self.ai.decide({"frame_number": 42})
        assert decision["actions"] == []
        assert "error" in decision["reasoning"].lower()


# ─── Tool schema structure ────────────────────────────────────────────────────

class TestToolSchemas:
    EXPECTED_NAMES = {"shell", "write_file", "apple_script", "gui_click", "no_op"}

    def test_all_expected_tools_present(self):
        names = {t["name"] for t in STRIDE_TOOLS}
        assert names == self.EXPECTED_NAMES

    def test_each_tool_has_required_top_level_keys(self):
        for tool in STRIDE_TOOLS:
            assert "name" in tool, f"Missing 'name' in {tool}"
            assert "description" in tool, f"Missing 'description' in {tool}"
            assert "input_schema" in tool, f"Missing 'input_schema' in {tool}"

    def test_each_schema_has_required_fields(self):
        for tool in STRIDE_TOOLS:
            schema = tool["input_schema"]
            assert schema["type"] == "object", f"{tool['name']} schema type must be 'object'"
            assert "properties" in schema, f"{tool['name']} schema missing 'properties'"
            assert "required" in schema, f"{tool['name']} schema missing 'required'"

    def test_all_required_fields_are_in_properties(self):
        for tool in STRIDE_TOOLS:
            schema = tool["input_schema"]
            for req in schema["required"]:
                assert req in schema["properties"], (
                    f"{tool['name']}: required field '{req}' not in properties"
                )

    def test_no_op_requires_observations_and_reasoning(self):
        no_op = next(t for t in STRIDE_TOOLS if t["name"] == "no_op")
        assert "observations" in no_op["input_schema"]["required"]
        assert "reasoning" in no_op["input_schema"]["required"]

    def test_shell_requires_command_and_description(self):
        shell = next(t for t in STRIDE_TOOLS if t["name"] == "shell")
        required = shell["input_schema"]["required"]
        assert "command" in required
        assert "description" in required


# ─── No-API-key fallback (regression) ────────────────────────────────────────

def test_no_api_key_returns_noop():
    old_key = os.environ.pop("ANTHROPIC_API_KEY", None)
    try:
        ai = AIClient(api_key=None)
        decision = ai.decide({"frame_number": 1})
        assert decision["actions"] == []
        assert "not set" in decision["reasoning"].lower()
    finally:
        if old_key:
            os.environ["ANTHROPIC_API_KEY"] = old_key


# ─── is_retryable classification ──────────────────────────────────────────────

class TestIsRetryable:
    def setup_method(self):
        self.ai = _make_ai()

    def test_rate_limit_is_retryable(self):
        exc = anthropic.RateLimitError(
            message="429", response=MagicMock(status_code=429), body={}
        )
        assert self.ai._is_retryable(exc)

    def test_internal_server_error_is_retryable(self):
        exc = anthropic.InternalServerError(
            message="500", response=MagicMock(status_code=500), body={}
        )
        assert self.ai._is_retryable(exc)

    def test_connection_error_is_retryable(self):
        exc = anthropic.APIConnectionError(request=MagicMock())
        assert self.ai._is_retryable(exc)

    def test_timeout_error_is_retryable(self):
        exc = anthropic.APITimeoutError(request=MagicMock())
        assert self.ai._is_retryable(exc)

    def test_auth_error_is_not_retryable(self):
        exc = anthropic.AuthenticationError(
            message="401", response=MagicMock(status_code=401), body={}
        )
        assert not self.ai._is_retryable(exc)

    def test_bad_request_is_not_retryable(self):
        exc = anthropic.BadRequestError(
            message="400", response=MagicMock(status_code=400), body={}
        )
        assert not self.ai._is_retryable(exc)

    def test_arbitrary_exception_is_not_retryable(self):
        assert not self.ai._is_retryable(ValueError("oops"))
