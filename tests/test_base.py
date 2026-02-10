import json

# Add import for config testing
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from amok.base import BaseAgent
from amok.lib import AgentResponse, AgentSettings


class DummyAgent(BaseAgent):
    @classmethod
    def _get_settings_class(cls) -> type[AgentSettings]:
        return AgentSettings

    def compose_user_prompt(self) -> str:
        return "user prompt"

    def compose_system_prompt(self) -> str:
        return "system prompt"


class InvalidAgent(BaseAgent):
    """Agent with invalid settings class for testing."""

    @classmethod
    def _get_settings_class(cls) -> type:
        return dict  # Invalid - not a subclass of AgentSettings

    def compose_user_prompt(self) -> str:
        return "user prompt"

    def compose_system_prompt(self) -> str:
        return "system prompt"


def make_settings():
    return AgentSettings(
        base_url="http://localhost",
        model="gpt-test",
        api_key="sk-test",
        temperature=0.5,
        max_tokens=10,
        ssl_verify=True,
    )


def test_init_sets_attributes():
    settings = make_settings()
    agent = DummyAgent(settings)
    assert agent.openai_client is not None
    assert agent.model == settings.model
    assert agent.temperature == settings.temperature
    assert agent.max_tokens == settings.max_tokens
    assert agent.ssl_verify == settings.ssl_verify
    assert agent.stream is False


def test_init_rejects_invalid_api_mode():
    settings = make_settings()
    settings.api_mode = "response"
    with pytest.raises(ValueError, match="Unsupported api_mode"):
        DummyAgent(settings)


def test_init_normalizes_flat_function_tools_for_chat_mode():
    settings = make_settings()
    settings.tools = [
        {
            "type": "function",
            "name": "lookup_weather",
            "description": "Get weather",
            "parameters": {"type": "object", "properties": {}},
        },
    ]
    agent = DummyAgent(settings)
    assert agent.tools == [
        {
            "type": "function",
            "function": {
                "name": "lookup_weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]


def test_init_normalizes_nested_function_tools_for_responses_mode():
    settings = make_settings()
    settings.api_mode = "responses"
    settings.tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup_weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]
    agent = DummyAgent(settings)
    assert agent.tools == [
        {
            "type": "function",
            "name": "lookup_weather",
            "description": "Get weather",
            "parameters": {"type": "object", "properties": {}},
        },
    ]


def test_run_success(monkeypatch):
    settings = make_settings()
    agent = DummyAgent(settings)
    mock_completion = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "<think>think</think>response"
    mock_completion.choices = [mock_choice]
    agent.openai_client.chat.completions.create = MagicMock(
        return_value=mock_completion,
    )
    resp = agent.run(body="test body")
    assert isinstance(resp, AgentResponse)
    assert resp.thought == "think"
    assert resp.response == "response"
    agent.openai_client.chat.completions.create.assert_called_once_with(
        model=settings.model,
        messages=[
            {
                "role": "system",
                "content": "detailed thinking on\n<think>\nsystem prompt",
            },
            {"role": "user", "content": "user prompt\n<BODY>\ntest body\n</BODY>"},
        ],
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        stream=False,
    )


def test_run_success_with_chat_tool_calls():
    settings = make_settings()
    settings.tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup_weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]
    settings.tool_choice = "auto"
    settings.parallel_tool_calls = True
    agent = DummyAgent(settings)
    mock_completion = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = ""
    mock_choice.message.tool_calls = [
        SimpleNamespace(
            id="call_123",
            type="function",
            function=SimpleNamespace(
                name="lookup_weather",
                arguments='{"city":"NYC"}',
            ),
        ),
    ]
    mock_completion.choices = [mock_choice]
    agent.openai_client.chat.completions.create = MagicMock(
        return_value=mock_completion,
    )

    resp = agent.run(body="test body")

    assert resp.response == ""
    assert resp.thought is None
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0]["type"] == "function"
    assert resp.tool_calls[0]["id"] == "call_123"
    assert resp.tool_calls[0]["name"] == "lookup_weather"
    assert resp.tool_calls[0]["arguments"] == '{"city":"NYC"}'

    agent.openai_client.chat.completions.create.assert_called_once()
    call_kwargs = agent.openai_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["tools"] == settings.tools
    assert call_kwargs["tool_choice"] == "auto"
    assert call_kwargs["parallel_tool_calls"] is True


def test_run_success_with_chat_tool_calls_as_dict_items():
    settings = make_settings()
    agent = DummyAgent(settings)
    mock_completion = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = ""
    mock_choice.message.tool_calls = [
        {
            "id": "call_456",
            "type": "function",
            "function": {
                "name": "lookup_weather",
                "arguments": '{"city":"Boston"}',
            },
        },
    ]
    mock_completion.choices = [mock_choice]
    agent.openai_client.chat.completions.create = MagicMock(
        return_value=mock_completion,
    )

    resp = agent.run(body="test body")

    assert resp.response == ""
    assert resp.thought is None
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0]["id"] == "call_456"
    assert resp.tool_calls[0]["type"] == "function"
    assert resp.tool_calls[0]["name"] == "lookup_weather"
    assert resp.tool_calls[0]["arguments"] == '{"city":"Boston"}'


def test_run_success_no_body(monkeypatch):
    """Test run method when no body is provided."""
    settings = make_settings()
    agent = DummyAgent(settings)
    mock_completion = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "<think>think</think>response"
    mock_completion.choices = [mock_choice]
    agent.openai_client.chat.completions.create = MagicMock(
        return_value=mock_completion,
    )
    resp = agent.run(body=None)
    assert isinstance(resp, AgentResponse)
    assert resp.thought == "think"
    assert resp.response == "response"
    agent.openai_client.chat.completions.create.assert_called_once_with(
        model=settings.model,
        messages=[
            {
                "role": "system",
                "content": "detailed thinking on\n<think>\nsystem prompt",
            },
            {"role": "user", "content": "user prompt\n"},
        ],
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        stream=False,
    )


def test_run_success_responses_api():
    """Test run method when configured to use the Responses API."""
    settings = make_settings()
    settings.api_mode = "responses"
    agent = DummyAgent(settings)
    mock_response = MagicMock()
    mock_response.output_text = "<think>think</think>response"
    mock_response.output = []
    agent.openai_client.responses.create = MagicMock(return_value=mock_response)

    resp = agent.run(body="test body")

    assert isinstance(resp, AgentResponse)
    assert resp.thought == "think"
    assert resp.response == "response"
    assert resp.tool_calls == []
    agent.openai_client.responses.create.assert_called_once_with(
        model=settings.model,
        input=[
            {
                "role": "system",
                "content": "detailed thinking on\n<think>\nsystem prompt",
            },
            {"role": "user", "content": "user prompt\n<BODY>\ntest body\n</BODY>"},
        ],
        temperature=settings.temperature,
        max_output_tokens=settings.max_tokens,
        stream=False,
    )


def test_run_success_responses_api_tool_calls_only():
    """Test run method for Responses API with tool calls and no text output."""
    settings = make_settings()
    settings.api_mode = "responses"
    settings.tools = [
        {
            "type": "function",
            "name": "lookup_weather",
            "description": "Get weather",
            "parameters": {"type": "object", "properties": {}},
        },
    ]
    settings.tool_choice = "auto"
    settings.parallel_tool_calls = True
    agent = DummyAgent(settings)
    mock_response = MagicMock()
    mock_response.output_text = ""
    mock_response.output = [
        SimpleNamespace(
            type="function_call",
            id="fc_1",
            name="lookup_weather",
            arguments='{"city":"NYC"}',
            call_id="call_1",
            status="completed",
        ),
    ]
    agent.openai_client.responses.create = MagicMock(return_value=mock_response)

    resp = agent.run(body="test body")

    assert resp.response == ""
    assert resp.thought is None
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0]["type"] == "function_call"
    assert resp.tool_calls[0]["name"] == "lookup_weather"
    assert resp.tool_calls[0]["arguments"] == '{"city":"NYC"}'
    agent.openai_client.responses.create.assert_called_once()
    call_kwargs = agent.openai_client.responses.create.call_args.kwargs
    assert call_kwargs["tools"] == settings.tools
    assert call_kwargs["tool_choice"] == "auto"
    assert call_kwargs["parallel_tool_calls"] is True


def test_run_no_openai_client():
    settings = make_settings()
    agent = DummyAgent(settings)
    agent.openai_client = None
    with pytest.raises(ValueError, match="OpenAI client not initialized"):
        agent.run(body="test body")


def test_run_no_response(monkeypatch):
    settings = make_settings()
    agent = DummyAgent(settings)
    mock_completion = MagicMock()
    mock_completion.choices = []
    agent.openai_client.chat.completions.create = MagicMock(
        return_value=mock_completion,
    )
    with pytest.raises(ValueError, match="No response from the model"):
        agent.run(body="test body")


def test_run_no_completion():
    """Test run method when completion is None."""
    settings = make_settings()
    agent = DummyAgent(settings)
    agent.openai_client.chat.completions.create = MagicMock(return_value=None)
    with pytest.raises(ValueError, match="No response from the model"):
        agent.run(body="test body")


def test_run_no_message():
    """Test run method when completion has no message."""
    settings = make_settings()
    agent = DummyAgent(settings)
    mock_completion = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message = None
    mock_completion.choices = [mock_choice]
    agent.openai_client.chat.completions.create = MagicMock(
        return_value=mock_completion,
    )
    with pytest.raises(ValueError, match="No response from the model"):
        agent.run(body="test body")


def test_run_no_message_content():
    """Test run method when message has no content."""
    settings = make_settings()
    agent = DummyAgent(settings)
    mock_completion = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = None
    mock_completion.choices = [mock_choice]
    agent.openai_client.chat.completions.create = MagicMock(
        return_value=mock_completion,
    )
    with pytest.raises(ValueError, match="No response from the model"):
        agent.run(body="test body")


def test_parse_response_with_thought():
    content = "<think>foo</think>bar"
    response, thought = BaseAgent._parse_response(content)
    assert response == "bar"
    assert thought == "foo"


def test_parse_response_without_thought():
    content = "just response"
    response, thought = BaseAgent._parse_response(content)
    assert response == "just response"
    assert thought is None


def test_parse_response_multiple_thoughts():
    content = "<think>first</think>main<think>second</think>"
    response, thought = BaseAgent._parse_response(content)
    assert thought == "first"
    assert "second" in response


def test_parse_response_empty_content():
    """Test parsing response with empty content."""
    content = ""
    response, thought = BaseAgent._parse_response(content)
    assert response == ""
    assert thought is None


def test_parse_response_only_thought():
    """Test parsing response with only thought tags."""
    content = "<think>just thinking</think>"
    response, thought = BaseAgent._parse_response(content)
    assert response == ""
    assert thought == "just thinking"


def test_parse_response_thought_with_newlines():
    """Test parsing response with thought containing newlines."""
    content = "<think>multi\nline\nthought</think>response content"
    response, thought = BaseAgent._parse_response(content)
    assert response == "response content"
    assert thought == "multi\nline\nthought"


def test_parse_response_nested_tags():
    """Test parsing response with nested or similar tags."""
    content = "before <think>thinking about <other>tags</other></think> after"
    response, thought = BaseAgent._parse_response(content)
    assert response == "before  after"
    assert thought == "thinking about <other>tags</other>"


def test_read_cfg_success():
    """Test successful configuration reading from file."""
    config_data = {
        "base_url": "http://test.com",
        "model": "test-model",
        "api_key": "test-key",
        "temperature": 0.8,
        "max_tokens": 2000,
        "ssl_verify": False,
        "thinking_mode": False,
        "extra_field": "should_be_ignored",  # This should be filtered out
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config_data, f)
        f.flush()

        agent = DummyAgent.from_cfg(f.name)

        assert agent.model == "test-model"
        assert agent.temperature == 0.8
        assert agent.max_tokens == 2000
        assert agent.ssl_verify is False
        assert agent.is_thinking_agent is False
        assert agent.openai_client is not None


def test_validated_settings_success():
    """Test successful settings validation."""
    settings = {
        "base_url": "http://test.com",
        "model": "test-model",
        "api_key": "test-key",
        "temperature": 0.5,
        "max_tokens": 1500,
        "ssl_verify": True,
        "thinking_mode": True,
        "api_mode": "responses",
        "tools": [{"type": "function", "name": "lookup_weather"}],
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "extra_field": "should_be_ignored",
    }

    validated = DummyAgent.validated_settings(settings)

    # Should only include fields that are in AgentSettings
    expected_fields = {
        "base_url",
        "model",
        "api_key",
        "temperature",
        "max_tokens",
        "ssl_verify",
        "thinking_mode",
        "api_mode",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
    }
    assert set(validated.keys()) == expected_fields
    assert validated["base_url"] == "http://test.com"
    assert validated["model"] == "test-model"
    assert "extra_field" not in validated


def test_validated_settings_invalid_class():
    """Test validation with invalid settings class."""
    settings = {"base_url": "http://test.com", "model": "test-model"}

    with pytest.raises(
        TypeError,
        match="Settings class dict must inherit from AgentSettings",
    ):
        InvalidAgent.validated_settings(settings)


def test_validated_settings_partial_fields():
    """Test validation with only some fields present."""
    settings = {
        "base_url": "http://test.com",
        "model": "test-model",
        "unknown_field": "value",
    }

    validated = DummyAgent.validated_settings(settings)

    assert validated == {"base_url": "http://test.com", "model": "test-model"}


def test_validated_settings_empty():
    """Test validation with empty settings."""
    settings = {}

    validated = DummyAgent.validated_settings(settings)

    assert validated == {}
