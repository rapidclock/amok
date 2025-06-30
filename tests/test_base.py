from unittest.mock import MagicMock

import pytest

from amok.base import BaseAgent
from amok.lib import AgentResponse, AgentSettings


class DummyAgent(BaseAgent):
    def read_cfg(self) -> None:
        pass

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


def test_run_success(monkeypatch):
    settings = make_settings()
    agent = DummyAgent(settings)
    mock_completion = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "<thought>think</thought>response"
    mock_completion.choices = [mock_choice]
    agent.openai_client.chat.completions.create = MagicMock(
        return_value=mock_completion
    )
    resp = agent.run()
    assert isinstance(resp, AgentResponse)
    assert resp.thought == "think"
    assert resp.response == "response"


def test_run_no_openai_client():
    settings = make_settings()
    agent = DummyAgent(settings)
    agent.openai_client = None
    with pytest.raises(ValueError, match="OpenAI client not initialized"):
        agent.run()


def test_run_no_response(monkeypatch):
    settings = make_settings()
    agent = DummyAgent(settings)
    mock_completion = MagicMock()
    mock_completion.choices = []
    agent.openai_client.chat.completions.create = MagicMock(
        return_value=mock_completion
    )
    with pytest.raises(ValueError, match="No response from the model"):
        agent.run()


def test_parse_response_with_thought():
    content = "<thought>foo</thought>bar"
    response, thought = BaseAgent._parse_response(content)
    assert response == "bar"
    assert thought == "foo"


def test_parse_response_without_thought():
    content = "just response"
    response, thought = BaseAgent._parse_response(content)
    assert response == "just response"
    assert thought is None


def test_parse_response_multiple_thoughts():
    content = "<thought>first</thought>main<thought>second</thought>"
    response, thought = BaseAgent._parse_response(content)
    assert thought == "first"
    assert "second" in response
