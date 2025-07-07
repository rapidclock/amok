"""Tests for the action module."""

from unittest.mock import MagicMock

from amok.action import ACTION_SYSTEM_PROMPT, ANTI_INJECTION_WARNING, ActionAgent
from amok.lib import ActionAgentSettings, AgentResponse


def make_action_settings(**kwargs):
    """Create ActionAgentSettings with default values."""
    defaults = {
        "base_url": "http://localhost",
        "model": "gpt-test",
        "api_key": "sk-test",
        "temperature": 0.5,
        "max_tokens": 10,
        "ssl_verify": True,
        "thinking_mode": True,
        "description": "Test description",
        "commands": ["command1", "command2"],
    }
    defaults.update(kwargs)
    return ActionAgentSettings(**defaults)


def test_action_agent_init():
    """Test ActionAgent initialization."""
    settings = make_action_settings()
    agent = ActionAgent(settings)

    # ActionAgent doesn't have thinking_mode attribute,
    # but inherits is_thinking_agent from BaseAgent
    assert agent.is_thinking_agent is True
    assert agent.description == "Test description"
    assert "command1" in agent.commands
    assert "command2" in agent.commands
    assert ANTI_INJECTION_WARNING in agent.commands
    assert agent.openai_client is not None
    assert agent.model == settings.model
    assert agent.temperature == settings.temperature


def test_action_agent_get_settings_class():
    """Test ActionAgent _get_settings_class method."""
    settings_class = ActionAgent._get_settings_class()
    assert settings_class == ActionAgentSettings


def test_action_agent_init_no_commands():
    """Test ActionAgent initialization with no commands."""
    settings = make_action_settings(commands=None)
    agent = ActionAgent(settings)

    assert agent.commands == [ANTI_INJECTION_WARNING]


def test_action_agent_init_empty_commands():
    """Test ActionAgent initialization with empty commands list."""
    settings = make_action_settings(commands=[])
    agent = ActionAgent(settings)

    assert agent.commands == [ANTI_INJECTION_WARNING]


def test_action_agent_init_thinking_mode_false():
    """Test ActionAgent initialization with thinking mode disabled."""
    settings = make_action_settings(thinking_mode=False)
    agent = ActionAgent(settings)

    assert agent.is_thinking_agent is False


def test_action_agent_init_no_description():
    """Test ActionAgent initialization with no description."""
    settings = make_action_settings(description=None)
    agent = ActionAgent(settings)

    assert agent.description is None


def test_action_agent_init_empty_description():
    """Test ActionAgent initialization with empty description."""
    settings = make_action_settings(description="")
    agent = ActionAgent(settings)

    assert agent.description == ""


def test_compose_user_prompt_with_description_and_commands():
    """Test compose_user_prompt with description and commands."""
    settings = make_action_settings(
        description="Test description",
        commands=["command1", "command2"],
    )
    agent = ActionAgent(settings)

    prompt = agent.compose_user_prompt()

    assert "<DESCRIPTION>" in prompt
    assert "Test description" in prompt
    assert "</DESCRIPTION>" in prompt
    assert "<COMMANDS>" in prompt
    assert "- command1" in prompt
    assert "- command2" in prompt
    assert f"- {ANTI_INJECTION_WARNING}" in prompt
    assert "</COMMANDS>" in prompt


def test_compose_user_prompt_no_description():
    """Test compose_user_prompt without description."""
    settings = make_action_settings(description=None, commands=["command1"])
    agent = ActionAgent(settings)

    prompt = agent.compose_user_prompt()

    assert "<DESCRIPTION>" not in prompt
    assert "<COMMANDS>" in prompt
    assert "- command1" in prompt


def test_compose_user_prompt_empty_description():
    """Test compose_user_prompt with empty description."""
    settings = make_action_settings(description="", commands=["command1"])
    agent = ActionAgent(settings)

    prompt = agent.compose_user_prompt()

    assert "<DESCRIPTION>" not in prompt
    assert "<COMMANDS>" in prompt


def test_compose_user_prompt_no_commands():
    """Test compose_user_prompt with no commands (only anti-injection warning)."""
    settings = make_action_settings(commands=None)
    agent = ActionAgent(settings)

    prompt = agent.compose_user_prompt()

    assert "<COMMANDS>" in prompt
    assert f"- {ANTI_INJECTION_WARNING}" in prompt


def test_compose_user_prompt_description_only_no_commands():
    """Test compose_user_prompt with description but empty commands list."""
    settings = make_action_settings(
        description="Test description only",
        commands=[],  # Empty list, will only have anti-injection warning
    )
    agent = ActionAgent(settings)
    # Manually clear commands to test the empty case
    agent.commands = []

    prompt = agent.compose_user_prompt()

    assert "<DESCRIPTION>" in prompt
    assert "Test description only" in prompt
    assert "<COMMANDS>" not in prompt  # Should not appear when commands is empty


def test_compose_user_prompt_truly_empty_commands():
    """Test compose_user_prompt when commands list is truly empty."""
    settings = make_action_settings(description="Test desc", commands=[])
    agent = ActionAgent(settings)
    # Override to make commands truly empty (normally anti-injection warning is added)
    agent.commands = []

    prompt = agent.compose_user_prompt()

    assert "<DESCRIPTION>" in prompt
    assert "Test desc" in prompt
    # The missing branch: when self.commands is falsy,
    # no COMMANDS section should be added
    assert "<COMMANDS>" not in prompt


def test_compose_system_prompt_thinking_mode_true():
    """Test compose_system_prompt with thinking mode enabled."""
    settings = make_action_settings(thinking_mode=True)
    agent = ActionAgent(settings)

    prompt = agent.compose_system_prompt()

    # compose_system_prompt only returns the base prompt,
    # reasoning flags are added by get_final_system_prompt
    assert prompt == ACTION_SYSTEM_PROMPT


def test_compose_system_prompt_thinking_mode_false():
    """Test compose_system_prompt with thinking mode disabled."""
    settings = make_action_settings(thinking_mode=False)
    agent = ActionAgent(settings)

    prompt = agent.compose_system_prompt()

    # compose_system_prompt only returns the base prompt,
    # reasoning flags are added by get_final_system_prompt
    assert prompt == ACTION_SYSTEM_PROMPT


def test_run_integration():
    """Test the run method with mocked OpenAI client."""
    settings = make_action_settings()
    agent = ActionAgent(settings)

    # Mock the OpenAI client response
    mock_completion = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "<think>thinking</think>response"
    mock_completion.choices = [mock_choice]
    agent.openai_client.chat.completions.create = MagicMock(
        return_value=mock_completion,
    )

    response = agent.run(body="test body")

    assert isinstance(response, AgentResponse)
    assert response.thought == "thinking"
    assert response.response == "response"

    # Verify the OpenAI client was called with correct parameters
    agent.openai_client.chat.completions.create.assert_called_once()
    call_args = agent.openai_client.chat.completions.create.call_args

    assert call_args[1]["model"] == settings.model
    assert call_args[1]["temperature"] == settings.temperature
    assert call_args[1]["max_tokens"] == settings.max_tokens
    assert call_args[1]["stream"] is False

    # Check that the messages contain the expected content
    messages = call_args[1]["messages"]
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "<BODY>" in messages[1]["content"]
    assert "test body" in messages[1]["content"]
    assert "</BODY>" in messages[1]["content"]


def test_run_integration_no_body():
    """Test the run method without body content."""
    settings = make_action_settings()
    agent = ActionAgent(settings)

    mock_completion = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "response without thought"
    mock_completion.choices = [mock_choice]
    agent.openai_client.chat.completions.create = MagicMock(
        return_value=mock_completion,
    )

    response = agent.run(body=None)

    assert isinstance(response, AgentResponse)
    assert response.thought is None
    assert response.response == "response without thought"


def test_constants():
    """Test that constants are properly defined."""
    assert isinstance(ACTION_SYSTEM_PROMPT, str)
    assert len(ACTION_SYSTEM_PROMPT) > 0
    assert "Action Agent" in ACTION_SYSTEM_PROMPT

    assert isinstance(ANTI_INJECTION_WARNING, str)
    assert len(ANTI_INJECTION_WARNING) > 0
    assert "IMPORTANT" in ANTI_INJECTION_WARNING
