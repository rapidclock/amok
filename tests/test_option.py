"""Tests for the option module."""

from unittest.mock import MagicMock

from amok.lib import AgentResponse, OptionAgentResponse, OptionAgentSettings
from amok.option import OPTION_SYSTEM_PROMPT, OptionAgent


def make_option_settings(**kwargs):
    """Create OptionAgentSettings with default values."""
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
        "options": ["option1", "option2", "option3"],
    }
    defaults.update(kwargs)
    return OptionAgentSettings(**defaults)


def test_option_agent_init():
    """Test OptionAgent initialization."""
    settings = make_option_settings()
    agent = OptionAgent(settings)

    assert agent.is_thinking_agent is True
    assert agent.description == "Test description"
    assert "command1" in agent.commands
    assert "command2" in agent.commands
    assert agent.options == ["option1", "option2", "option3", "None"]
    assert agent.openai_client is not None
    assert agent.model == settings.model
    assert agent.temperature == settings.temperature


def test_option_agent_get_settings_class():
    """Test OptionAgent _get_settings_class method."""
    settings_class = OptionAgent._get_settings_class()
    assert settings_class == OptionAgentSettings


def test_option_agent_init_no_options():
    """Test OptionAgent initialization with no options."""
    settings = make_option_settings(options=None)
    agent = OptionAgent(settings)

    assert agent.options == ["None"]


def test_option_agent_init_empty_options():
    """Test OptionAgent initialization with empty options list."""
    settings = make_option_settings(options=[])
    agent = OptionAgent(settings)

    assert agent.options == ["None"]


def test_option_agent_init_thinking_mode_false():
    """Test OptionAgent initialization with thinking mode disabled."""
    settings = make_option_settings(thinking_mode=False)
    agent = OptionAgent(settings)

    assert agent.is_thinking_agent is False


def test_compose_user_prompt_with_options():
    """Test compose_user_prompt with options."""
    settings = make_option_settings(
        description="Test description",
        commands=["command1"],
        options=["first option", "second option", "third option"],
    )
    agent = OptionAgent(settings)

    prompt = agent.compose_user_prompt()

    # Check that parent user prompt is included
    assert "<DESCRIPTION>" in prompt
    assert "Test description" in prompt
    assert "</DESCRIPTION>" in prompt
    assert "<COMMANDS>" in prompt
    assert "- command1" in prompt
    assert "</COMMANDS>" in prompt

    # Check that options section is added
    assert "<OPTIONS>" in prompt
    assert "0 : first option" in prompt
    assert "1 : second option" in prompt
    assert "2 : third option" in prompt
    assert "</OPTIONS>" in prompt


def test_compose_user_prompt_no_options():
    """Test compose_user_prompt with no options."""
    settings = make_option_settings(options=[])
    agent = OptionAgent(settings)

    prompt = agent.compose_user_prompt()

    # Check that parent user prompt is included
    assert "<DESCRIPTION>" in prompt
    assert "<COMMANDS>" in prompt

    # Check that empty options section is still added
    assert "<OPTIONS>" in prompt
    assert "</OPTIONS>" in prompt


def test_compose_user_prompt_single_option():
    """Test compose_user_prompt with single option."""
    settings = make_option_settings(options=["only option"])
    agent = OptionAgent(settings)

    prompt = agent.compose_user_prompt()

    assert "<OPTIONS>" in prompt
    assert "0 : only option" in prompt
    assert "</OPTIONS>" in prompt


def test_compose_user_prompt_empty_options_after_init():
    """Test compose_user_prompt with options cleared after initialization."""
    settings = make_option_settings(options=["option1"])
    agent = OptionAgent(settings)
    agent.options = []  # Manually clear options

    prompt = agent.compose_user_prompt()

    assert "<OPTIONS>\n</OPTIONS>" in prompt


def test_compose_system_prompt():
    """Test compose_system_prompt."""
    settings = make_option_settings()
    agent = OptionAgent(settings)

    prompt = agent.compose_system_prompt()

    assert prompt == OPTION_SYSTEM_PROMPT


def test_run_integration():
    """Test the run method with mocked OpenAI client."""
    settings = make_option_settings()
    agent = OptionAgent(settings)

    # Mock the OpenAI client response
    mock_completion = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "<think>thinking</think>1"
    mock_completion.choices = [mock_choice]
    agent.openai_client.chat.completions.create = MagicMock(
        return_value=mock_completion,
    )

    response = agent.run(body="test body")

    assert isinstance(response, AgentResponse)
    assert response.thought == "thinking"
    assert response.response == "option2"  # Index 1 should return "option2"
    assert response.option_index == 1  # Check the option_index is set correctly

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
    assert "<OPTIONS>" in messages[1]["content"]


def test_run_integration_no_body():
    """Test the run method without body content."""
    settings = make_option_settings()
    agent = OptionAgent(settings)

    mock_completion = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "0"
    mock_completion.choices = [mock_choice]
    agent.openai_client.chat.completions.create = MagicMock(
        return_value=mock_completion,
    )

    response = agent.run(body=None)

    assert isinstance(response, AgentResponse)
    assert response.thought is None
    assert response.response == "option1"  # Index 0 should return "option1"
    assert response.option_index == 0  # Check the option_index is set correctly


def test_constants():
    """Test that constants are properly defined."""
    assert isinstance(OPTION_SYSTEM_PROMPT, str)
    assert len(OPTION_SYSTEM_PROMPT) > 0
    assert "Option Agent" in OPTION_SYSTEM_PROMPT


def test_inheritance():
    """Test that OptionAgent properly inherits from ActionAgent."""
    settings = make_option_settings()
    agent = OptionAgent(settings)

    # Should have all ActionAgent attributes
    assert hasattr(agent, "description")
    assert hasattr(agent, "commands")
    assert hasattr(agent, "options")

    # Should inherit methods from ActionAgent and BaseAgent
    assert hasattr(agent, "run")
    assert hasattr(agent, "compose_user_prompt")
    assert hasattr(agent, "compose_system_prompt")


def test_process_output_option_invalid_index():
    """Test process_output_option with invalid option index."""
    settings = make_option_settings()
    agent = OptionAgent(settings)

    # Test with out-of-bounds index
    result = OptionAgentResponse(thought=None, response="99", option_index=None)
    processed = agent.process_output_option(result)

    assert processed.response == ""
    assert processed.option_index == -1


def test_process_output_option_invalid_response():
    """Test process_output_option with non-numeric response."""
    settings = make_option_settings()
    agent = OptionAgent(settings)

    # Test with non-numeric response
    result = OptionAgentResponse(
        thought=None,
        response="not_a_number",
        option_index=None,
    )
    processed = agent.process_output_option(result)

    assert processed.response == ""
    assert processed.option_index == -1


def test_process_output_option_none_response():
    """Test process_output_option with None response."""
    settings = make_option_settings()
    agent = OptionAgent(settings)

    # Test with None response (triggers AttributeError)
    result = OptionAgentResponse(thought=None, response=None, option_index=None)
    processed = agent.process_output_option(result)

    assert processed.response == ""
    assert processed.option_index == -1
