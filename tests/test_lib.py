from amok.lib import ActionAgentSettings, AgentResponse, AgentSettings


def test_agent_settings_required_fields():
    s = AgentSettings(base_url="http://x", model="m")
    assert s.base_url == "http://x"
    assert s.model == "m"


def test_agent_settings_defaults():
    s = AgentSettings(base_url="u", model="m")
    assert s.api_key == "sk-xxxxxxx"
    assert s.temperature == 0.7
    assert s.max_tokens == 1000
    assert s.ssl_verify is True


def test_agent_settings_custom_values():
    s = AgentSettings(
        base_url="b",
        model="m",
        api_key="k",
        temperature=0.1,
        max_tokens=42,
        ssl_verify=False,
    )
    assert s.api_key == "k"
    assert s.temperature == 0.1
    assert s.max_tokens == 42
    assert s.ssl_verify is False


def test_agent_settings_repr_and_eq():
    s1 = AgentSettings(base_url="a", model="b")
    s2 = AgentSettings(base_url="a", model="b")
    s3 = AgentSettings(base_url="a", model="c")
    assert s1 == s2
    assert s1 != s3
    assert "AgentSettings" in repr(s1)


def test_agent_response_fields():
    r = AgentResponse(thought="t", response="r")
    assert r.thought == "t"
    assert r.response == "r"
    r2 = AgentResponse(thought=None, response="foo")
    assert r2.thought is None
    assert r2.response == "foo"


def test_agent_response_repr_and_eq():
    r1 = AgentResponse(thought="t", response="r")
    r2 = AgentResponse(thought="t", response="r")
    r3 = AgentResponse(thought=None, response="r")
    assert r1 == r2
    assert r1 != r3
    assert "AgentResponse" in repr(r1)


def test_action_agent_settings_defaults():
    """Test ActionAgentSettings with default values."""
    s = ActionAgentSettings(base_url="http://localhost", model="test-model")
    assert s.thinking_mode is True
    assert s.description == ""
    assert s.commands == []


def test_action_agent_settings_custom_values():
    """Test ActionAgentSettings with custom values."""
    commands = ["command1", "command2"]
    s = ActionAgentSettings(
        base_url="http://test",
        model="test-model",
        thinking_mode=False,
        description="Test description",
        commands=commands,
        temperature=0.3,
        max_tokens=500,
    )
    assert s.thinking_mode is False
    assert s.description == "Test description"
    assert s.commands == commands
    assert s.temperature == 0.3
    assert s.max_tokens == 500


def test_action_agent_settings_post_init_none_commands():
    """Test ActionAgentSettings __post_init__ when commands is None."""
    s = ActionAgentSettings(base_url="http://test", model="test-model", commands=None)
    assert s.commands == []


def test_action_agent_settings_post_init_existing_commands():
    """Test ActionAgentSettings __post_init__ when commands already exist."""
    commands = ["existing_command"]
    s = ActionAgentSettings(
        base_url="http://test", model="test-model", commands=commands
    )
    assert s.commands == commands


def test_action_agent_settings_inheritance():
    """Test that ActionAgentSettings inherits from AgentSettings."""
    s = ActionAgentSettings(
        base_url="http://test", model="test-model", api_key="custom-key"
    )
    assert isinstance(s, AgentSettings)
    assert s.api_key == "custom-key"
    assert s.base_url == "http://test"
    assert s.model == "test-model"


def test_action_agent_settings_repr_and_eq():
    """Test ActionAgentSettings repr and equality."""
    s1 = ActionAgentSettings(
        base_url="http://test", model="test-model", description="desc"
    )
    s2 = ActionAgentSettings(
        base_url="http://test", model="test-model", description="desc"
    )
    s3 = ActionAgentSettings(
        base_url="http://test", model="test-model", description="different"
    )
    assert s1 == s2
    assert s1 != s3
    assert "ActionAgentSettings" in repr(s1)
