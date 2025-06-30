from amok.lib import AgentResponse, AgentSettings


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
