"""Tests for the chain_agents utility."""

from unittest.mock import MagicMock

import pytest

from amok.lib import AgentResponse
from amok.utils import chain_agents


class MockAgent:
    def __init__(self, name):
        self.name = name
        self.run = MagicMock()

    def __call__(self, body):
        return self.run(body)


def test_chain_agents_multiple():
    """Test chaining multiple agents successfully."""
    agent1 = MockAgent("agent1")
    agent2 = MockAgent("agent2")
    agent3 = MockAgent("agent3")

    agent1.run.return_value = AgentResponse(thought=None, response="response1")
    agent2.run.return_value = AgentResponse(thought=None, response="response2")
    agent3.run.return_value = AgentResponse(thought=None, response="response3")

    final_response, all_responses = chain_agents("initial", agent1, agent2, agent3)

    assert final_response == "response3"
    assert len(all_responses) == 3
    assert all_responses[0].response == "response1"
    assert all_responses[1].response == "response2"
    assert all_responses[2].response == "response3"

    agent1.run.assert_called_once_with("initial")
    agent2.run.assert_called_once_with("response1")
    agent3.run.assert_called_once_with("response2")


def test_chain_agents_single():
    """Test chaining a single agent."""
    agent = MockAgent("single_agent")
    agent.run.return_value = AgentResponse(thought=None, response="final_response")

    final_response, all_responses = chain_agents("start", agent)

    assert final_response == "final_response"
    assert len(all_responses) == 1
    assert all_responses[0].response == "final_response"
    agent.run.assert_called_once_with("start")


def test_chain_agents_no_agents():
    """Test that chaining with no agents raises a ValueError."""
    with pytest.raises(ValueError, match="At least one agent must be provided"):
        chain_agents("initial_body")
