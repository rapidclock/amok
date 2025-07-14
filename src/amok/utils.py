"""Utility functions for the Amok project."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import BaseAgent
    from .lib import AgentResponse


def surround_with_tags(text: str, tag: str) -> str:
    """Surround the given text with specified tags.

    Args:
        text: The text to be surrounded.
        tag: The tag to use for surrounding the text.

    Returns:
        The text surrounded by the specified tags.

    """
    if text is None or text.strip() == "":
        return ""
    if tag is None or tag.strip() == "":
        msg = "Tag cannot be None or empty."
        raise ValueError(msg)
    return f"<{tag.upper()}>\n{text.strip()}\n</{tag.upper()}>"


def chain_agents(
    initial_body: str,
    *agents: "BaseAgent",
) -> tuple[str, list["AgentResponse"]]:
    """Chain a list of agents in order, passing the response from one agent to the next.

    Args:
        initial_body: The initial body text to send to the first agent
        *agents: Variable number of agents to chain in order

    Returns:
        A tuple containing:
        - final_response: The response from the last agent
        - all_responses: List of all AgentResponse objects from each agent

    Raises:
        ValueError: If no agents are provided

    """
    if not agents:
        msg = "At least one agent must be provided"
        raise ValueError(msg)

    all_responses = []
    current_body = initial_body

    for agent in agents:
        response = agent.run(current_body)
        all_responses.append(response)
        current_body = response.response

    final_response = all_responses[-1].response
    return final_response, all_responses
