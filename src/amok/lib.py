"""Library of dataclasses for amok."""

from dataclasses import dataclass


@dataclass
class AgentSettings:
    """Settings for an agent."""

    base_url: str
    model: str
    api_key: str = "sk-xxxxxxx"
    temperature: float = 0.7
    max_tokens: int = 1000
    ssl_verify: bool = True


@dataclass
class AgentResponse:
    """Response from an agent."""

    thought: str | None
    response: str
