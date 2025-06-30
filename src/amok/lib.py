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
class ActionAgentSettings(AgentSettings):
    """Settings for an Action Agent."""

    thinking_mode: bool = True
    description: str = ""
    commands: list[str] = None

    def __post_init__(self):
        """Initialize commands list if None."""
        if self.commands is None:
            self.commands = []


@dataclass
class AgentResponse:
    """Response from an agent."""

    thought: str | None
    response: str
