"""Library of dataclasses for amok."""

from dataclasses import dataclass, field


@dataclass
class AgentSettings:
    """Settings for an agent."""

    base_url: str
    model: str
    api_key: str = "sk-xxxxxxx"
    temperature: float = 0.7
    max_tokens: int = 1000
    ssl_verify: bool = True
    thinking_mode: bool = True


@dataclass
class ActionAgentSettings(AgentSettings):
    """Settings for an Action Agent."""

    description: str = ""
    commands: list[str] = field(default_factory=list)


@dataclass
class OptionAgentSettings(ActionAgentSettings):
    """Settings for an Option Agent."""

    options: list[str] = field(default_factory=list)


@dataclass
class AgentResponse:
    """Response from an agent."""

    thought: str | None
    response: str


@dataclass
class OptionAgentResponse(AgentResponse):
    """Response from an Option Agent."""

    option_index: int | None = None

    def __post_init__(self) -> None:
        """Ensure option_index is initialized."""
        if self.option_index is None or self.response is None:
            self.option_index = -1
