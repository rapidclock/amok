"""Option agent for amok."""

from amok import AgentSettings, BaseAgent


class OptionAgent(BaseAgent):
    """An agent that provides options."""

    description: str | None = None
    options: list[str] = list([])

    def __init__(self, settings: AgentSettings) -> None:
        """Initialize the option agent."""
        super().__init__(settings)
        self.read_cfg()

    def read_cfg(self) -> None:
        """Read the agent's configuration."""
        pass

    def compose_user_prompt(self) -> str:
        """Compose the user prompt."""
        pass

    def compose_system_prompt(self) -> str:
        """Compose the system prompt."""
        pass
