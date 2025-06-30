"""Action agent for amok."""

from amok.base import BaseAgent
from amok.lib import AgentSettings


class ActionAgent(BaseAgent):
    """An agent that performs actions based on user input."""

    description: str | None = None
    commands: list[str] = list([])

    def __init__(self, settings: AgentSettings) -> None:
        """Initialize the action agent."""
        super().__init__(settings)
        self.read_cfg()

    def read_cfg(self) -> None:
        """Read the agent's configuration."""
        # Implementation for reading configuration settings goes here.

    def compose_user_prompt(self) -> str:
        """Compose the user prompt."""
        # Implementation for composing the user prompt goes here.
        return "User prompt for ActionAgent."

    def compose_system_prompt(self) -> str:
        """Compose the system prompt."""
        # Implementation for composing the system prompt goes here.
        return "System prompt for ActionAgent."
