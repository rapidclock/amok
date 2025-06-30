"""Base classes for agents."""

import re
from abc import ABC, abstractmethod

from openai import OpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from amok.lib import AgentResponse, AgentSettings


class BaseAgent(ABC):
    """Abstract base class for all agents."""

    openai_client: OpenAI | None = None
    model: str
    temperature: float
    max_tokens: int
    ssl_verify: bool
    stream: bool

    def __init__(self, settings: AgentSettings) -> None:
        """Initialize the agent with the given settings.

        Args:
            settings: The settings for the agent.

        """
        self.openai_client = OpenAI(
            base_url=settings.base_url, api_key=settings.api_key
        )
        self.model = settings.model
        self.temperature = settings.temperature
        self.max_tokens = settings.max_tokens
        self.ssl_verify = settings.ssl_verify
        self.stream = False

    @abstractmethod
    def read_cfg(self) -> None:
        """Read the agent's configuration."""
        pass

    @abstractmethod
    def compose_user_prompt(self) -> str:
        """Compose the user prompt."""
        pass

    @abstractmethod
    def compose_system_prompt(self) -> str:
        """Compose the system prompt."""
        pass

    def run(self) -> AgentResponse:
        """Execute the agent's main logic.

        This method composes the user and system prompts, sends them to the
        OpenAI API, and processes the response.

        Returns
        -------
            An `AgentResponse` object containing the model's response and any
            accompanying thought process.

        Raises
        ------
            ValueError: If the OpenAI client is not initialized or if the model
                        returns no response.

        """
        user_prompt = self.compose_user_prompt()
        system_prompt = self.compose_system_prompt()
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized.")
        completion: ChatCompletion = self.openai_client.chat.completions.create(
            model=self.model,
            messages=[
                ChatCompletionSystemMessageParam(role="system", content=system_prompt),
                ChatCompletionUserMessageParam(role="user", content=user_prompt),
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=self.stream,
        )
        if (
            not completion
            or not completion.choices
            or not completion.choices[0].message
            or not completion.choices[0].message.content
        ):
            raise ValueError("No response from the model.")
        content: str = completion.choices[0].message.content
        response, thought = self._parse_response(content)
        return AgentResponse(thought=thought, response=response)

    @staticmethod
    def _parse_response(content: str) -> tuple[str, str | None]:
        """Parse the model response to separate thought from the response.

        Args:
        ----
            content: The raw response content from the model.

        Returns:
        -------
            A tuple containing the response and the thought.

        """
        thought_pattern = r"<thought>(.*?)</thought>"
        match = re.search(thought_pattern, content, re.DOTALL)
        if match:
            # The thought is the content of the first capture group.
            thought = match.group(1).strip()
            # The response is the original content with the thought block removed.
            # count=1 ensures we only replace the first occurrence.
            response = re.sub(
                thought_pattern, "", content, count=1, flags=re.DOTALL
            ).strip()
        else:
            # If no thought tag is found, the entire content is the response.
            thought = None
            response = content.strip()
        return response, thought
