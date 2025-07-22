"""Base classes for agents."""

import re
from abc import ABC, abstractmethod
from typing import Any, Self

import httpx
from openai import OpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from .config import BaseConfigParser, ConfigParserFactory
from .lib import AgentResponse, AgentSettings
from .utils import surround_with_tags


class BaseAgent(ABC):
    """Abstract base class for all agents."""

    openai_client: OpenAI | None = None
    model: str
    temperature: float
    max_tokens: int
    ssl_verify: bool
    stream: bool
    is_thinking_agent: bool = True
    body_tag: str = "BODY"

    def __init__(self, settings: AgentSettings) -> None:
        """Initialize the agent with the given settings.

        Args:
            settings: The settings for the agent.

        """
        self.model = settings.model
        self.temperature = settings.temperature
        self.max_tokens = settings.max_tokens
        self.ssl_verify = settings.ssl_verify
        self.is_thinking_agent = settings.thinking_mode
        self.stream = False
        http_client = httpx.Client(verify=settings.ssl_verify)
        self.openai_client = OpenAI(
            base_url=settings.base_url,
            api_key=settings.api_key,
            http_client=http_client,
        )

    @classmethod
    def from_cfg(cls, cfg_file_path: str) -> Self:
        """Create an Agent based on a configuration file."""
        setting_class: type[AgentSettings] = cls._get_settings_class()
        cfg_parser: BaseConfigParser = ConfigParserFactory.get_parser(cfg_file_path)
        cfg: dict[str, Any] = cfg_parser.load(cfg_file_path)
        validated_cfg: dict[str, Any] = cls.validated_settings(cfg)
        settings = setting_class(**validated_cfg)
        return cls(settings)

    @classmethod
    def validated_settings(cls, settings: dict[str, Any]) -> dict[str, Any]:
        """Validate and return the settings for the agent."""
        settings_class = cls._get_settings_class()
        if not issubclass(settings_class, AgentSettings):
            msg = (
                f"Settings class {settings_class.__name__} must inherit "
                f"from AgentSettings"
            )
            raise TypeError(
                msg,
            )
        # Get expected fields from the settings class
        expected_fields = set()
        for cls_in_mro in settings_class.__mro__:
            if hasattr(cls_in_mro, "__annotations__"):
                expected_fields.update(cls_in_mro.__annotations__.keys())

        # Filter and validate config
        validated_config = {}
        for key, value in settings.items():
            if key in expected_fields:
                validated_config[key] = value

        return validated_config

    @classmethod
    @abstractmethod
    def _get_settings_class(cls) -> type[AgentSettings]:
        """Get the settings class for the agent."""
        pass

    @abstractmethod
    def compose_user_prompt(self) -> str:
        """Compose the user prompt."""
        pass

    @abstractmethod
    def compose_system_prompt(self) -> str:
        """Compose the system prompt."""
        pass

    def run(self, body: str | None) -> AgentResponse:
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
        system_prompt, user_prompt = self.generate_prompts(body)
        if not self.openai_client:
            msg = "OpenAI client not initialized."
            raise ValueError(msg)
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
            msg = "No response from the model."
            raise ValueError(msg)
        content: str = completion.choices[0].message.content
        response, thought = self._parse_response(content)
        return AgentResponse(thought=thought, response=response)

    def generate_prompts(self, body: str | None) -> tuple[str, str]:
        """Generate the system and user prompts for the given body.

        This method processes the prompts and returns them as a tuple.
        It is used to prepare the prompts before sending them to the OpenAI API.

        Args:
            body: The body content to include in the user prompt.

        Returns:
            A tuple containing the system prompt and the user prompt.

        """
        system_prompt = self.get_final_system_prompt()
        user_prompt = self.compose_user_prompt() + "\n" + self.process_body(body)
        return system_prompt, user_prompt

    def get_final_system_prompt(self) -> str:
        """Add a thinking option to the system prompt if the agent is a thinking agent.

        Returns:
            The modified system prompt with the thinking option added if applicable.

        """
        final_system_prompt = "\n".join(
            [
                f"detailed thinking {'on' if self.is_thinking_agent else 'off'}",
                f"<think>{'</think>' if not self.is_thinking_agent else ''}",
                self.compose_system_prompt(),
            ],
        )
        return final_system_prompt

    def process_body(self, body: str | None) -> str:
        """Process the body content to ensure it prepped to be added to the user prompt.

        Args:
            body: The body content to process.

        Returns:
            The processed body content ready to be included in the user prompt.

        """
        if body is None:
            return ""
        return surround_with_tags(body, self.body_tag)

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
        thought_pattern = r"<think>(.*?)</think>"
        match = re.search(thought_pattern, content, re.DOTALL)
        if match:
            # The thought is the content of the first capture group.
            thought = match.group(1).strip()
            # The response is the original content with the thought block removed.
            # count=1 ensures we only replace the first occurrence.
            response = re.sub(
                thought_pattern,
                "",
                content,
                count=1,
                flags=re.DOTALL,
            ).strip()
        else:
            # If no thought tag is found, the entire content is the response.
            thought = None
            response = content.strip()
        return response, thought
