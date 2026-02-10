"""Base classes for agents."""

import re
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any, Self

import httpx
from openai import OpenAI

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
    api_mode: str
    tools: list[dict[str, Any]]
    tool_choice: str | dict[str, Any] | None
    parallel_tool_calls: bool | None
    is_thinking_agent: bool = True
    body_tag: str = "BODY"
    supported_api_modes: tuple[str, ...] = ("chat.completions", "responses")

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
        if settings.api_mode not in self.supported_api_modes:
            msg = (
                f"Unsupported api_mode '{settings.api_mode}'. "
                f"Supported values are: {', '.join(self.supported_api_modes)}."
            )
            raise ValueError(msg)
        self.api_mode = settings.api_mode
        self.tools = self._normalize_tools(settings.tools)
        self.tool_choice = settings.tool_choice
        self.parallel_tool_calls = settings.parallel_tool_calls
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
        if self.api_mode == "responses":
            content, tool_calls = self._run_responses(system_prompt, user_prompt)
        else:
            content, tool_calls = self._run_chat_completions(system_prompt, user_prompt)
        if not content and not tool_calls:
            msg = "No response from the model."
            raise ValueError(msg)
        response, thought = self._parse_response(content)
        return AgentResponse(thought=thought, response=response, tool_calls=tool_calls)

    def _run_chat_completions(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Execute a request using the Chat Completions API."""
        if not self.openai_client:
            msg = "OpenAI client not initialized."
            raise ValueError(msg)
        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": self.stream,
        }
        if self.tools:
            request_kwargs["tools"] = self.tools
        if self.tool_choice is not None:
            request_kwargs["tool_choice"] = self.tool_choice
        if self.parallel_tool_calls is not None:
            request_kwargs["parallel_tool_calls"] = self.parallel_tool_calls

        completion = self.openai_client.chat.completions.create(**request_kwargs)
        if (
            not completion
            or not completion.choices
            or not completion.choices[0].message
        ):
            return "", []

        message = completion.choices[0].message
        content = self._extract_chat_message_text(message.content)
        tool_calls = self._extract_chat_tool_calls(message.tool_calls)
        return content, tool_calls

    def _run_responses(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Execute a request using the Responses API."""
        if not self.openai_client:
            msg = "OpenAI client not initialized."
            raise ValueError(msg)
        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_output_tokens": self.max_tokens,
            "stream": self.stream,
        }
        if self.tools:
            request_kwargs["tools"] = self.tools
        if self.tool_choice is not None:
            request_kwargs["tool_choice"] = self.tool_choice
        if self.parallel_tool_calls is not None:
            request_kwargs["parallel_tool_calls"] = self.parallel_tool_calls

        response = self.openai_client.responses.create(**request_kwargs)
        content = self._extract_response_output_text(response)
        tool_calls = self._extract_response_tool_calls(getattr(response, "output", []))
        return content, tool_calls

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

    @staticmethod
    def _extract_chat_message_text(content: str | list[Any] | None) -> str:
        """Extract text content from a chat completion message."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                text = getattr(item, "text", None)
                if isinstance(text, str):
                    parts.append(text)
                elif isinstance(item, dict):
                    dict_text = item.get("text")
                    if isinstance(dict_text, str):
                        parts.append(dict_text)
            return "\n".join(parts).strip()
        return ""

    @staticmethod
    def _extract_chat_tool_calls(tool_calls: object) -> list[dict[str, Any]]:
        """Extract normalized tool-call metadata from chat completions output."""
        if not isinstance(tool_calls, Iterable) or isinstance(tool_calls, str | bytes):
            return []
        extracted: list[dict[str, Any]] = []
        for tool_call in tool_calls:
            function = getattr(tool_call, "function", None)
            if function is None and isinstance(tool_call, dict):
                function = tool_call.get("function")
            call_payload: dict[str, Any] = {
                "id": getattr(tool_call, "id", None),
                "type": getattr(tool_call, "type", None),
            }
            if isinstance(tool_call, dict):
                if "id" in tool_call:
                    call_payload["id"] = tool_call.get("id")
                if "type" in tool_call:
                    call_payload["type"] = tool_call.get("type")
            if function is not None:
                if isinstance(function, dict):
                    call_payload["name"] = function.get("name")
                    call_payload["arguments"] = function.get("arguments")
                else:
                    call_payload["name"] = getattr(function, "name", None)
                    call_payload["arguments"] = getattr(function, "arguments", None)
            extracted.append(call_payload)
        return extracted

    @staticmethod
    def _extract_response_output_text(response: object) -> str:
        """Extract text content from a Responses API result."""
        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str) and output_text.strip():
            return output_text
        output_items = getattr(response, "output", [])
        if not output_items:
            return ""
        text_parts: list[str] = []
        for item in output_items:
            item_type = getattr(item, "type", None)
            if item_type != "message":
                continue
            content_items = getattr(item, "content", [])
            for content in content_items:
                text = getattr(content, "text", None)
                if isinstance(text, str):
                    text_parts.append(text)
                elif isinstance(content, dict):
                    dict_text = content.get("text")
                    if isinstance(dict_text, str):
                        text_parts.append(dict_text)
        return "\n".join(text_parts).strip()

    @staticmethod
    def _extract_response_tool_calls(output_items: object) -> list[dict[str, Any]]:
        """Extract normalized tool-call metadata from Responses API output."""
        if not isinstance(output_items, Iterable) or isinstance(
            output_items,
            str | bytes,
        ):
            return []
        extracted: list[dict[str, Any]] = []
        for item in output_items:
            item_type = getattr(item, "type", None)
            if item_type not in {"function_call", "function_tool_call"}:
                continue
            extracted.append(
                {
                    "id": getattr(item, "id", None),
                    "type": item_type,
                    "name": getattr(item, "name", None),
                    "arguments": getattr(item, "arguments", None),
                    "call_id": getattr(item, "call_id", None),
                    "status": getattr(item, "status", None),
                },
            )
        return extracted

    def _normalize_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Normalize function-tool payloads for the selected API mode.

        Supports both of these input shapes:
        - Chat shape: {"type":"function","function":{"name":"...","parameters":{...}}}
        - Responses shape: {"type":"function","name":"...","parameters":{...}}
        """
        if not tools:
            return []
        normalized: list[dict[str, Any]] = []
        for tool in tools:
            normalized.append(self._normalize_single_tool(tool))
        return normalized

    def _normalize_single_tool(self, tool: dict[str, Any]) -> dict[str, Any]:
        """Normalize one tool payload based on api mode."""
        if not isinstance(tool, dict) or tool.get("type") != "function":
            return tool
        if self.api_mode == "chat.completions":
            return self._normalize_function_tool_for_chat(tool)
        return self._normalize_function_tool_for_responses(tool)

    @staticmethod
    def _normalize_function_tool_for_chat(tool: dict[str, Any]) -> dict[str, Any]:
        """Convert flat responses-style function tools into chat shape."""
        function_payload = tool.get("function")
        if isinstance(function_payload, dict):
            return tool
        converted: dict[str, Any] = {"type": "function", "function": {}}
        for key in ("name", "description", "parameters", "strict"):
            if key in tool:
                converted["function"][key] = tool[key]
        return converted

    @staticmethod
    def _normalize_function_tool_for_responses(tool: dict[str, Any]) -> dict[str, Any]:
        """Convert nested chat-style function tools into responses shape."""
        function_payload = tool.get("function")
        if not isinstance(function_payload, dict):
            return tool
        converted: dict[str, Any] = {"type": "function"}
        for key in ("name", "description", "parameters", "strict"):
            if key in function_payload:
                converted[key] = function_payload[key]
        return converted
