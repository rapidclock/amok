"""Configuration Parser Implementations for various formats."""

import json
import tomllib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

import yaml


class BaseConfigParser(ABC):
    """Abstract base class for all configuration parsers."""

    @abstractmethod
    def parse(self, config_str: str) -> dict[str, Any]:
        """Parse the config contents into a dictionary."""
        pass

    def load(self, path: str, encoding: str = "utf-8") -> dict[str, Any]:
        """Load configuration from a file."""
        with Path(path).open(encoding=encoding) as file:
            config_str = file.read()
        return self.parse(config_str)


class JsonConfigParser(BaseConfigParser):
    """Parser for JSON configuration files."""

    def parse(self, config_str: str) -> dict[str, Any]:
        """Parse the JSON config string into a dictionary."""
        return json.loads(config_str)


class YamlConfigParser(BaseConfigParser):
    """Parser for YAML configuration files."""

    def parse(self, config_str: str) -> dict[str, Any]:
        """Parse the YAML config string into a dictionary."""
        return yaml.safe_load(config_str)


class TomlConfigParser(BaseConfigParser):
    """Parser for TOML configuration files."""

    def parse(self, config_str: str) -> dict[str, Any]:
        """Parse the TOML config string into a dictionary."""
        return tomllib.loads(config_str)


class ConfigParserFactory:
    """Factory for creating configuration parsers."""

    _parsers: ClassVar[dict[str, type[BaseConfigParser]]] = {
        "json": JsonConfigParser,
        "yaml": YamlConfigParser,
        "yml": YamlConfigParser,
        "toml": TomlConfigParser,
    }

    @classmethod
    def get_parser(cls, file_path: str) -> BaseConfigParser:
        """Get a configuration parser based on the type."""
        extension = Path(file_path).suffix.lower()
        return cls.get_parser_for_extension(extension)

    @classmethod
    def get_parser_for_extension(cls, file_extension: str) -> BaseConfigParser:
        """Get a configuration parser for a specific file extension."""
        file_extension = file_extension.lower()
        # Remove leading dot if present
        if file_extension.startswith("."):
            file_extension = file_extension[1:]
        if file_extension not in cls._parsers:
            msg = f"Unsupported config file type: {file_extension}"
            raise ValueError(msg)
        return cls._parsers[file_extension]()

    @classmethod
    def register_parser(
        cls,
        file_extension: str,
        parser_class: type[BaseConfigParser],
    ) -> None:
        """Register a new parser for a file extension."""
        if not issubclass(parser_class, BaseConfigParser):
            msg = "Parser class must inherit from BaseConfigParser"
            raise TypeError(msg)
        cls._parsers[file_extension.lower()] = parser_class
