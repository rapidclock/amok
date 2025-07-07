"""Tests for the config parsers module."""

import json
import tempfile
import tomllib

import pytest
import yaml

from amok.config.parsers import (
    BaseConfigParser,
    ConfigParserFactory,
    JsonConfigParser,
    TomlConfigParser,
    YamlConfigParser,
)


class TestJsonConfigParser:
    """Tests for JsonConfigParser."""

    def test_parse_valid_json(self):
        """Test parsing valid JSON string."""
        parser = JsonConfigParser()
        config_str = '{"key": "value", "number": 42}'
        result = parser.parse(config_str)
        assert result == {"key": "value", "number": 42}

    def test_parse_invalid_json(self):
        """Test parsing invalid JSON string."""
        parser = JsonConfigParser()
        config_str = '{"key": "value",}'  # Invalid JSON with trailing comma
        with pytest.raises(json.JSONDecodeError):
            parser.parse(config_str)

    def test_load_from_file(self):
        """Test loading JSON from file."""
        parser = JsonConfigParser()
        config_data = {"test": "data", "nested": {"key": "value"}}
        config_str = json.dumps(config_data)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(config_str)
            f.flush()

            result = parser.load(f.name)
            assert result == config_data


class TestYamlConfigParser:
    """Tests for YamlConfigParser."""

    def test_parse_valid_yaml(self):
        """Test parsing valid YAML string."""
        parser = YamlConfigParser()
        config_str = """
        key: value
        number: 42
        list:
          - item1
          - item2
        """
        result = parser.parse(config_str)
        assert result == {"key": "value", "number": 42, "list": ["item1", "item2"]}

    def test_parse_invalid_yaml(self):
        """Test parsing invalid YAML string."""
        parser = YamlConfigParser()
        config_str = """
        key: value
        invalid: [
          item1,
          item2,
        """  # Invalid YAML - unclosed bracket
        with pytest.raises(yaml.YAMLError):
            parser.parse(config_str)

    def test_load_from_file(self):
        """Test loading YAML from file."""
        parser = YamlConfigParser()
        config_data = {"test": "data", "nested": {"key": "value"}}
        config_str = yaml.dump(config_data)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_str)
            f.flush()

            result = parser.load(f.name)
            assert result == config_data


class TestTomlConfigParser:
    """Tests for TomlConfigParser."""

    def test_parse_valid_toml(self):
        """Test parsing valid TOML string."""
        parser = TomlConfigParser()
        config_str = """
        key = "value"
        number = 42

        [nested]
        key = "nested_value"
        """
        result = parser.parse(config_str)
        assert result == {
            "key": "value",
            "number": 42,
            "nested": {"key": "nested_value"},
        }

    def test_parse_invalid_toml(self):
        """Test parsing invalid TOML string."""
        parser = TomlConfigParser()
        config_str = """
        key = "value"
        invalid syntax here
        """
        with pytest.raises(
            tomllib.TOMLDecodeError,
        ):  # tomllib raises TOMLDecodeError for invalid TOML
            parser.parse(config_str)

    def test_load_from_file(self):
        """Test loading TOML from file."""
        parser = TomlConfigParser()
        config_str = """
        key = "value"
        number = 42

        [nested]
        key = "nested_value"
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(config_str)
            f.flush()

            result = parser.load(f.name)
            assert result == {
                "key": "value",
                "number": 42,
                "nested": {"key": "nested_value"},
            }


class TestConfigParserFactory:
    """Tests for ConfigParserFactory."""

    def test_get_parser_json(self):
        """Test getting JSON parser."""
        parser = ConfigParserFactory.get_parser("config.json")
        assert isinstance(parser, JsonConfigParser)

    def test_get_parser_yaml(self):
        """Test getting YAML parser."""
        parser = ConfigParserFactory.get_parser("config.yaml")
        assert isinstance(parser, YamlConfigParser)

    def test_get_parser_yml(self):
        """Test getting YAML parser with .yml extension."""
        parser = ConfigParserFactory.get_parser("config.yml")
        assert isinstance(parser, YamlConfigParser)

    def test_get_parser_toml(self):
        """Test getting TOML parser."""
        parser = ConfigParserFactory.get_parser("config.toml")
        assert isinstance(parser, TomlConfigParser)

    def test_get_parser_unsupported_extension(self):
        """Test getting parser for unsupported extension."""
        with pytest.raises(ValueError, match="Unsupported config file type"):
            ConfigParserFactory.get_parser("config.xml")

    def test_get_parser_for_extension_json(self):
        """Test getting parser for JSON extension."""
        parser = ConfigParserFactory.get_parser_for_extension("json")
        assert isinstance(parser, JsonConfigParser)

    def test_get_parser_for_extension_yaml(self):
        """Test getting parser for YAML extension."""
        parser = ConfigParserFactory.get_parser_for_extension("yaml")
        assert isinstance(parser, YamlConfigParser)

    def test_get_parser_for_extension_yml(self):
        """Test getting parser for YML extension."""
        parser = ConfigParserFactory.get_parser_for_extension("yml")
        assert isinstance(parser, YamlConfigParser)

    def test_get_parser_for_extension_toml(self):
        """Test getting parser for TOML extension."""
        parser = ConfigParserFactory.get_parser_for_extension("toml")
        assert isinstance(parser, TomlConfigParser)

    def test_get_parser_for_extension_case_insensitive(self):
        """Test getting parser for extensions with different cases."""
        parser = ConfigParserFactory.get_parser_for_extension("JSON")
        assert isinstance(parser, JsonConfigParser)

        parser = ConfigParserFactory.get_parser_for_extension("YAML")
        assert isinstance(parser, YamlConfigParser)

    def test_get_parser_for_extension_unsupported(self):
        """Test getting parser for unsupported extension."""
        with pytest.raises(ValueError, match="Unsupported config file type"):
            ConfigParserFactory.get_parser_for_extension("xml")

    def test_register_parser(self):
        """Test registering a new parser."""

        class CustomParser(BaseConfigParser):
            def parse(self, config_str: str) -> dict:
                return {"custom": "parser"}

        # Register the custom parser
        ConfigParserFactory.register_parser("custom", CustomParser)

        # Verify it can be retrieved
        parser = ConfigParserFactory.get_parser_for_extension("custom")
        assert isinstance(parser, CustomParser)

    def test_register_parser_invalid_class(self):
        """Test registering parser with invalid class."""

        class InvalidParser:
            pass

        with pytest.raises(
            TypeError,
            match="Parser class must inherit from BaseConfigParser",
        ):
            ConfigParserFactory.register_parser("invalid", InvalidParser)

    def test_register_parser_case_insensitive(self):
        """Test registering parser with case-insensitive extension."""

        class CaseParser(BaseConfigParser):
            def parse(self, config_str: str) -> dict:
                return {"case": "test"}

        ConfigParserFactory.register_parser("CASE", CaseParser)

        # Should be able to retrieve with lowercase
        parser = ConfigParserFactory.get_parser_for_extension("case")
        assert isinstance(parser, CaseParser)


class TestBaseConfigParser:
    """Tests for BaseConfigParser."""

    def test_load_with_custom_encoding(self):
        """Test loading file with custom encoding."""

        class TestParser(BaseConfigParser):
            def parse(self, config_str: str) -> dict:
                return {"content": config_str.strip()}

        parser = TestParser()
        content = "test content"

        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False) as f:
            f.write(content)
            f.flush()

            result = parser.load(f.name, encoding="utf-8")
            assert result == {"content": content}

    def test_load_default_encoding(self):
        """Test loading file with default encoding."""

        class TestParser(BaseConfigParser):
            def parse(self, config_str: str) -> dict:
                return {"content": config_str.strip()}

        parser = TestParser()
        content = "test content"

        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False) as f:
            f.write(content)
            f.flush()

            result = parser.load(f.name)  # No encoding specified
            assert result == {"content": content}
