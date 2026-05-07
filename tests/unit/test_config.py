"""Unit tests for configuration settings."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from korp.config import Settings
from tests.configutils import get_test_settings


def test_plugins_config_loaded_from_yaml_file(tmp_path: Path) -> None:
    """Load plugin config from PLUGINS_CONFIG_FILE when present."""
    plugins_file = tmp_path / "plugins.yaml"
    plugins_file.write_text(
        'plugins.example:\n  greeting: "Hello from YAML!"\n',
        encoding="utf-8",
    )

    settings = get_test_settings(PLUGINS_CONFIG_FILE=plugins_file)

    assert settings.PLUGINS_CONFIG == {
        "plugins.example": {
            "greeting": "Hello from YAML!",
        },
    }


def test_plugins_config_explicit_value_overrides_yaml_file(tmp_path: Path) -> None:
    """Explicit PLUGINS_CONFIG should override entries loaded from YAML."""
    plugins_file = tmp_path / "plugins.yaml"
    plugins_file.write_text(
        'plugins.example:\n  greeting: "Hello from YAML!"\nother.plugin:\n  enabled: true\n',
        encoding="utf-8",
    )

    settings = get_test_settings(
        PLUGINS_CONFIG_FILE=plugins_file,
        PLUGINS_CONFIG={"plugins.example": {"greeting": "Hello from override!"}},
    )

    assert settings.PLUGINS_CONFIG == {
        "plugins.example": {
            "greeting": "Hello from override!",
        },
        "other.plugin": {
            "enabled": True,
        },
    }


def test_plugins_config_file_requires_mapping(tmp_path: Path) -> None:
    """Raise validation error if YAML root is not a mapping."""
    plugins_file = tmp_path / "plugins.yaml"
    plugins_file.write_text('- "not-a-mapping"\n', encoding="utf-8")

    with pytest.raises(ValidationError):
        get_test_settings(PLUGINS_CONFIG_FILE=plugins_file)
