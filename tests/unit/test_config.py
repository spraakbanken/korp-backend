"""Unit tests for configuration settings."""

from pathlib import Path

import pytest
from pydantic import ValidationError

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


def test_cors_allows_wildcard_without_credentials() -> None:
    """Allow wildcard origins when credentials are disabled."""
    settings = get_test_settings(CORS_ALLOW_ORIGINS=["*"], CORS_ALLOW_CREDENTIALS=False)

    assert settings.CORS_ALLOW_ORIGINS == ["*"]


def test_cors_rejects_wildcard_with_credentials() -> None:
    """Reject wildcard origins when credentials are enabled."""
    with pytest.raises(ValidationError, match=r"CORS_ALLOW_ORIGINS cannot contain '\*'"):
        get_test_settings(CORS_ALLOW_ORIGINS=["*"], CORS_ALLOW_CREDENTIALS=True)


@pytest.mark.parametrize("origin_regex", [".*", "^.*$", ".+", r"[\s\S]*"])
def test_cors_rejects_match_all_regex_with_credentials(origin_regex: str) -> None:
    """Reject obvious match-all origin regexes when credentials are enabled."""
    with pytest.raises(ValidationError, match="CORS_ALLOW_ORIGIN_REGEX cannot match all origins"):
        get_test_settings(CORS_ALLOW_ORIGIN_REGEX=origin_regex, CORS_ALLOW_CREDENTIALS=True)


def test_cors_allows_constrained_regex_with_credentials() -> None:
    """Allow a constrained origin regex when credentials are enabled."""
    settings = get_test_settings(
        CORS_ALLOW_ORIGIN_REGEX=r"https://([a-z0-9-]+\.)?example\.org",
        CORS_ALLOW_CREDENTIALS=True,
    )

    assert settings.CORS_ALLOW_ORIGIN_REGEX == r"https://([a-z0-9-]+\.)?example\.org"
