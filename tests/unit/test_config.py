"""Unit tests for configuration settings."""

import pytest
from pydantic import ValidationError

from tests.configutils import get_test_settings


@pytest.mark.parametrize("plugins", [[], ["plugins.auth"]])
def test_rejects_protected_file(plugins: list[str]) -> None:
    """Reject a protected-corpus file even when the authorization plugin is disabled."""
    with pytest.raises(ValidationError, match="PROTECTED_FILE is no longer supported") as exc_info:
        get_test_settings(
            PLUGINS=plugins,
            PLUGINS_CONFIG={"plugins.auth": {"PROTECTED_FILE": "/corpora/protected.txt"}},
        )

    message = str(exc_info.value)
    assert "Protected: true" in message
    assert "invalidate cached corpus metadata" in message


@pytest.mark.parametrize("auth_config", [{}, {"PROTECTED_FILE": ""}, {"PROTECTED_FILE": None}])
def test_allows_auth_without_protected_file(auth_config: dict) -> None:
    """Allow authorization configuration without an active protected-corpus file."""
    settings = get_test_settings(PLUGINS_CONFIG={"plugins.auth": auth_config})

    assert settings.PLUGINS_CONFIG["plugins.auth"] == auth_config


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
