"""Unit tests for the example auth plugin."""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import anyio

from korp.config import settings
from plugins.example_auth import ExampleAuth


class FakeCWB:
    """Minimal CWB stub for the example plugin tests."""

    @staticmethod
    def run_cqp(command: str | list[str], attr_ignore: bool = False, abort_signal: Any | None = None) -> Iterator[str]:
        """Return fixed corpus list for `show corpora`."""
        del attr_ignore, abort_signal
        assert command == "show corpora;"
        return iter(["3.5.0", "A", "B"])


def test_example_auth_get_protected_corpora_uses_plugin_config(monkeypatch: Any) -> None:
    """Resolve protected corpora from plugin config metadata."""
    monkeypatch.setattr(
        settings,
        "PLUGINS_CONFIG",
        {
            "plugins.example_auth": {
                "protected_corpora": ["A"],
                "protection_details": {"A": {"license": "restricted"}},
            }
        },
    )

    async def _run() -> None:
        authorizer = ExampleAuth(cwb=FakeCWB(), cache=object())  # type: ignore
        auth_ctx = SimpleNamespace(cache_enabled=False, request=SimpleNamespace(headers={}))

        protected = await authorizer.get_protected_corpora(auth_ctx)  # type: ignore

        assert protected == ["A"]

    anyio.run(_run)


def test_example_auth_check_authorization_denies_missing_header_access(monkeypatch: Any) -> None:
    """Deny access for protected corpora not listed in required header."""
    monkeypatch.setattr(
        settings,
        "PLUGINS_CONFIG",
        {
            "plugins.example_auth": {
                "protected_corpora": ["A"],
                "required_header": "X-Authorized-Corpora",
            }
        },
    )

    async def _run() -> None:
        authorizer = ExampleAuth(cwb=FakeCWB(), cache=object())  # type: ignore
        auth_ctx = SimpleNamespace(cache_enabled=False, request=SimpleNamespace(headers={}))

        ok, unauthorized, message = await authorizer.check_authorization(["A", "B"], auth_ctx)  # type: ignore

        assert not ok
        assert unauthorized == ["A"]
        assert message is not None
        assert "X-Authorized-Corpora" in message

    anyio.run(_run)


def test_example_auth_check_authorization_allows_header_access(monkeypatch: Any) -> None:
    """Allow access when protected corpora are listed in the required header."""
    monkeypatch.setattr(
        settings,
        "PLUGINS_CONFIG",
        {
            "plugins.example_auth": {
                "protected_corpora": ["A"],
                "required_header": "X-Authorized-Corpora",
            }
        },
    )

    async def _run() -> None:
        authorizer = ExampleAuth(cwb=FakeCWB(), cache=object())  # type: ignore
        auth_ctx = SimpleNamespace(cache_enabled=False, request=SimpleNamespace(headers={"X-Authorized-Corpora": "A"}))

        ok, unauthorized, message = await authorizer.check_authorization(["A", "B"], auth_ctx)  # type: ignore

        assert ok
        assert unauthorized == []
        assert message is None

    anyio.run(_run)
