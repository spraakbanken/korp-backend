"""Example translation plugin with two dummy query languages.

Enable with PLUGINS=["plugins.example_query_translation"].

Both `dummy-word` and `dummy-lemma` accept one token of letters, digits, or underscores.

For a real translation library, keep cheap validation inline and offload potentially expensive synchronous parsing,
compilation, and formatting to a worker thread (anything that takes more than 50 ms). For example:

    return await anyio.to_thread.run_sync(self._translate_sync, language, query)

Here, `_translate_sync` is your synchronous helper returning the finished CQP string.
"""

import re
from collections.abc import Mapping
from typing import Any

from korp.dependencies import Ctx


class ExampleQueryTranslator:
    """Example query translator for two dummy query languages.

    Korp shares this instance across concurrent requests. Keep per-query parser and compilation state local to
    each call, and treat any configuration stored on the instance as read-only during translation.
    """

    languages = frozenset({"dummy-word", "dummy-lemma"})

    async def translate(self, ctx: Ctx, language: str, query: str) -> str:  # ruff: ignore[no-self-use, unused-method-argument]
        """Translate one simple token query in one of the dummy languages into a CQP expression.

        Returns:
            One CQP expression.

        Raises:
            ValueError: If the input is not a single supported token.
        """
        # This simple validation and string formatting are cheap enough to run on the event loop.

        # Use ValueError to signal errors with the input.
        if not re.fullmatch(r"\w+", query):
            raise ValueError("Dummy queries accept one token containing only letters, digits, and underscores.")

        # language has already been validated against self.languages by the caller, so we can safely use it here.
        attribute = "word" if language == "dummy-word" else "lemma"

        # Return one non-empty CQP expression without a trailing semicolon or outer within clause.
        return f'[{attribute}="{query}"]'


def create_translator(config: Mapping[str, Any]) -> ExampleQueryTranslator:  # ruff: ignore[unused-function-argument]
    """Create a query translator provider for this plugin.

    Korp calls this factory at app creation with this plugin's `PLUGINS_CONFIG` entry (or an empty mapping).
    Validate settings and initialize reusable library configuration here.

    Returns:
        The configured provider.
    """
    return ExampleQueryTranslator()


QUERY_TRANSLATOR_FACTORY = create_translator
