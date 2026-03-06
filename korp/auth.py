"""Authorization for the Korp API."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import TYPE_CHECKING

from korp.dependencies import AuthContext, Ctx

if TYPE_CHECKING:
    from korp.cwb import CWB
    from korp.memcached import Memcached


class KorpAuthorizationError(Exception):
    """Custom exception for Korp authorization errors."""


def _make_auth_context(ctx: Ctx) -> AuthContext:
    """Create an AuthContext from a request context.

    Returns:
        An AuthContext with request and cache settings from the given context.
    """
    return AuthContext(request=ctx.request, cache_enabled=ctx.common.cache)


async def get_protected_corpora(ctx: Ctx) -> list[str]:
    """Return a list of corpora with restricted access."""
    authorizer = ctx.request.app.state.authorizer
    if authorizer:
        return await authorizer.get_protected_corpora(_make_auth_context(ctx))
    return []


async def check_authorization(corpora: Iterable[str], ctx: Ctx) -> None:
    """Take a list of corpora, and if any of them are protected, check authorization.

    Args:
        corpora: List of corpus names.
        ctx: Request context used to build authorizer context.

    Raises:
        KorpAuthorizationError: If the user is not authorized to access one or more of the specified corpora.
    """
    authorizer = ctx.request.app.state.authorizer
    if authorizer:
        # Split parallel corpora
        corpora = [cc for c in corpora for cc in c.split("|")]

        success, unauthorized, message = await authorizer.check_authorization(corpora, _make_auth_context(ctx))
        if not success:
            if not message:
                message = "You do not have access to the following corpora: {}".format(", ".join(unauthorized))
            raise KorpAuthorizationError(message)


class Authorizer(ABC):
    """Class to subclass when implementing an authorizer plugin.

    The authorizer is responsible for determining which corpora have restricted access, and for checking whether a user
    is authorized to access a given list of corpora. The authorizer can use any information available in the request
    context, such as headers or cookies, to make these determinations. The authorizer can also use the CWB and cache to
    look up information about corpora or users if needed.

    When creating an authorizer plugin, you must define a module-level variable `AUTHORIZER_CLASS` that references your
    Authorizer subclass.
    """

    def __init__(self, cwb: CWB, cache: Memcached) -> None:
        """Initialize authorizer with app-scoped dependencies."""
        self.cwb = cwb
        self.cache = cache

    @abstractmethod
    async def get_protected_corpora(self, auth_ctx: AuthContext) -> list[str]:
        """Get list of corpora with restricted access, in uppercase."""

    @abstractmethod
    async def check_authorization(
        self, corpora: list[str], auth_ctx: AuthContext
    ) -> tuple[bool, list[str], str | None]:
        """Take a list of corpora and check that the user has permission to access them."""
