"""Authorization using JWT.

This plugin checks if the user has access to protected corpora based on JWT token scopes. It retrieves the list of
protected corpora from CWB info and checks the user's JWT token for the allowed corpora in the scopes. If the user
tries to access a protected corpus that is not in their allowed scopes, access is denied.

The plugin expects the JWT to be provided in the Authorization header as a Bearer token, and it uses a public key
to validate the token. The public key file can be configured using the "pubkey_file" setting in the plugin
configuration.

The optional "license_claim" setting names a top-level JWT claim containing a list of license access grants
(strings). When configured, a protected corpus is also accessible if its CWB "License" value appears in that
claim. License values are case-sensitive; metadata key names are case-insensitive. Explicit corpus scopes still
grant access regardless of license metadata. Without this setting, only corpus scopes grant access.

To determine which corpora are protected, it checks the CWB info for each corpus. A corpus is considered protected if
it has the "Protected" key set to "true" in its CWB `.info` file.

To use this plugin, you need to install the `pyjwt[crypto]` package.
"""

from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

import jwt  # type: ignore

from korp import auth, plugin, utils
from korp.dependencies import AuthContext
from plugins import protection_cwb

if TYPE_CHECKING:
    from korp.cwb import CWB
    from korp.memcached import Memcached

bp = plugin.Plugin("auth_jwt", __name__)


class AuthJWT(auth.Authorizer):
    """Authorizer plugin using JWT token scopes."""

    def __init__(self, cwb: CWB, cache: Memcached) -> None:
        """Initialize JWT settings.

        Raises:
            ValueError: If license_claim is not a non-empty string or None.
        """
        super().__init__(cwb, cache)
        license_claim = bp.config("license_claim")
        if license_claim is not None and (not isinstance(license_claim, str) or not license_claim.strip()):
            raise ValueError("license_claim must be a non-empty top-level JWT claim name or None.")
        self.license_claim: str | None = license_claim

    @classmethod
    def openapi_security(cls) -> tuple[dict[str, dict[str, str]], list[dict[str, list[str]]]]:
        """Document the bearer JWT consumed by this plugin.

        Returns:
            OpenAPI security schemes and operation requirements.
        """
        return {"bearerAuth": {"type": "http", "scheme": "bearer", "bearerFormat": "JWT"}}, [{"bearerAuth": []}]

    @classmethod
    def cache_vary_headers(cls) -> tuple[str, ...]:
        """Vary private cached responses by bearer credentials.

        Returns:
            The authorization header used by this plugin.
        """
        return ("Authorization",)

    async def _fetch_protection_info(self, corpora: list[str], auth_ctx: AuthContext) -> dict[str, auth.ProtectionInfo]:
        """Fetch per-corpus protection metadata from CWB.

        Returns:
            Protection metadata keyed by corpus.
        """
        return await protection_cwb.fetch_protection_info(
            self.cwb, corpora, self.cache, auth_ctx, detail_keys=("License",)
        )

    async def get_protected_corpora(self, auth_ctx: AuthContext) -> list[str]:
        """Get list of corpora with restricted access.

        Returns:
            Lowercase corpus ids marked as protected.
        """
        corpora = protection_cwb.list_corpora(self.cwb)
        protection_info = await self._get_protection_info(corpora, auth_ctx)
        return [corpus for corpus in corpora if protection_info[corpus].protected]

    async def check_authorization(
        self, corpora: list[str], auth_ctx: AuthContext
    ) -> tuple[bool, list[str], str | None]:
        """Check access through JWT corpus scopes or configured license grants.

        Args:
            corpora: A list of corpora to check access for.
            auth_ctx: The authentication context containing the request and other info.

        Returns:
            A tuple containing:
                - A boolean indicating if access is granted.
                - A list of unauthorized corpora (if access is denied).
                - An optional message (e.g., for errors).
        """
        protection_info = await self._get_protection_info(corpora, auth_ctx)
        protected_requested = [corpus for corpus in corpora if protection_info[corpus].protected]
        if protected_requested:
            try:
                user_corpora, user_licenses = self._get_user_grants(auth_ctx)
            except auth.KorpAuthorizationError as error:
                return False, [], str(error)

            unauthorized = []
            for corpus in protected_requested:
                if corpus in user_corpora:
                    continue
                license_value = next(
                    (value for key, value in protection_info[corpus].details.items() if key.casefold() == "license"),
                    None,
                )
                if isinstance(license_value, str) and license_value and license_value in user_licenses:
                    continue
                unauthorized.append(corpus)
            if unauthorized:
                return False, unauthorized, None
        return True, [], None

    async def get_user_corpora(self, auth_ctx: AuthContext) -> list[str]:
        """Return explicit JWT corpus grants, excluding license-based access.

        Returns:
            Sorted lowercase corpus ids, or an empty list for anonymous requests or tokens without corpus grants.
        """
        corpora, _licenses = self._get_user_grants(auth_ctx)
        return sorted(corpora)

    def _get_user_grants(self, auth_ctx: AuthContext) -> tuple[set[str], set[str]]:
        """Validate the JWT and extract corpus and license grants.

        Returns:
            Explicit corpus grants and configured license grants.

        Raises:
            auth.KorpAuthorizationError: If supplied credentials or corpus claims are invalid, or no key is configured.
        """
        auth_header = auth_ctx.request.headers.get("Authorization")
        if not auth_header:
            return set(), set()
        scheme, separator, token = auth_header.partition(" ")
        if scheme.casefold() != "bearer" or not separator or not token.strip():
            raise auth.KorpAuthorizationError("Could not validate the provided JWT.")
        if not self.jwt_key:
            raise auth.KorpAuthorizationError("JWT public key is not configured.")
        try:
            user_token = jwt.decode(token.strip(), key=self.jwt_key, algorithms=["RS256"])
        except jwt.ExpiredSignatureError:
            raise auth.KorpAuthorizationError("The provided JWT has expired") from None
        except jwt.InvalidTokenError:
            raise auth.KorpAuthorizationError("Could not validate the provided JWT.") from None

        scope = user_token.get("scope", {})
        if not isinstance(scope, dict):
            raise auth.KorpAuthorizationError("Invalid corpus grants in the provided JWT.")
        corpora = scope.get("corpora", {})
        if not isinstance(corpora, (dict, list)) or not all(isinstance(corpus, str) for corpus in corpora):
            raise auth.KorpAuthorizationError("Invalid corpus grants in the provided JWT.")
        user_corpora = {utils.normalize_corpus_id(corpus) for corpus in corpora}
        user_licenses: set[str] = set()
        if self.license_claim is not None:
            grants = user_token.get(self.license_claim)
            if isinstance(grants, list) and all(isinstance(grant, str) for grant in grants):
                user_licenses.update(grants)
        return user_corpora, user_licenses

    @cached_property
    def jwt_key(self) -> str | None:
        """Public key for validating JWTs."""
        pubkey_file = bp.config("pubkey_file")
        if not pubkey_file:
            return None
        return Path(pubkey_file).read_text(encoding="utf-8")


AUTHORIZER_CLASS = AuthJWT
