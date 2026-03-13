"""Authorization using JWT.

This plugin checks if the user has access to protected corpora based on JWT token scopes. It retrieves the list of
protected corpora from CWB info and checks the user's JWT token for the allowed corpora in the scopes. If the user
tries to access a protected corpus that is not in their allowed scopes, access is denied.

The plugin expects the JWT to be provided in the Authorization header as a Bearer token, and it uses a public key
to validate the token. The public key file can be configured using the "pubkey_file" setting in the plugin
configuration.

To determine which corpora are protected, it checks the CWB info for each corpus. A corpus is considered protected if
it has the "Protected" key set to "true" in its CWB `.info` file.

To use this plugin, you need to install the `pyjwt[crypto]` package.
"""

import time
from functools import cached_property
from pathlib import Path

import jwt  # type: ignore

from korp import auth, plugin
from korp.dependencies import AuthContext
from plugins import protection_cwb

bp = plugin.Plugin("auth_jwt", __name__)


class AuthJWT(auth.Authorizer):
    """Authorizer plugin using JWT token scopes."""

    async def _fetch_protection_info(
        self, corpora: list[str], auth_ctx: AuthContext
    ) -> dict[str, auth.ProtectionInfo]:
        """Fetch per-corpus protection metadata from CWB.

        Returns:
            Protection metadata keyed by corpus.
        """
        return await protection_cwb.fetch_protection_info(self.cwb, corpora, self.cache, auth_ctx)

    async def get_protected_corpora(self, auth_ctx: AuthContext) -> list[str]:
        """Get list of corpora with restricted access.

        Returns:
            Uppercased corpus ids marked as protected.
        """
        corpora = protection_cwb.list_corpora(self.cwb)
        protection_info = await self._get_protection_info(corpora, auth_ctx)
        return [corpus.upper() for corpus in corpora if protection_info[corpus].protected]

    async def check_authorization(
        self, corpora: list[str], auth_ctx: AuthContext
    ) -> tuple[bool, list[str], str | None]:
        """Check if the user has access to the specified corpora based on JWT scopes.

        Args:
            corpora: A list of corpora to check access for.
            auth_ctx: The authentication context containing the request and other info.

        Returns:
            A tuple containing:
                - A boolean indicating if access is granted.
                - A list of unauthorized corpora (if access is denied).
                - An optional message (e.g., for errors).
        """
        corpora_upper = [corpus.upper() for corpus in corpora]
        protection_info = await self._get_protection_info(corpora_upper, auth_ctx)
        protected_requested = [corpus for corpus in corpora_upper if protection_info[corpus].protected]
        if protected_requested:
            user_corpora = []

            # Get authorization header
            auth_header = auth_ctx.request.headers.get("Authorization")
            if auth_header and " " in auth_header:
                auth_token = auth_header.split(" ")[1]

                # Parse JWT
                if not self.jwt_key:
                    return False, [], "JWT public key is not configured."
                try:
                    user_token = jwt.decode(auth_token, key=self.jwt_key, algorithms=["RS256"])
                except jwt.ExpiredSignatureError:
                    return False, [], "The provided JWT has expired"
                except jwt.InvalidTokenError:
                    return False, [], "Could not validate the provided JWT."

                if user_token.get("exp") and user_token["exp"] < time.time():
                    return False, [], "The provided JWT has expired"

                user_corpora.extend(corpus.upper() for corpus in user_token.get("scope", {}).get("corpora", {}))

            unauthorized = [corpus for corpus in protected_requested if corpus not in user_corpora]
            if unauthorized:
                return False, unauthorized, None
        return True, [], None

    @cached_property
    def jwt_key(self) -> str | None:
        """Return the public key for validating JWTs."""
        pubkey_file = bp.config("pubkey_file")
        if not pubkey_file:
            return None
        return Path(pubkey_file).read_text(encoding="utf-8")


AUTHORIZER_CLASS = AuthJWT
