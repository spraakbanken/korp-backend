"""Authorization using JWT.

This plugin checks if the user has access to protected corpora based on JWT token scopes. It retrieves the list of
protected corpora from CWB info and checks the user's JWT token for the allowed corpora in the scopes. If the user
tries to access a protected corpus that is not in their allowed scopes, access is denied.

The plugin expects the JWT to be provided in the Authorization header as a Bearer token, and it uses a public key
to validate the token. The public key file can be configured using the "pubkey_file" setting in the plugin
configuration.

To use this plugin, you need to install the `pyjwt[crypto]` package.
"""

import time
from functools import cached_property
from pathlib import Path

import jwt  # type: ignore

from korp import utils

bp = utils.Plugin("auth_jwt", __name__)


class AuthJWT(utils.Authorizer):
    """Authorizer plugin using JWT token scopes."""

    async def get_protected_corpora(self, auth_ctx: utils.AuthContext) -> list[str]:
        """Get list of corpora with restricted access.

        Args:
            auth_ctx: The authentication context.

        Returns:
            A list of protected corpora.
        """
        key = None
        if auth_ctx.cache_enabled:
            key = f"protected:{await utils.cache_prefix(self.cache)}"
            result = await self.cache.get(key)
            if result is not None:
                return result

        corpora_lines = self.cwb.run_cqp("show corpora;")
        next(corpora_lines, None)  # Skip version number

        protected_corpora = [corpus.upper() for corpus in corpora_lines if self._is_protected(corpus)]

        if auth_ctx.cache_enabled and key:
            await self.cache.add(key, protected_corpora)
        return protected_corpora

    def _is_protected(self, corpus: str) -> bool:
        """Check whether a corpus is marked as protected in CWB info.

        Args:
            corpus: The name of the corpus to check.

        Returns:
            True if the corpus is protected, False otherwise.
        """
        lines = self.cwb.run_cqp([f"{corpus};", "info; .EOL.;", "exit;"])
        next(lines, None)  # Skip version number

        for line in lines:
            if line == utils.END_OF_LINE:
                break
            if ":" in line and not line.endswith(":"):
                key, value = (part.strip() for part in line.split(":", 1))
                if key == "Protected":
                    return value.lower() == "true"
        return False

    async def check_authorization(
        self, corpora: list[str], auth_ctx: utils.AuthContext
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
        protected = await self.get_protected_corpora(auth_ctx)
        if protected:
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

            unauthorized = [c.upper() for c in corpora if c.upper() in protected and c.upper() not in user_corpora]
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
