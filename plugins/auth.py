"""Authentication and authorization using SB-auth.

This plugin implements a simple authentication mechanism using an external authentication server. It checks the user's
credentials against the authentication server and retrieves the list of corpora the user has access to.

To determine which corpora are protected, it checks the CWB info for each corpus. A corpus is considered protected if
it has the "Protected" key set to "true" in its CWB `.info` file.
"""

import base64
import hashlib
import json
import urllib.error
import urllib.parse
import urllib.request

from korp import auth, caching, cqp, plugin
from korp.dependencies import AuthContext, CtxDep
from korp.handler import api_handler

router = plugin.Plugin("authenticate", __name__)


@router.get("/authenticate", response_model=None)
@router.post("/authenticate", response_model=None, include_in_schema=False)
@api_handler(cache_headers=False)
def authenticate(ctx: CtxDep) -> dict:
    """Authenticate a user against an authentication server.

    Args:
        ctx: Request context.

    Returns:
        A dictionary containing the list of corpora the user has access to, or an empty dictionary if authentication
            fails.
    """
    auth_header = ctx.request.headers.get("Authorization")
    return _authenticate_from_auth_header(auth_header)


def _authenticate_from_auth_header(auth_header: str | None) -> dict:
    """Authenticate from an Authorization header value.

    Args:
        auth_header: The value of the Authorization header.

    Returns:
        A dictionary containing the list of corpora the user has access to, or an empty dictionary if authentication
            fails.

    Raises:
        KorpAuthorizationError: If there is an error during authentication (e.g., contacting the authentication server).
    """
    auth_data = None
    if auth_header and auth_header.lower().startswith("basic "):
        try:
            encoded = auth_header.split(" ", 1)[1]
            decoded = base64.b64decode(encoded).decode("utf-8")
            username, password = decoded.split(":", 1)
            auth_data = {"username": username, "password": password}
        except Exception:
            auth_data = None

    if auth_data:
        postdata = {
            "username": auth_data["username"],
            "password": auth_data["password"],
            "checksum": hashlib.md5(
                bytes(auth_data["username"] + auth_data["password"] + router.config("AUTH_SECRET"), "utf-8")
            ).hexdigest(),
        }

        try:
            contents = (
                urllib.request.urlopen(router.config("AUTH_SERVER"), urllib.parse.urlencode(postdata).encode("utf-8"))
                .read()
                .decode("utf-8")
            )
            auth_response = json.loads(contents)
        except urllib.error.HTTPError:
            raise auth.KorpAuthorizationError("Could not contact authentication server.") from None
        except ValueError:
            raise auth.KorpAuthorizationError("Invalid response from authentication server.") from None
        except Exception:
            raise auth.KorpAuthorizationError("Unexpected error during authentication.") from None

        if auth_response["authenticated"]:
            permitted_resources = auth_response["permitted_resources"]
            result = {"corpora": []}
            if "corpora" in permitted_resources:
                for c in permitted_resources["corpora"]:
                    if permitted_resources["corpora"][c]["read"]:
                        result["corpora"].append(c.upper())
            return result

    return {}


class Auth(auth.Authorizer):
    """Authorizer class that checks if the user has access to protected corpora based on the authentication response."""

    async def get_protected_corpora(self, auth_ctx: AuthContext) -> list[str]:
        """Get list of protected corpora.

        Args:
            auth_ctx: The authentication context.

        Returns:
            A list of protected corpora.
        """
        key = None
        if auth_ctx.cache_enabled:
            key = f"protected:{await caching.cache_prefix(self.cache)}"
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
            `True` if the corpus is protected, `False` otherwise.
        """
        lines = self.cwb.run_cqp([f"{corpus};", "info; .EOL.;", "exit;"])
        next(lines, None)  # Skip version number

        for line in lines:
            if line == cqp.END_OF_LINE:
                break
            if ":" in line and not line.endswith(":"):
                key, value = (part.strip() for part in line.split(":", 1))
                if key == "Protected":
                    return value.lower() == "true"
        return False

    async def check_authorization(
        self, corpora: list[str], auth_ctx: AuthContext
    ) -> tuple[bool, list[str], str | None]:
        """Take a list of corpora, and check if the user has access to them.

        Args:
            corpora: A list of corpora to check access for.
            auth_ctx: The authentication context.

        Returns:
            A tuple containing:
                - A boolean indicating if access is granted.
                - A list of unauthorized corpora (if access is denied).
                - An optional message (not used in this implementation).
        """
        protected = await self.get_protected_corpora(auth_ctx)
        c = [c for c in corpora if c.upper() in protected]
        if c:
            auth = _authenticate_from_auth_header(auth_ctx.request.headers.get("Authorization"))
            unauthorized = [x for x in c if x.upper() not in auth.get("corpora", [])]
            if not auth or unauthorized:
                return False, unauthorized, None
        return True, [], None


AUTHORIZER_CLASS = Auth
