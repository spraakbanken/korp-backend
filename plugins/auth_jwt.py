"""Authorization using JWT.

For corpora with Protected: true in the .info file:
- If corpus is in JWT scope.corpora (explicit grant), it's authorized
- Otherwise, if License field is present in .info, check if JWT userClasses contains that license
  (e.g., License: ACA → requires "ACA" in jwt["userClasses"], License: ACA-Fi → requires "ACA-Fi" in jwt["userClasses"])
- Otherwise, it's not authorized

"""

import sys
from pathlib import Path
from typing import List, Tuple, Optional

import jwt  # From pyjwt[crypto]
from flask import current_app as app
from flask import request

from korp import cwb, utils
from korp import memcached
from korp.views import info

bp = utils.Plugin("auth_jwt", __name__)


class AuthJWT(utils.Authorizer):

    def __init__(self):
        self._pubkey = None

    def get_protected_corpora(self, use_cache: bool = True) -> List[str]:
        """Get list of corpora with restricted access.

        Returns all corpora where Protected: true (regardless of License type).
        """
        # Only use the cache if memcached was successfully initialized.
        use_cache = use_cache and memcached.active
        key = None

        if use_cache:
            try:
                with memcached.get_client() as mc:
                    key = f"protected:{utils.cache_prefix(mc)}"
                    result = mc.get(key)
                if result is not None:
                    return result
            except Exception as e:
                print(
                    f"Warning: auth_jwt: memcached read failed, falling back to uncached lookup: {e}",
                    file=sys.stderr,
                )
                use_cache = False

        # Get list of all corpora from CWB
        corpora = cwb.run_cqp("show corpora;")
        next(corpora)  # Skip version number
        corpus_info = utils.generator_to_dict(info.corpus_info({"corpus": list(corpora)}))
        protected_corpora = []
        for corpus, c_info in corpus_info["corpora"].items():
            protected_value = c_info["info"].get("Protected", "").lower()
            if protected_value in ("true", "yes"):
                protected_corpora.append(corpus.upper())

        if use_cache and key is not None:
            try:
                with memcached.get_client() as mc:
                    mc.add(key, protected_corpora)
            except Exception as e:
                print(f"Warning: auth_jwt: memcached write failed: {e}", file=sys.stderr)
        return protected_corpora

    def check_authorization(self, corpora: List[str]) -> Tuple[bool, List[str], Optional[str]]:
        """Check if user is authorized to access the given corpora.

        For corpora with License field: check if JWT userClasses contains that license.
        For corpora without License field: check if corpus is in JWT scope.corpora.

        Returns:
            Tuple of (success, unauthorized_corpora, error_message)
        """
        protected_set = set(self.get_protected_corpora())

        # Find which requested corpora need authorization
        corpora_to_check = [c.upper() for c in corpora if c.upper() in protected_set]

        if not corpora_to_check:
            return True, [], None

        # Parse JWT if present
        user_token = None
        user_scope_corpora = set()

        auth_header = request.headers.get("Authorization")
        if auth_header and " " in auth_header:
            auth_token = auth_header.split(" ")[1]

            # Parse JWT (expiry is handled by the auth server, not here)
            user_token = jwt.decode(auth_token, key=self.jwt_key, algorithms=["RS256"],
                                    options={"verify_exp": False})

            # Collect user's granted corpora from scope
            for corpus in user_token.get("scope", {}).get("corpora", {}).keys():
                user_scope_corpora.add(corpus.upper())

        # Get license info for protected corpora
        corpus_info = utils.generator_to_dict(info.corpus_info({"corpus": corpora_to_check}))

        # Check authorization for each corpus
        unauthorized = []
        for corpus_upper in corpora_to_check:
            if corpus_upper in user_scope_corpora:
                continue
            license_value = (
                corpus_info.get("corpora", {})
                .get(corpus_upper, {})
                .get("info", {})
                .get("License", "")
            )
            if license_value and user_token and license_value in user_token.get("userClasses", []):
                continue
            unauthorized.append(corpus_upper)

        if unauthorized:
            return False, unauthorized, None
        return True, [], None

    @property
    def jwt_key(self):
        """Return the public key for validating JWTs."""
        if not self._pubkey:
            if bp.config("pubkey_file"):
                self._pubkey = open(Path(app.instance_path) / bp.config("pubkey_file")).read()
        return self._pubkey
