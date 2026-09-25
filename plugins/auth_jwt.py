"""Authorization using JWT."""

import time
from pathlib import Path
from typing import List, Tuple, Optional

import jwt  # From pyjwt[crypto]
from flask import current_app as app
from flask import request

from korp import utils
from plugins import protection_cwb

bp = utils.Plugin("auth_jwt", __name__)


class AuthJWT(utils.Authorizer):

    def __init__(self):
        self._pubkey = None

    def get_protected_corpora(self, use_cache: bool = True) -> List[str]:
        """Get list of corpora with restricted access."""
        return protection_cwb.get_all_protected_corpora(use_cache)

    def check_authorization(self, corpora: List[str]) -> Tuple[bool, List[str], Optional[str]]:
        protected = protection_cwb.get_protected_corpora(corpora)
        if protected:
            user_corpora = []

            # Get authorization header
            auth_header = request.headers.get("Authorization")
            if auth_header and " " in auth_header:
                auth_token = auth_header.split(" ")[1]

                # Parse JWT
                user_token = jwt.decode(auth_token, key=self.jwt_key, algorithms=["RS256"])
                if user_token["exp"] < time.time():
                    return False, [], "The provided JWT has expired"

                for corpus, level in user_token.get("scope", {}).get("corpora", {}).items():
                    user_corpora.append(corpus.upper())

            unauthorized = [c.upper() for c in corpora if c.upper() in protected and c.upper() not in user_corpora]
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
