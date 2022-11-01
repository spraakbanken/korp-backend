"""Authorization using JWT."""

import time
from pathlib import Path
from typing import List, Tuple, Optional

import jwt  # From pyjwt[crypto]
from flask import current_app as app
from flask import request

from korp import cwb, utils
from korp import memcached
from korp.views import info

bp = utils.Plugin("wsauth", __name__)


class WSAuth(utils.Authorizer):

    def __init__(self):
        self._pubkey = None

    def get_protected_corpora(self, use_cache: bool = True) -> List[str]:
        """Get list of corpora with restricted access."""
        if use_cache:
            with memcached.get_client() as mc:
                key = f"protected:{utils.cache_prefix(mc)}"
                result = mc.get(key)
            if result is not None:
                return result

        # Get list of all corpora from CWB
        corpora = cwb.run_cqp("show corpora;")
        next(corpora)  # Skip version number
        corpus_info = utils.generator_to_dict(info.corpus_info({"corpus": list(corpora)}))
        protected_corpora = []
        for corpus, c_info in corpus_info["corpora"].items():
            if c_info["info"].get("Protected", "false").lower() == "true":
                protected_corpora.append(corpus.upper())

        if use_cache:
            with memcached.get_client() as mc:
                mc.add(key, protected_corpora)
        return protected_corpora

    def check_authorization(self, corpora: List[str]) -> Tuple[bool, List[str], Optional[str]]:
        protected = self.get_protected_corpora()
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
