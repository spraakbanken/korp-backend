"""Authentication and authorization using SB-auth."""
import hashlib
import json
import urllib.error
import urllib.parse
import urllib.request
from typing import List, Tuple, Optional

from flask import request

from korp import utils

bp = utils.Plugin("authenticate", __name__)


@bp.route("/authenticate", methods=["GET", "POST"])
@utils.main_handler
def authenticate(_=None):
    """Authenticate a user against an authentication server."""

    auth_data = request.authorization

    if auth_data:
        postdata = {
            "username": auth_data["username"],
            "password": auth_data["password"],
            "checksum": hashlib.md5(bytes(auth_data["username"] + auth_data["password"] +
                                          bp.config("AUTH_SECRET"), "utf-8")).hexdigest()
        }

        try:
            contents = urllib.request.urlopen(bp.config("AUTH_SERVER"),
                                              urllib.parse.urlencode(postdata).encode("utf-8")).read().decode("utf-8")
            auth_response = json.loads(contents)
        except urllib.error.HTTPError:
            raise utils.KorpAuthorizationError("Could not contact authentication server.")
        except ValueError:
            raise utils.KorpAuthorizationError("Invalid response from authentication server.")
        except:
            raise utils.KorpAuthorizationError("Unexpected error during authentication.")

        if auth_response["authenticated"]:
            permitted_resources = auth_response["permitted_resources"]
            result = {"corpora": []}
            if "corpora" in permitted_resources:
                for c in permitted_resources["corpora"]:
                    if permitted_resources["corpora"][c]["read"]:
                        result["corpora"].append(c.upper())
            yield result
            return

    yield {}


class SBAuth(utils.Authorizer):

    def __init__(self):
        self._protected = []

    def get_protected_corpora(self, use_cache: bool = True):
        """Get list of protected corpora."""
        if bp.config("PROTECTED_FILE"):
            with open(bp.config("PROTECTED_FILE")) as infile:
                return [x.strip() for x in infile.readlines()]
        else:
            return []

    def check_authorization(self, corpora: List[str]) -> Tuple[bool, List[str], Optional[str]]:
        """Take a list of corpora, and check if the user has access to them."""

        if bp.config("PROTECTED_FILE"):
            protected = self.get_protected_corpora()
            c = [c for c in corpora if c.upper() in protected]
            if c:
                auth = utils.generator_to_dict(authenticate({}))
                unauthorized = [x for x in c if x.upper() not in auth.get("corpora", [])]
                if not auth or unauthorized:
                    return False, unauthorized, None
        return True, [], None
