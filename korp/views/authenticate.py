import hashlib
import json
import urllib.error
import urllib.parse
import urllib.request

from flask import Blueprint, request
from flask import current_app as app

from korp import utils

bp = Blueprint("authenticate", __name__)


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
                                          app.config["AUTH_SECRET"], "utf-8")).hexdigest()
        }

        try:
            contents = urllib.request.urlopen(app.config["AUTH_SERVER"],
                                              urllib.parse.urlencode(postdata).encode("utf-8")).read().decode("utf-8")
            auth_response = json.loads(contents)
        except urllib.error.HTTPError:
            raise utils.KorpAuthenticationError("Could not contact authentication server.")
        except ValueError:
            raise utils.KorpAuthenticationError("Invalid response from authentication server.")
        except:
            raise utils.KorpAuthenticationError("Unexpected error during authentication.")

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
