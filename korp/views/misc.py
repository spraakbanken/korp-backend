from flask import Blueprint

from korp import utils

bp = Blueprint("misc", __name__)


@bp.route("/optimize", methods=["GET", "POST"])
@utils.main_handler
def optimize(args):
    utils.assert_key("cqp", args, r"", True)

    cqpparams = {"within": args.get("within") or "sentence"}
    if args.get("cut"):
        cqpparams["cut"] = args["cut"]

    free_search = not utils.parse_bool(args, "in_order", True)

    cqp = args["cqp"]
    result = {"cqp": utils.query_optimize(cqp, cqpparams, find_match=False, expand=False, free_search=free_search)}
    yield result
