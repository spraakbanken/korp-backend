import math
from collections import defaultdict

from flask import Blueprint

from korp import utils
from . import count

bp = Blueprint("loglike", __name__)


@bp.route("/loglike", methods=["GET", "POST"])
@utils.main_handler
@utils.prevent_timeout
def loglike(args):
    """Do a log-likelihood comparison on two queries."""
    def expected(total, wordtotal, sumtotal):
        """ The expected is that the words are uniformely distributed over the corpora. """
        return wordtotal * (float(total) / sumtotal)

    def compute_loglike(wf1_tot1, wf2_tot2):
        """ Compute log-likelihood for a single pair. """
        wf1, tot1 = wf1_tot1
        wf2, tot2 = wf2_tot2
        e1 = expected(tot1, wf1 + wf2, tot1 + tot2)
        e2 = expected(tot2, wf1 + wf2, tot1 + tot2)
        (l1, l2) = (0, 0)
        if wf1 > 0:
            l1 = wf1 * math.log(wf1 / e1)
        if wf2 > 0:
            l2 = wf2 * math.log(wf2 / e2)
        loglike = 2 * (l1 + l2)
        return round(loglike, 2)

    def compute_list(d1, tot1, ref, reftot):
        """ Compute log-likelyhood for lists. """
        result = []
        all_w = set(d1.keys()).union(set(ref.keys()))
        for w in all_w:
            ll = compute_loglike((d1.get(w, 0), tot1), (ref.get(w, 0), reftot))
            result.append((ll, w))
        result.sort(reverse=True)
        return result

    def compute_ll_stats(ll_list, count, sets):
        """ Calculate max, min, average, and truncates word list. """
        tot = len(ll_list)
        new_list = []

        set1count, set2count = 0, 0
        for ll_w in ll_list:
            ll, w = ll_w

            if (sets[0]["freq"].get(w) and not sets[1]["freq"].get(w)) or sets[0]["freq"].get(w) and (
                    sets[0]["freq"].get(w, 0) / (sets[0]["total"] * 1.0)) > (
                    sets[1]["freq"].get(w, 0) / (sets[1]["total"] * 1.0)):
                set1count += 1
                if set1count <= count or not count:
                    new_list.append((ll * -1, w))
            else:
                set2count += 1
                if set2count <= count or not count:
                    new_list.append((ll, w))

            if count and (set1count >= count and set2count >= count):
                break

        nums = [ll for (ll, _) in ll_list]
        return (
            new_list,
            round(sum(nums) / float(tot), 2) if tot else 0.0,
            min(nums) if nums else 0.0,
            max(nums) if nums else 0.0
        )

    utils.assert_key("set1_cqp", args, r"", True)
    utils.assert_key("set2_cqp", args, r"", True)
    utils.assert_key("set1_corpus", args, r"", True)
    utils.assert_key("set2_corpus", args, r"", True)
    utils.assert_key("group_by", args, utils.IS_IDENT, False)
    utils.assert_key("group_by_struct", args, utils.IS_IDENT, False)
    utils.assert_key("ignore_case", args, utils.IS_IDENT)
    utils.assert_key("max", args, utils.IS_NUMBER, False)

    maxresults = int(args.get("max") or 15)

    set1 = args.get("set1_corpus").upper()
    if isinstance(set1, str):
        set1 = set1.split(utils.QUERY_DELIM)
    set1 = set(set1)
    set2 = args.get("set2_corpus").upper()
    if isinstance(set2, str):
        set2 = set2.split(utils.QUERY_DELIM)
    set2 = set(set2)

    corpora = set1.union(set2)
    utils.check_authorization(corpora)

    same_cqp = args.get("set1_cqp") == args.get("set2_cqp")

    result = {}

    # If same CQP for both sets, handle as one query for better performance
    if same_cqp:
        args["cqp"] = args.get("set1_cqp")
        args["corpus"] = utils.QUERY_DELIM.join(corpora)
        count_result = utils.generator_to_dict(count.count(args))

        sets = [{"total": 0, "freq": defaultdict(int)}, {"total": 0, "freq": defaultdict(int)}]
        for i, cset in enumerate((set1, set2)):
            for corpus in cset:
                sets[i]["total"] += count_result["corpora"][corpus]["sums"]["absolute"]
                if len(cset) == 1:
                    sets[i]["freq"] = dict((tuple(
                        (y[0], y[1] if isinstance(y[1], tuple) else (y[1],)) for y in sorted(x["value"].items())),
                                            x["absolute"])
                                           for x in count_result["corpora"][corpus]["rows"])
                else:
                    for w, f in ((tuple(
                            (y[0], y[1] if isinstance(y[1], tuple) else (y[1],)) for y in sorted(x["value"].items())),
                                  x["absolute"])
                                 for x in count_result["corpora"][corpus]["rows"]):
                        sets[i]["freq"][w] += f

    else:
        args1, args2 = args.copy(), args.copy()
        args1["corpus"] = utils.QUERY_DELIM.join(set1)
        args1["cqp"] = args.get("set1_cqp")
        args2["corpus"] = utils.QUERY_DELIM.join(set2)
        args2["cqp"] = args.get("set2_cqp")
        count_result = [utils.generator_to_dict(count.count(args1)), utils.generator_to_dict(count.count(args2))]

        sets = [{}, {}]
        for i, cset in enumerate((set1, set2)):
            sets[i]["total"] = count_result[i]["combined"]["sums"]["absolute"]
            sets[i]["freq"] = dict((tuple(
                (y[0], y[1] if isinstance(y[1], tuple) else (y[1],)) for y in sorted(x["value"].items())),
                                    x["absolute"])
                                   for x in count_result[i]["combined"]["rows"])

    ll_list = compute_list(sets[0]["freq"], sets[0]["total"], sets[1]["freq"], sets[1]["total"])
    (ws, avg, mi, ma) = compute_ll_stats(ll_list, maxresults, sets)

    result["loglike"] = {}
    result["average"] = avg
    result["set1"] = {}
    result["set2"] = {}

    for (ll, w) in ws:
        w_formatted = " ".join(w[0][1])
        result["loglike"][w_formatted] = ll
        result["set1"][w_formatted] = sets[0]["freq"].get(w, 0)
        result["set2"][w_formatted] = sets[1]["freq"].get(w, 0)

    yield result
