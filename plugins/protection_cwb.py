"""Helpers for CWB-based corpus protection metadata."""

from typing import List

from korp import cwb, utils
from korp.views import info


def get_protected_corpora(corpora: List[str], use_cache: bool = True) -> List[str]:
    """Return requested corpora whose CWB info has `Protected: true`."""
    if not corpora:
        return []

    args = {"corpus": [corpus.upper() for corpus in corpora]}
    if not use_cache:
        args["cache"] = False
    corpus_info = utils.generator_to_dict(
        info.corpus_info(args, no_combined_cache=True)
    )

    return [
        corpus.upper()
        for corpus, data in corpus_info["corpora"].items()
        if data["info"].get("Protected", "false").lower() == "true"
    ]


def get_all_protected_corpora(use_cache: bool = True) -> List[str]:
    """Return all corpora whose CWB info has `Protected: true`."""
    corpora = cwb.run_cqp("show corpora;")
    next(corpora)  # Skip CQP version
    return get_protected_corpora(list(corpora), use_cache)
