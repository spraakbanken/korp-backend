"""List of all API routers."""

from korp.api.routers import (
    attribute_values,
    cache,
    concordance,
    corpus_config,
    dependency_relations,
    frequencies,
    info,
    lexeme_counts,
    log_likelihood,
    misc,
    token_distribution,
)

routers = [
    attribute_values.router,
    cache.router,
    concordance.router,
    corpus_config.router,
    dependency_relations.router,
    frequencies.router,
    info.router,
    lexeme_counts.router,
    log_likelihood.router,
    misc.router,
    token_distribution.router,
]
