"""List of all API routers."""

from korp.api.routers import (
    attribute_values,
    cache,
    concordance,
    corpus_config,
    count,
    dependency_relations,
    info,
    lexeme_counts,
    log_likelihood,
    misc,
    timespan,
)

routers = [
    attribute_values.router,
    cache.router,
    concordance.router,
    corpus_config.router,
    count.router,
    dependency_relations.router,
    info.router,
    lexeme_counts.router,
    log_likelihood.router,
    misc.router,
    timespan.router,
]
