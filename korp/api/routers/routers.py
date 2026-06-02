"""List of all API routers."""

from korp.api.routers import (
    attr_values,
    cache,
    corpus_config,
    count,
    info,
    lexeme_count,
    loglike,
    misc,
    query,
    relations,
    timespan,
)

routers = [
    attr_values.router,
    cache.router,
    corpus_config.router,
    count.router,
    info.router,
    lexeme_count.router,
    loglike.router,
    misc.router,
    query.router,
    relations.router,
    timespan.router,
]
