"""List of all API routers."""

from korp.api.routers import (
    attr_values,
    cache,
    corpus_config,
    count,
    info,
    lexeme_count,
    log_likelihood,
    misc,
    query,
    timespan,
    word_picture,
)

routers = [
    attr_values.router,
    cache.router,
    corpus_config.router,
    count.router,
    info.router,
    lexeme_count.router,
    log_likelihood.router,
    misc.router,
    query.router,
    timespan.router,
    word_picture.router,
]
