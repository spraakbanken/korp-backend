"""List of all API routers."""

from korp.api.routers import (
    attr_values,
    cache,
    concordance,
    corpus_config,
    count,
    info,
    lexeme_counts,
    log_likelihood,
    misc,
    timespan,
    word_picture,
)

routers = [
    attr_values.router,
    cache.router,
    concordance.router,
    corpus_config.router,
    count.router,
    info.router,
    lexeme_counts.router,
    log_likelihood.router,
    misc.router,
    timespan.router,
    word_picture.router,
]
