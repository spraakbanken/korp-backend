# Changelog

## [Unreleased]

Version 9 is a breaking rewrite of the backend and its HTTP API.

### Added

- Rebuilt the application as an ASGI service using FastAPI.
- Added strict JSON request bodies to data-retrieval routes. Short requests can still use GET query parameters; POST is
  available for corpus lists, CQP queries, and other inputs that may exceed URL limits.
- Added typed response contracts for every core route and typed NDJSON event contracts for streamed responses.
- Added `/health` for lightweight service health checks.
- Added `/dependency-relations/time` and `/dependency-relations/time/sentences` for time-sliced dependency-relation
  statistics and their example sentences.
- Added optional per-route rate limiting, including `429` responses and configurable rate-limit headers.
- Added configurable CORS support and `ROOT_PATH` support for deployments below a URL prefix.
- Added authorization-aware OpenAPI security schemes for corpus-protected operations.
- Added HTTP client-cache policies. Anonymous responses can be public; authorization plugins can opt into private
  credential-varying caches; unsafe, debug, streamed, and administrative responses use `no-store`.
- Added request and database slow-operation logging, database connection/read/pool timeouts, and configurable error
  traceback exposure.
- Added extensive unit, contract, plugin, and functional tests (thanks to @janiemi for the original test contribution).

### Changed

- Dropped support for Python versions older than 3.11 and moved packaging and dependency management to `pyproject.toml`
  and `uv`.
- Replaced Flask, gevent, and the WSGI entrypoint with FastAPI and a native ASGI deployment. Production installations
  use Gunicorn's ASGI worker; development uses the FastAPI CLI.
- Replaced synchronous `mysqlclient` access with async SQLAlchemy and `asyncmy`, and replaced `pymemcache` with
  `aiomcache`.
- Configuration now uses Pydantic settings loaded from environment variables or `.env`, rather than
  `instance/config.py`.
- Renamed database settings from `DBHOST`, `DBPORT`, `DBNAME`, `DBUSER`, and `DBPASSWORD` to `DB_HOST`, `DB_PORT`,
  `DB_NAME`, `DB_USER`, and `DB_PASSWORD`, and added `DB_CHARSET`.
- Renamed `DBWPTABLE` to `DB_DEPENDENCY_RELATIONS_TABLE_PREFIX`, replaced the hard-coded `lemgram_index` table with the
  configurable `DB_LEXEME_COUNTS_TABLE` (default `lexeme_counts`), and renamed `CACHE_MAX_QUERY_DATA` to
  `CACHE_MAX_CONCORDANCE_CACHE_SIZE`.
- Memcached configuration now accepts a host and port, not a Unix socket. Caching is active only when Memcached is
  configured and `CACHE_DIR` exists.
- Plugins now export FastAPI `APIRouter` instances. An authorization plugin may additionally export one
  `AUTHORIZER_CLASS`; plugin configuration can be loaded from YAML and overridden from `.env`.
- Built-in authorization plugins read the `Protected` value from CWB corpus metadata.
- Route and parameter names now use descriptive kebab-case/plural vocabulary. The main mappings are `/query` to
  `/concordance`, `/count*` to `/frequencies*`, `/timespan` to `/token-distribution`, `/loglike` to `/log-likelihood`,
  `/lemgram_count` to `/lexeme-counts`, `/attr_values` to `/attribute-values`, `/corpus_*` to `/corpora/*`, and the
  `/relations*` family to `/dependency-relations*`.
- Numbered `cqp1`/`cqp2` and `subcqp0`/`subcqp1` inputs are replaced by repeated `cqp` and `subcqp` query parameters, or
  ordered JSON arrays in POST bodies.
- Pagination uses `offset` plus a row count in `limit`, rather than inclusive `start` and `end`. An omitted `limit` or
  `max_results` means unlimited where supported; zero is no longer a magic unlimited value.
- `incremental` is replaced by `stream`. `stream=true` returns `application/x-ndjson` events (`progress`, `result`,
  `error`, `keepalive`, and `complete`) rather than incrementally assembling one JSON document.
- Ordinary responses are buffered without whitespace keepalives. Failures use their HTTP 4xx/5xx status and one Problem
  Details-style JSON shape with stable error codes; only errors after an NDJSON stream starts remain HTTP 200 stream
  events.
- Request validation is stricter: unknown parameters and JSON fields are rejected, and JSON collection fields must be
  arrays.
- Corpus IDs are lowercase throughout responses, configuration, authorization, and cache keys.
- Removed JSONP output and the legacy `callback` and `encoding` controls; responses are UTF-8 JSON or NDJSON.
- Common response timing is now `elapsed` instead of `time`, and optional debug output is `debug` instead of `DEBUG`.
- Concordance responses use `total_hits`, `hits_by_corpus`, `pagination_state`, and the `matches` field is now always an
  array.
- Frequency responses always use arrays for the main query plus subqueries, put pre-pagination `total_rows` on each
  statistics object, and represent all grouped attribute values as arrays.
- Time-based responses use typed period arrays with inclusive ISO 8601 `start`/`end` boundaries instead of dynamic
  period keys. Undated material uses `{ "dated": false, ... }`.
- Corpus metadata uses `positional`, `structural`, and `alignment` instead of the old single letter abbreviations; known
  CWB metadata is typed and snake_case, while installation-specific keys are preserved in `additional`.
- Log-likelihood results are row objects containing `value`, `score`, `set1`, and `set2`, rather than four parallel
  maps.
- Dependency relation fields use descriptive names (`relation`, `dependent`, `dependent_pos`, `dependent_extra`, and
  `sources`).
- Attribute-value responses always include every requested attribute.
- Cache refresh moved from `GET|POST /cache` to `POST /admin/cache/refresh` and now reports whether server-side caching
  is enabled.

### Fixed

- Fixed free-order concordance queries returning results beyond the requested page.
- Added a check for structural attributes containing tab characters.
- Fixed frequency calculations using `relative_to_struct`, including multi-token queries.
- Fixed log-likelihood grouping by multiple positional or structural attributes.

### Removed

- Removed `/struct_values`; use `/attribute-values` with the `attributes` parameter.
- Removed the `/` alias for `/info`.
- Removed form-encoded POST requests. POST operations now accept strict `application/json` bodies.

## [8.2.0] - 2024-05-16

### Added

- Rudimentary plugin system added (will probably be replaced by something better in the future).
- Added support for authorization plugins.
- Improved caching for `/count`.
- Now aborts searches if client disconnects.
- Added `CACHE_MAX_QUERY_DATA` config variable, setting an optional max size per cached query data file.

### Changed

- Code was refactored into more manageable pieces.
- Now prevents `/corpus_config` timeout.
- Switched to using `pymemcache` instead of `pylibmc`.
- Speeded up loading of YAML config files.
- Optimized word picture SQL query. Now much faster!

### Fixed

- Fixed crashes during cache cleaning.
- Fixed longstanding bug in timespan caching.
- Fixed crash when there are no corpus config files.
- Fixed crash when trying to access non-existent mode in corpus_config.
- Fixed crash when using semicolons at the end of CQP queries.
- Fixed bug in `/count`, where a pipe would be returned for unannotated tokens when `top` was used.
- Fixed caching bug in `/count` leading to wrong relative total.

## [8.1.0] - 2022-09-14

### Added

- Added `/corpus_config` endpoint, for serving corpus configuration used by the Korp frontend.
- Added `per_corpus` and `combined` parameters to `/count_time`.
- Added more information about word picture data to readme.

### Fixed

- Fixed some crashes related to caching.
- Fix bug in `/count_time` when no corpora are within date range.
- Made cache invalidation more reliable.

## [8.0.0] - 2019-09-05

### Added

- Added OpenAPI specification.

### Changed

- `/info` has been split into two endpoints: `/info` and `/corpus_info`.
- New improved `/count` format.
- `/loglike` parameters `group_by` and `group_by_struct` are now optional.
- Removed backward compatible parameters for all endpoints.
- Better representation of structural attributes on token level in `/query`.

[8.2.0]: https://github.com/spraakbanken/korp-backend/releases/tag/v8.2.0
