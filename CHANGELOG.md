# Changelog

## [Unreleased]

This release includes a major refactor of the codebase, switching to the FastAPI framework and improving type hinting
and documentation.

### Added

- Migrated the codebase from Flask to FastAPI.
- Migrated database access to async SQLAlchemy.
- Migrated caching to use `aiomcache` for async Memcached access.
- Improved type hinting and documentation throughout the codebase.
- Added support for client-side caching via HTTP Cache-Control headers.
- Added tests (thanks to @janiemi for the contribution!).
- Added a new `/relations_time` endpoint for retrieving Word Picture data over time.

### Changed

- Dropped support for Python versions older than 3.11.
- Better error message when structural attributes contain tabs.

### Fixed

- Made corrections to the API documentation.
- Fixed bug when using the `/count` parameter `relative_to_struct` with multiple tokens.

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

[8.3.0]: https://github.com/spraakbanken/korp-backend/releases/tag/v8.3.0
[8.2.0]: https://github.com/spraakbanken/korp-backend/releases/tag/v8.2.0
