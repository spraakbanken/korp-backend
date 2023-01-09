# Changelog

## [8.2.0] - Unreleased

### Added

- Rudimentary plugin system added (will probably be replaced by something better in the future).
- Added support for authorization plugins.
- Improved caching for `/count`.
- Now aborts searches if client disconnects.

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
