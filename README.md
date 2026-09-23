# Korp Backend

This is the backend for [Korp](https://spraakbanken.gu.se/korp), a corpus search tool developed by
[Språkbanken](https://spraakbanken.gu.se) at the University of Gothenburg, Sweden.

The code is distributed under the [MIT license](https://opensource.org/licenses/MIT).

The Korp backend is a Python 3 ASGI application built with FastAPI, acting as a wrapper for [Corpus
Workbench](https://cwb.sourceforge.io/).

To see what has changed in recent versions, see the [CHANGELOG](CHANGELOG.md).

## Requirements

To use the basic features of the Korp backend you need the following:

- [Python 3.11+](https://python.org/)
- [Corpus Workbench](https://cwb.sourceforge.io/) (CWB) 3.4.12 or newer

To use database-backed features such as dependency relations, lexeme counts, and time-based statistics, you also need:

- [MariaDB](https://mariadb.org/) or [MySQL](https://www.mysql.com/)

For optional (but strongly recommended) caching you need:

- [Memcached](https://memcached.org/)

## Installing the required software

These instructions assume you are running a UNIX-like operating system (Linux, macOS, etc).

### Corpus Workbench

Download the current stable version of [Corpus Workbench](https://cwb.sourceforge.io/). Install by following the
[*Installing the CWB Core*](https://cwb.sourceforge.io/install.php) instructions, either by using the provided packages
or building from source. Refer to the included `INSTALL` text file for further instructions.

CWB needs two directories for storing the corpora: one for the data, and one for the corpus registry.

## Installing the Korp backend

Begin by cloning the Korp backend repository. Use the `master` branch for the latest stable version, or use a specific
release tag. The default `dev` branch is intended for development and may be unstable.

For the latest stable branch:

```sh
git clone --branch master --single-branch https://github.com/spraakbanken/korp-backend.git
cd korp-backend
```

For a specific release, replace `v9.0.0` with the desired release tag:

```sh
git clone --branch v9.0.0 --single-branch https://github.com/spraakbanken/korp-backend.git
cd korp-backend
```

For setting up a virtual Python environment and installing the required Python modules, we recommend using
[uv](https://docs.astral.sh/uv/). uv can also be used to install a compatible version of Python if you don't have Python
3.11 or newer already.

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if you don't have it already.
2. While in the Korp backend directory, run:

   ```sh
   uv sync
   ```

   This will create a virtual environment in the `.venv` directory and install the dependencies.

### Optional dependencies

Korp provides the following optional dependency groups:

- `rate-limiting`: support for per-route rate limiting.
- `server`: [Gunicorn](https://gunicorn.org/) for running Korp in production.
- `jwt`: JWT support for authentication plugins.

If you want to install the optional dependencies, use the `--extra` option with `uv sync`:

```sh
uv sync --extra rate-limiting --extra server
```

To install all available extras, use:

```sh
uv sync --all-extras
```

With `pip`, extras use the equivalent syntax, for example `pip install '.[rate-limiting,server]'`.

An alternative to using `uv` is to set up a virtual environment manually using Python's built-in `venv` module and
install the dependencies using `pip`:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install --group dev  # Optional development tools and test dependencies
```

## Configuring Korp

Korp reads configuration from environment variables and from a `.env` file in the directory where the server is started.
To create a local configuration from the provided template, run:

```sh
cp .env.template .env
```

The template documents the commonly used settings and provides example values. Environment variables take precedence
over values in `.env`. Paths may contain `~`, which Korp expands to the user's home directory.

The following variables are required. The executable and registry paths should be absolute:

- `CQP_EXECUTABLE`
- `CWB_SCAN_EXECUTABLE`
- `CWB_REGISTRY`

Dependency relations, lexeme counts, token distribution, and other time-based statistics require MySQL or MariaDB.
Configure the database connection with:

- `DB_NAME`
- `DB_USER`
- `DB_PASSWORD`
- `DB_HOST` and `DB_PORT` if the database is not available at the defaults

Caching is enabled only when both settings below are configured, the Memcached server is reachable, and `CACHE_DIR`
already exists and is writable by the server process:

- `CACHE_DIR`
- `MEMCACHED_SERVER` (a `host:port` address; Unix sockets are not supported)

For browser-based cross-origin access (CORS), configure explicitly per deployment:

- `CORS_ALLOW_ORIGINS` (list of allowed origins; leave empty to disable cross-origin browser access)
- `CORS_ALLOW_ORIGIN_REGEX` (optional regex alternative for origin matching)
- `CORS_ALLOW_CREDENTIALS` (must be `false` if the origins list or regex allows every origin)
- `CORS_ALLOW_METHODS` and `CORS_ALLOW_HEADERS`

Error tracebacks are disabled by default. For local development they can be enabled with
`ERROR_TRACEBACKS_ENABLED=true`; an individual request must also set `debug=true` before a traceback is returned.

Optional route-level rate limiting can be enabled with:

- `RATE_LIMIT_ENABLED=true`
- `RATE_LIMIT_STORAGE_URI` (optional, defaults to `async+memcached://<MEMCACHED_SERVER>`)
- `RATE_LIMIT_HEADERS` (`none`, `on_reject`, or `always`; defaults to `on_reject`)
- `RATE_LIMIT_DEFAULT` (e.g. `10/minute` or `1/second;60/minute`; empty string means no limit)
- `RATE_LIMITS` (JSON object of per-route overrides, e.g. `{"concordance": "30/minute"}`; an empty value disables the
  limit for that route)

Rate limiting requires Korp to be installed with the `rate-limiting` extra dependencies, i.e. `uv sync --extra
rate-limiting` or `pip install '.[rate-limiting]'`.

Enable plugins by listing their importable module names in `PLUGINS`. Plugin-specific settings can be stored in a YAML
file referenced by `PLUGINS_CONFIG_FILE`:

```dotenv
PLUGINS=["plugins.example", "plugins.example_auth"]
PLUGINS_CONFIG_FILE="plugins.yaml"
```

Example `plugins.yaml`:

```yaml
plugins.example:
  greeting: "Hello!"

plugins.example_auth:
  protected_corpora: ["corpus1", "corpus2"]
  protection_details:
    corpus1:
      license: "restricted"
  required_header: "X-Authorized-Corpora"
```

At most one enabled plugin may provide an authorization class. If `PLUGINS_CONFIG` is also set in `.env`, its values
override matching top-level plugin entries from the YAML file.

## Running the backend

### Development

During development, you can use FastAPI's built-in development server, which will automatically reload the server when
you make changes to the code:

```sh
uv run fastapi dev korp/main.py
```

### Production

For deployment, use an ASGI server. Install the `server` extra, then run [Gunicorn](https://gunicorn.org/) with its
native ASGI worker:

```sh
uv run gunicorn korp.main:app --worker-class asgi --bind 0.0.0.0:8000 --workers 4
```

If a reverse proxy exposes the API below the domain root, set `ROOT_PATH` to that URL prefix. For example, use
`ROOT_PATH="/korp"` when the external API URL is `https://example.org/korp`. This ensures that links generated by
FastAPI, including the OpenAPI schema URL used by Swagger UI and ReDoc, include the prefix.

For improved performance with Gunicorn or Uvicorn, we recommend also installing the package `uvloop` (this is included
in the `server` extra dependencies if you use `uv sync --extra server`).

## Cache management

Most caching is done using Memcached, except for concordance query data, which is temporarily saved to disk to speed up
pagination. Memcached removes expired entries itself, but Korp must still invalidate cached data when corpora or corpus
configuration files change. A bodyless `POST /admin/cache/refresh` performs that invalidation and removes expired disk
cache files. We recommend calling it regularly from a scheduled task.

The backend does not itself authenticate this administrative route. You may want to restrict it at the reverse proxy or
network boundary so that it is available only to trusted callers.

## API documentation

The API documentation is available at `/docs` (Swagger UI) and `/redoc` (ReDoc) while the server is running. The
generated OpenAPI 3.1 document at `/openapi.json` is the authoritative machine-readable API contract.

## Adding corpora

Korp works as a layer on top of Corpus Workbench for most corpus search functionality. See the [CWB corpus encoding
tutorial](http://cwb.sourceforge.net/files/CWB_Encoding_Tutorial.pdf) for information regarding encoding corpora. Note
that Korp requires your corpora to be encoded in UTF-8. Values of structural CWB attributes may not contain tab
characters. Once CWB is aware of your corpora they will be accessible through the Korp API.

### Adding additional info about the corpus

For Korp to show the number of sentences and the date when a corpus was last updated, you have to manually add this
information. Create a file called `.info` in the directory of the CWB data files for the corpus, and add to it the
following lines (editing the values to match your material). Be sure to end the file with a blank line:

```text
Sentences: 12345
Updated: 2019-11-30
FirstDate: 2001-01-16 00:00:00
LastDate: 2001-01-30 23:59:59
```

Once this file is in place, Korp will be able to access this information.

### Corpus structure requirements

To use the basic concordance features of Korp there are no particular requirements regarding the markup or annotations
of your corpora.

To use the dependency-relation routes, your corpus must adhere to the following format:

- The structural annotation marking sentences must be named `sentence`.
- Every sentence annotation must have an attribute named `id` with a value that is unique within the corpus.

To use time-based statistics, your corpus needs to be annotated with date information using the following four
structural CWB attributes: `text_datefrom`, `text_timefrom`, `text_dateto`, and `text_timeto`. The date format should be
*YYYYMMDD*, and the time format *hhmmss*. A corpus dated 2006 would have the following values:

- `text_datefrom:  20060101`
- `text_timefrom:  000000`
- `text_dateto:    20061231`
- `text_timeto:    235959`

## Database tables

This section describes the database tables needed by the dependency-relation, lexeme-count, token-distribution, and
time-based frequency routes. If you don't need any of these features, you can skip this section.

### Database tables for dependency relations

The dependency relation data consists of head-relation-dependent triplets and frequencies. For every corpus, you need
six database tables. The prefix of the table names (`relations` by default) can be configured using the
`DB_DEPENDENCY_RELATIONS_TABLE_PREFIX` variable. The table structures are as follows (the `CORPUSNAME` part of the table
names should be replaced with the actual corpus name, in uppercase):

```text
Table name: relations_CORPUSNAME  
Charset:    UTF-8  

Columns:  
    id             int                  A unique ID (within this table)  
    head           int                  Reference to an ID in the strings table (below). The head word in the relation  
    rel            enum(...)            The syntactic relation  
    dep            int                  Reference to an ID in the strings table (below). The dependent in the relation  
    freq           int                  Frequency of the triplet (head, rel, dep)  
    bfhead         bool                 True if head is a base form (or lexeme)  
    bfdep          bool                 True if dep  is a base form (or lexeme)  
    wfhead         bool                 True if head is a word form  
    wfdep          bool                 True if dep is a word form  

Indexes:  
    (head, wfhead, dep, rel, freq, id)  
    (dep, wfdep, head, rel, freq, id)  
    (head, dep, bfhead, bfdep, rel, freq, id)  
    (dep, head, bfhead, bfdep, rel, freq, id)


Table name: relations_CORPUSNAME_strings  
Charset:    UTF-8  

Columns:  
    id             int                  A unique ID (within this table)  
    string         varchar(100)         The head or dependent string  
    stringextra    varchar(32)          Optional preposition for the dependent  
    pos            varchar(5)           Part-of-speech for the head or dependent  

Indexes:  
    (string, id, pos, stringextra)  
    (id, string, pos, stringextra)


Table name: relations_CORPUSNAME_rel  
Charset:    UTF-8  

Columns:  
    rel            enum(...)            The syntactic relation  
    freq           int                  Frequency of the relation  

Indexes:  
    (rel, freq)  


Table name: relations_CORPUSNAME_head_rel  
Charset:    UTF-8  

Columns:  
    head           int                  Reference to an ID in the strings table. The head word in the relation  
    rel            enum(...)            The syntactic relation  
    freq           int                  Frequency of the pair (head, rel)  

Indexes:  
    (head, rel, freq)


Table name: relations_CORPUSNAME_dep_rel  
Charset:    UTF-8  

Columns:  
    dep            int                  Reference to an ID in the strings table. The dependent in the relation  
    rel            enum(...)            The syntactic relation  
    freq           int                  Frequency of the pair (rel, dep)  

Indexes:  
    (dep, rel, freq)


Table name: relations_CORPUSNAME_sentences  
Charset:    UTF-8  

Columns:  
    id             int                  An ID from relations_CORPUSNAME
    sentence       varchar(64)          A sentence ID (see the section about corpus structure above)  
    start          int                  The position of the first word of the relation in the sentence  
    end            int                  The position of the last word of the relation in the sentence  

Indexes:  
    id
```

In the main `relations_CORPUSNAME` table, each relation should be represented three times. Once with both dependent and
head as base forms, once with dependent as base form and head as word form, and once with dependent as word form and
head as base form. This is to allow searching for both base forms and word forms, giving different results for different
searched word forms, while the results are always displayed as base forms. If the base form annotation is missing for a
dependent, head or both, the word form can be used as both word form and base form by setting both bfhead/bfdep and
wfhead/wfdep to True. In such a case you won't need all three rows for that relation.

The `sentences` table contains sentence IDs for sentences containing the relations, with start and end values to point
out exactly where in the sentences the relations occur (1 being the first word of the sentence).

### Lexeme counts

The lexeme counts table contains the number of occurrences of each lexeme in each corpus. This is used by the frontend
to grey out auto-completion suggestions which would not give any results in the selected corpora. The lexeme counts data
consists of a single MySQL table, with the following layout:

```text
Table name: lexeme_counts  
Charset:    UTF-8  

Columns:  
    lexeme       varchar(64)         The lexeme  
    freq         int                 Number of occurrences  
    corpus       varchar(64)         The corpus name  

Indexes:  
    (lexeme, corpus, freq)
```

### Time data

For token distributions and other time-based statistics, add token-per-time-span data to your database. For tokens
without date or time information, use the date 0000-00-00 00:00:00. Use the following table layout:

```text
Table name: timedata  
Charset:    UTF-8  

Columns:  
    corpus    varchar(64)        The corpus name
    datefrom  datetime           Full from-date and time
    dateto    datetime           Full to-date and time
    tokens    int                Number of tokens between from-date and (including) to-date

Indexes:  
    (corpus, datefrom, dateto)


Table name: timedata_date  
Charset:    UTF-8  

Columns:  
    corpus    varchar(64)        The corpus name
    datefrom  date               From-date (only date part)
    dateto    date               To-date (only date part)
    tokens    int                Number of tokens between from-date and (including) to-date

Indexes:  
    (corpus, datefrom, dateto)
```

## Corpus Configuration for the Korp Frontend

The corpus configuration used by the Korp frontend is served by the backend. The config variable `CORPUS_CONFIG_DIR`
should point to a directory having the following structure:

```text
.
├── attributes
│   ├── positional
│   │   ├── lemma.yaml
│   │   ├── msd.yaml
│   │   ├── ...
│   │   └── pos.yaml
│   └── structural
│       ├── author.yaml
│       ├── title.yaml
│       ├── ...
│       └── year.yaml
├── corpora
│   ├── corpus1.yaml
│   ├── corpus2.yaml
│   ├── ...
│   └── yet-another-corpus.yaml
└── modes
    ├── default.yaml
    ├── another.yaml
    ├── ...
    └── other.yaml
```

- The **modes** directory contains one YAML file per mode in Korp.
- The **corpora** directory contains one YAML file per corpus.
- The **attributes** directory contains two subdirectories: **positional** and **structural**, containing optional
  annotation presets referred to by the corpus configurations.

For some inspiration, you can look at the [config files](https://github.com/spraakbanken/korp-config) used by the Korp
instance at Språkbanken Text.

**Note:**  
Most settings in these files referring to labels or descriptions can optionally be localized using ISO 639-3 language
codes. For example, a label can look both like this:

```yaml
label: author
```

... and like this:

```yaml
label:
  eng: author
  swe: författare
```

### Mode Configuration

At least one mode file is required, and that file must be named `default.yaml`. This is the mode that will be loaded
when no mode is explicitly requested.

**Required:**

- **label**: The name of the mode, which will be shown in the interface.

**Optional:**

- **description**: A description of the mode, shown when first entering it. May include HTML.
- **order**: A number used for sorting the modes in the interface. Modes without an order will end up last.
- **folders**: A folder structure for the corpus selector. These folders can then be referenced by individual corpora.
  The folder structure can be of any depth, and folders can have any number of sub-folders (using the key `subfolders`).
  You may use HTML in the descriptions. Example:

  ```yaml
  folders:
    novels:
      title:
        eng: Novels
        swe: Skönlitteratur
      description:
        eng: Corpora consisting of novels.
        swe: Korpusar bestående av skönlitteratur.
      subfolders:
        classics:
          title:
            eng: Classics
            swe: Klassiker
        scifi:
          title: Science-Fiction
  ```

- **preselected_corpora**: A list of corpus IDs which will be pre-selected when the user enters the mode. You may also
  refer to folders by using the prefix `__`, and dot notation for referring to subfolders. Example:

  ```yaml
  preselected_corpora:
    - my-corpus
    - __novels.scifi
  ```

- Other than the above, you can also override almost all the global settings set in the frontend's `config.yaml`. See
  [the documentation for the
  frontend](https://github.com/spraakbanken/korp-frontend/blob/master/doc/frontend_devel.md#settings-in-configyml) for a
  list of available settings.

### Corpus Configuration

Corpus configuration files are placed in the `corpora` folder, and the filename of each configuration file should
correspond to a corpus ID in lowercase, followed by `.yaml`, e.g. `mycorpus.yaml`.

**Required:**

- **id**: The corpus' system name, same as the configuration file name (minus `.yaml`).
- **title**: Title of the corpus.
- **description**: Description of the corpus. HTML can be used.
- **mode**: A list of the modes in which the corpus will be included, optionally specifying a folder. Example:

  ```yaml
  mode:
    - name: default
      folder: novels.classics
  ```

**Optional:**

- **within**: Use this to override **default_within** (set in the global or mode config). **within** is a list of
  structural elements to use as boundaries when searching, ordered from smaller to bigger. Example:

  ```yaml
  within:
    - label:
        eng: sentence
        swe: mening
      value: sentence
    - label:
        eng: paragraph
        swe: stycke
      value: paragraph
  ```

- **context**: Use this to override **default_context** (set in the global or mode config). **context** is a list of
  structural elements that can be used as context in the displaying of the search results, ordered from smaller to
  bigger. Example:

  ```yaml
  context:
    - label:
        eng: 1 sentence
        swe: 1 mening
      value: 1 sentence
    - label:
        eng: 1 paragraph
        swe: 1 stycke
      value: 1 paragraph
  ```

- **attribute_filters**: A list of structural annotations (CWB structural attributes) on which the user will be able to
  filter the search results, using menus in both simple and extended search.
- **pos_attributes** and **struct_attributes**: Lists of positional and structural annotation definitions. Every item in
  each list should be an object with one key. The key should be the CWB attribute name, e.g. `msd` for a positional
  annotation or `text_title` for a structural annotation. The value should be either 1) an object with a complete
  annotation definition, or 2) a string referring to an annotation preset containing such a definition, e.g. `msd` to
  refer to `attributes/positional/msd.yaml`. With option 1, you may also refer to a preset by using the key `preset` and
  then extend/override that preset. The annotation definition tells the Korp frontend how to handle the annotation, such
  as how it should be presented in the sidebar and what interface widget to use in extended search. For more information
  about the available annotation-definition options, see the [Korp frontend
  documentation](https://github.com/spraakbanken/korp-frontend/blob/master/doc/frontend_devel.md#attribute-settings).
  Example:

  ```yaml
  struct_attributes:
    - text_title: title
    - text_type:
        label:
          eng: type
          swe: typ
    - text_source:
        preset: url
        label:
          eng: source
          swe: källa
  ```

- **custom_attributes**: See [Custom
  attributes](https://github.com/spraakbanken/korp-frontend/blob/master/doc/frontend_devel.md#custom-attributes).
- **reading_mode**: See [Reading
  mode](https://github.com/spraakbanken/korp-frontend/blob/master/doc/frontend_devel.md#reading-mode).

### Annotation presets

See **pos_attributes** and **struct_attributes** above.
