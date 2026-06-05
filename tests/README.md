# Tests for the Korp backend

This directory `tests` contains [Pytest](https://pytest.org) tests for the Korp backend.

## Prerequisites

To be able to run tests, you need the development dependencies installed. If you have installed the Korp backend with
`uv`, the development dependencies are installed by default. If installed with `pip`, you can install the development
dependencies by running

```sh
python3 -m pip install -e ".[dev]"
```

In addition, you need to have the Corpus Workbench (CWB), in particular `cwb-encode` and `cwb-makeall`, installed and on
`PATH` (see the [main README file](../README.md#corpus-workbench)).

For database tests, you also need to have a MySQL/MariaDB server with a user with the privilege to create databases.

## Running tests

To run tests, run

```sh
pytest
```

### Database access

To run successfully tests that require Korp MySQL database data, you may need to specify custom command-line options to
`pytest`:

- `--db-host=`_HOST_: Use host _HOST_ for the Korp MySQL test database
- `--db-port=`_PORT_: Use port _PORT_ for the Korp MySQL test database
- `--db-name=`_NAME_: Use database name _NAME_ for the Korp MySQL test database
- `--db-user=`_USER_: Use user _USER_ for the Korp MySQL test database
- `--db-password=`_PASSWORD_: Use password _PASSWORD_ for the Korp MySQL test database

If these are not specified explicitly, tests try to use the values specified in the Korp configuration for `DB_HOST`,
`DB_PORT`, `DB_USER` and `DB_PASSWORD`. That fails unless the user specified there has the privilege to create a
database or you specify with `--db-name` the name of an existing database in which the user has the table creation
privilege.

The database user should also have the file privilege to load data from files.

In addition, you can specify the database collation with a custom `pytest` command-line option:

- `--db-collate=`_COLLATE_: Use _COLLATE_ as the Korp MySQL test database collation. If not specified, use the default
  collation for the database character set specified in the Korp configuration variable `DB_CHARSET`.

If the test database cannot be created, a warning is issued and tests using the database (fixture `database`) are
skipped.

### Test coverage

To find out test coverage using [Coverage.py](https://coverage.readthedocs.io/), run

```sh
coverage -m pytest
```

and then, for example,

```sh
coverage report
```

## Directory Layout

This directory `tests/` contains:

- [`unit/`](unit): unit tests, typically testing functions in modules directly under the `korp` package
- [`functional/`](functional): functional tests, typically testing the endpoints (`korp.api.routers.*`)
- [`testing/`](testing): unit tests for functionality in test utility modules (`tests.*utils`)
- `data/`: test data
  - [`data/corpora/src`](data/corpora/src): corpus source data
  - [`data/corpora/config`](data/corpora/config): corpus configuration data
  - `data/corpora/cwb-cache`: cached CWB corpus data encoded from corpus source data (created by the tests)
  - [`data/db`](data/db): Korp MySQL database data
  - [`data/db/tableinfo`](data/db/tableinfo): YAML files with information for creating Korp MySQL database tables
- [`conftest.py`](conftest.py): Pytest configuration; in particular, fixtures to be used by individual tests
- [`configutils.py`](configutils.py): utility functions for processing the Korp configuration
- [`corpusutils.py`](corpusutils.py): utility functions for setting up CWB corpus data
- [`dbutils.py`](dbutils.py): `KorpDatabase` class for setting up and using Korp MySQL test database
- [`testutils.py`](testutils.py): utility functions for tests, typically functionality that recur in multiple tests but
  that cannot be made fixtures

## Adding tests

Individual test files and tests should follow Pytest conventions: the names of files containing tests should begin with
`test_`, as should also the names of test functions and methods. Tests can be grouped in classes whose names begin with
`Test`.

### Fixtures

The following Pytest fixtures have been defined in [`conftest.py`](conftest.py):

- `corpus_data_root`: Return CWB corpus root directory for a session
- `corpus_registry_dir`: Return CWB corpus registry directory for a session
- `cache_dir`: Return Korp cache directory
- `corpus_config_dir`: Return corpus configuration directory
- `corpus_configs`: Copy corpus configurations in `data/corpora/config` to a temporary directory used in tests
- `corpora`: Encode the corpora in `data/corpora/src` and return their ids
- `database`: Return a `KorpDatabase` object for a session
- `database_tables`: Import database data for the specified corpora and table types
- `app_factory`: Return a function to create and configure a Korp FastAPI app instance. The returned function optionally
  takes as its argument a `dict` for overriding default Korp configuration values
- `client`: Return a default `TestClient` created with `with TestClient(app)`
- `client_factory`: Return a context manager function for creating a `TestClient` with optional config overrides
- `get_json`: Return a function for sending a request with `client` and getting the response JSON data

### Functional tests

A typical functional test testing an endpoint uses the `get_json` and `corpora` fixtures. For example:

```python
def test_corpus_info_single_corpus(get_json, corpora):
    corpus = corpora[0].upper()
    response = get_json(
        "/corpora/info",
        params={
            "cache": "false",
            "corpus": corpus,
        })
    corpus_data = response["corpora"][corpus]
    attrs = corpus_data["attrs"]
    assert attrs
```

If the endpoint uses the Korp MySQL database, it should also use the `database_tables` fixture and load the appropriate
database table data. For example:

```python
def test_lexeme_counts_single_corpus(get_json, database_tables):
    """Test `/lexeme_counts` on a single corpus."""
    database_tables("testcorpus1", "lexeme_counts")
    lexeme = "test..nn.1"
    response = get_json(
        "/lexeme_counts",
        params={
            "corpus": "testcorpus1",
            "lexeme": lexeme,
            "cache": "false",
        })
    assert lexeme in response
```

### Corpus data

Each CWB corpus _corpus_ whose data is used in the tests should have a source file in `data/corpora/src`. Two different
corpus source formats are supported:

1. A slightly extended VRT (VeRticalized Text) format (the input format for CWB), in which structural attributes are
   marked with XML-style tags (with annotations as element attributes) and each token is on its own line, with
   positional (token) attributes separated by tabs. In VRT, the XML-style tags may _not_ be indented.

   Since the standard VRT content does not specify the names of positional attributes, the format has been extended so
   that their names can be specified in a special XML comment at the top of the file. A similar comment can also be used
   to specify the structural attributes, even though structural attributes can also be inferred from the file content.
   See below for more details and alternatives. For example:

   ```xml
   <!-- #vrt positional-attributes: word lemma -->
   <!-- #vrt structural-attributes: text:0+id paragraph:0+id sentence:0+id -->
   <text id="t1">
   <paragraph id="p1">
   <sentence id="s1" a="x">
   This	this
   is	be
   a	a
   test	test
   .	.
   </sentence>
   <sentence id="s2">
   Great	great
   !	!
   </sentence>
   </paragraph>
   </text>
   ```

2. An XML format of the kind of the XML export formats produced by [Sparv](https://spraakbanken.gu.se/sparv/):
   structural attributes are represented by XML elements like in VRT (but tags can be indented) and tokens by leaf-level
   `token` elements with the word form as the text content and token attributes as element attributes. For example, the
   following XML corresponds to the above VRT:

   ```xml
   <?xml version='1.0' encoding='UTF-8'?>
   <text id="t1">
     <paragraph id="p1">
       <sentence id="s1">
         <token lemma="this">This</token>
         <token lemma="be">is</token>
         <token lemma="a">a</token>
         <token lemma="test">test</token>
         <token lemma=".">.</token>
       </sentence>
       <sentence id="s2" a="x">
         <token lemma="great">Great</token>
         <token lemma="!">!</token>
       </sentence>
     </paragraph>
   </text>
   ```

   Possible elements above `text` are also included in the data.

For VRT source files, the positional and structural attributes can be specified in the following three ways. For XML
files, both the positional and structural attribute names can be inferred from the data as the positional attributes are
named attributes of `token` elements. However, the first approach can also be used for XML files to override the
inferred attributes.

1. In a YAML file _corpus_`.attrs.yaml` with content like the following (for the above examples):

   ```yaml
   pos_attributes:
   - word
   - lemma
   struct_attributes:
   - text:
     - id
   - sentence:
     - id
     - x
   ```

   In addition, if a structural attribute can be recursively nested, its name should be followed by the recursive
   nesting depth, separated by a space or colon:

   ```yaml
   struct_attributes:
   - div 2:
     - a5
     # …
   - np:3: []
   ```

   If a structural attribute has no annotations, the annotations should be specified as an empty list.

   If _corpus_`.attrs.yaml` lacks `pos_attributes` or `struct_attributes` information, the missing information is
   obtained with approach 2 if applicable, otherwise with approach 3.

2. If _corpus_`.attrs.yaml` does not exist, the attributes can be specified at the top of the VRT file as XML comments
   (an extension to the VRT format):

   ```xml
   <!-- #vrt positional-attributes: attr1 attr2 ... -->
   <!-- #vrt structural-attributes: text:0+a1+a2 sentence:0+a3+a4 ... -->
   ```

   Structural attributes are specified in the same way as for the `cwb-encode` tool. See the VRT file example above for
   a concrete example.

3. If _corpus_`.attrs.yaml` does not exist and the VRT file does not have a `positional-attributes` comment, positional
   attribute names are first taken from the following list: `word lemma pos msd deprel dephead ref lex/`, as many names
   as the first token line has tab-separated attributes. If the token line has more attributes, the rest are named as
   `attr`_n_, where _n_ is the number of the attribute.

   If the VRT file has no `structural-attributes` comment, the structural attributes and their annotations are inferred
   based on the content of the VRT file.

In approaches 1 and 2, a trailing slash in the name of a positional attribute or structural attribute annotation is
passed to `cwb-encode` to indicate that its values are to be validated and normalized as feature sets (multi-valued).
Approach 3 infers that a positional attribute or structural attribute annotation is feature-set-valued if all its values
begin and end with a vertical bar `|`. It is also inferred similarly from XML data.

In addition to the VRT file _corpus_`.vrt`, a corpus should have a corresponding info file _corpus_`.info` containing at
least the number of sentences and date of update in the ISO format as follows:

```text
Sentences: 2
Updated: 2023-01-20
```

Note that the encoded test corpus data is placed under a temporary directory for the duration of a test session, so test
corpora are isolated from any other CWB corpora in the system. Encoded test corpus data is cached under
`tests/data/corpora/cwb-cache` between test sessions, to avoid re-encoding it in each session.

### Corpus configuration data

Corpus configuration data used in tests for the `/corpora/config` endpoint is under `data/corpora/config` in the format
expected by Korp; please see [the documentation](../README.md#corpus-configuration-for-the-korp-frontend) for more
information.

### Database data

Test database data resides in files under the subdirectory `data/db/` and its subdirectories. It can be specified in two
formats:

1. SQL files (extension `.sql`) containing all the necessary table creation and value insertion statements. A single SQL
   file may contain data for multiple tables.
2. TSV (tab-separated values) files (extension `.tsv`), each containing values for a single table. The table is inferred
   based on the file name and _table information files_ in YAML format that also specify the table definition (see
   below).

TSV files should not have a header row: columns in the file must be in the order they are in the table definition.
Backslash escapes are not recognized, so values cannot contain tab or newline characters.

Each file, whether SQL or TSV, should contain data only for one corpus, whose id should be a part of the file (or
directory) name.

Data can be imported by specifying either the corpus id and type(s) of table(s) (one or more of `timedata`,
`lexeme_counts` and `dependency_relations`) or the files containing data (globs can be used).

The YAML files in [`data/db/tableinfo/`](data/db/tableinfo) contain table information specifying a mapping from (TSV)
data files to database tables (and indirectly also the other way round). Each file contains a sequence of one or more
mappings with the following keys recognized:

- `tablename`: The name of the table. The name may contain the format specification `{corpus}` or `{CORPUS}`, referring
  to the corpus id replacing the placeholder `{corpus}` in the regular expression in `filenames` (see below), in lower
  or upper case, respectively.
- `filenames`: A sequence of file name regular expressions. If a full file name matches one of the expressions and none
  of those in `exclude_filenames`, load the data from it to the table specified in `tablename`. Each regular expression
  should contain the placeholder `{corpus}` to be replaced with a corpus id for the fixture `database_tables` to be able
  to find the database data for a corpus.
- `exclude_filenames`: A sequence of excluded file name regular expressions.
- `definition`: A string containing the MySQL table definition: columns and possible keys.

If a file name would match regular expression in multiple mappings, the first mapping found is used. Regular expressions
are matched to absolute file names in their entirety, including the directory. If the regular expressions in `filenames`
and `exclude_filenames` do not begin with `.*/`, it is prefixed to the expression. The regular expressions should not
include the extension `.tsv` (or `.sql`).

The value of `definition` may contain variable references as `{`_var_`}`. Their values must be defined before use in a
separate sequence item with key `definition_vars` and value that is a mapping from variable names to values:

```yaml
- definition_vars:
    var1: value1
    var2: value2
```

Currently, the table information files support the following file name and directory naming schemes (under
`tests/data/db/`) for the various types of tables:

- _tabletype_`/`_corpus_`.`_ext_
- _tabletype_`/`_corpus_`[_:+]`_tabletype\_detailed_`.`_ext_
- _corpus_`/`_tabletype\_detailed_`.`_ext_

Here:

- _corpus_ = corpus id (in lower case)
- _tabletype_ = high-level table type: one of `lexeme_counts` (or `lexemes`), `timedata` and `dependency_relations`
- _tabletype\_detailed_ = more detailed table type (mainly for TSV files):
  - `lexeme_counts`: `lexeme_counts` (or `lexemes`) (the same as the high-level type)
  - `timedata`: `timedata` or `timedata_date`
  - `dependency_relations`: `relations`, `relations_strings`, `relations_rel`, `relations_head_rel`,
     `relations_dep_rel` or `relations_sentences`
- _ext_ = file type extension: `tsv` or `sql`
