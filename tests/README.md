
# Tests for the Korp backend

This directory `tests` contains [Pytest](https://pytest.org) tests for
the Korp backend.


## Prerequisites

To be able to run tests, you need to install the development
requirements by running
```
$ pip3 install -r requirements-dev.txt
```

In addition, you need to have the Corpus Workbench (CWB), in
particular `cwb-encode`, and the CWB Perl tools (for `cwb-make`),
installed and on `PATH` (see the [main README
file](../README.md#corpus-workbench)). The CWB Perl tools can be
installed from the CWB Subversion repository at
http://svn.code.sf.net/p/cwb/code/perl/trunk

For database tests, you also need to have a MySQL/MariaDB server with
a user with the privileges to create a database and grant access to
it.


## Running tests

To run tests, run
```
$ pytest
```


### Database access

To run successfully tests that require Korp MySQL database data, you
may need to specify custom command-line options to `pytest`:

- `--db-host=`_HOST_: Use host _HOST_ for the Korp MySQL test database
- `--db-port=`_PORT_: Use port _PORT_ for the Korp MySQL test database
- `--db-name=`_NAME_: Use database name _NAME_ for the Korp MySQL test
  database
- `--db-user=`_USER_: Use user _USER_ to access the Korp MySQL test
  database
- `--db-password=`_PASSWORD_: Use password _PASSWORD_ to access the
  Korp MySQL test database
- `--db-create-user=`_USER_: Use user _USER_ to create the Korp MySQL
  test database
- `--db-create-password=`_PASSWORD_: Use password _PASSWORD_ to create
  the Korp MySQL test database

If these are not specified explicitly, tests try to use the values
specified in the Korp configuration for `DBHOST`, `DBPORT`, `DBUSER`
and `DBPASSWORD`. That fails unless the user specified there has the
privilege to create a database or you specify with `--db-name` the
name of an existing database in which the user has the table creation
privilege.

The database user should also have the file privilege to load data
from files.

If the test database cannot be created, tests using the database
(fixture `database`) are skipped.


### Test coverage

To find out test coverage using
[Coverage.py](https://coverage.readthedocs.io/), run
```
$ coverage -m pytest
```
and then, for example,
```
$ coverage report
```


## Directory Layout

This directory `tests/` contains:

- [`unit/`](unit): unit tests, typically testing functions in modules
  directly under the `korp` package
- [`functional/`](functional): functional tests, typically testing the endpoints
  (`korp.views.*`)
- `data/`: test data
  - [`data/corpora/src`](data/corpora/src): corpus source data
  - [`data/corpora/config`](data/corpora/config): corpus configuration
    data
  - [`data/db`](data/db): Korp MySQL database data
  - [`data/db/tableinfo`](data/db/tableinfo): YAML files with
    information for creating Korp MySQL database tables
- [`conftest.py`](conftest.py): Pytest configuration; in particular,
  fixtures to be used by individual tests
- [`configutils.py`](configutils.py): utility functions for processing
  the Korp configuration
- [`corpusutils.py`](corpusutils.py): utility functions for setting up
  CWB corpus data
- [`dbutils.py`](dbutils.py): `KorpDatabase` class for setting up and
  using Korp MySQL test database
- [`testutils.py`](testutils.py): utility functions for tests, typically
  functionality that recur in multiple tests but that cannot be made fixtures


## Adding tests

Individual test files and tests should follow Pytest conventions: the
names of files containing tests should begin with `test_`, as should
also the names of test functions and methods. Tests can be grouped in
classes whose names begin with `Test`.


### Fixtures

The following Pytest fixtures have been defined in
[`conftest.py`](conftest.py):

- `corpus_data_root`: Return CWB corpus root directory for a session
- `corpus_registry_dir`: Return CWB corpus registry directory for a session
- `cache_dir`: Return Korp cache directory
- `corpus_config_dir`: Return corpus configuration directory
- `corpus_configs`: Copy corpus configurations in
  `data/corpora/config` to a temporary directory used in tests
- `corpora`: Encode the corpora in `data/corpora/src` and return their ids
- `database`: Return a `KorpDatabase` object for a session
- `database_tables`: Import database data for the specified corpora
  and table types
- `app`: Return a function to create and configure a Korp Flask app
  instance. The returned function optionally takes as its argument a
  `dict` for overriding default Korp configuration values.
- `client`: Return a function to create and return a test client. The
  returned function optionally takes as its argument a `dict` for
  overriding default Korp configuration values.


### Functional tests

A typical functional test testing an endpoint uses the `client` and
`corpora` fixtures. For example:

```python
def test_corpus_info_single_corpus(self, client, corpora):
    corpus = corpora[0].upper()
    response = client().get(
        "/corpus_info",
        query_string={
            "cache": "false",
            "corpus": corpus,
        })
    assert response.status_code == 200
    assert response.is_json == True
    data = response.get_json()
    corpus_data = data["corpora"][corpus]
    attrs = corpus_data["attrs"]
    assert attrs
```

If the endpoint uses the Korp MySQL database, it should also use the
`database` fixture and load the appropriate database table data with
`database.import_tables()`. For example:

```python
def test_lemgram_count_single_corpus(self, client, database):
    """Test /lemgram_count on a single corpus."""
    database.import_tables(["lemgram_index/*.tsv"])
    lemgram = "test..nn.1"
    response = client().get(
        "/lemgram_index",
        query_string={
            "corpus": "testcorpus1",
            "lemgram": lemgram,
            "cache": "false",
        })
    data = response.get_json()
    assert lemgram in data
```


### Corpus data

Each CWB corpus _corpus_ whose data is used in the tests should have a
source VRT file _corpus_`.vrt` in `data/corpora/src`. The corpus
source files use a slightly extended VRT (VeRticalized Text) format
(the input format for CWB), where structures are marked with XML-style
tags (with attributes) and each token is on its own line, token
attributes separated by tags.

The extension is that the positional and structural attributes need to
be declared at the top of the file as XML comments as follows:
```
<!-- #vrt positional-attributes: attr1 attr2 ... -->
<!-- #vrt structural-attributes: text:0+a1+a2 sentence:0+a3+a4 ... -->
```
For example:
```
<!-- #vrt positional-attributes: word lemma -->
<!-- #vrt structural-attributes: text:0+id paragraph:0+id sentence:0+id -->
<text id="t1">
<paragraph id="p1">
<sentence id="s1">
</sentence>
This	this
is	be
a	a
test	test
.	.
<sentence id="s2">
Great	great
!	!
</sentence>
</paragraph>
</text>
```

In addition to the VRT file _corpus_`.vrt`, a corpus should have a
corresponding info file _corpus_`.info` containing at least the number
of sentences and date of update in the ISO format as follows:
```
Sentences: 2
Updated: 2023-01-20
```

Note that the encoded test corpus data is placed under a temporary
directory for the duration of a test session, so test corpora are
isolated from any other CWB corpora in the system.


### Corpus configuration data

Corpus configuration data used in tests for the `/corpus_config`
endpoint is under `data/corpora/config` in the format expected by
Korp; please see [the
documentation](../README.md#corpus-configuration-for-the-korp-frontend)
for more information.


### Database data

Test database data resides in files under the subdirectory `data/db/`
and its subdirectories. It can be specified in two formats:

1. SQL files (extension `.sql`) containing all the necessary table
   creation and value insertion statements. A single SQL file may
   contain data for multiple tables.
2. TSV (tab-separated values) files (extension `.tsv`), each
   containing values for a single table. The table is inferred based
   on the file name and _table information files_ in YAML format that
   also specify the table definition (see below).

TSV files should not have a header row: columns in the file must be in
the order they are in the table definition. Backslash escapes are not
recognized, so values cannot contain tab or newline characters.

Each file, whether SQL or TSV, should contain data only for one
corpus, whose id should be a part of the file (or directory) name.

Data can be imported by specifying either the corpus id and type(s) of
table(s) (one or more of `timedata`, `lemgram_index` and `relations`)
or the files containing data (globs can be used).

The YAML files in [`data/db/tableinfo/`](data/db/tableinfo) contain
table information specifying a mapping from (TSV) data files to
database tables (and indirectly also the other way round). Each file
contains a sequence of one or more mappings with the following keys
recognized:

- `tablename`: The name of the table. The name may contain the format
  specification `{corpus}` or `{CORPUS}`, referring to the corpus id
  replacing the placeholder `{corpus}` in the regular expression in
  `filenames` (see below), in lower or upper case, respectively.
- `filenames`: A sequence of file name regular expressions. If a full
  file name matches one of the expressions and none of those in
  `exclude_filenames`, load the data from it to the table specified in
  `tablename`. Each regular expression should contain the placeholder
  `{corpus}` to be replaced with a corpus id for the fixture
  `database_tables` to be able to find the database data for a corpus.
- `exclude_filenames`: A sequence of excluded file name regular expressions.
- `definition`: A string containing the MySQL table definition:
  columns and possible keys.

If a file name would match regular expression in multiple mappings,
the first mapping found is used. Regular expressions are matched to
absolute file names in their entirety, including the directory. If the
regular expressions in `filenames` and `exclude_filenames` do not
begin with `.*/`, it is prefixed to the expression. The regular
expressions should not include the extension `.tsv` (or `.sql`).

The value of `definition` may contain variable references as
`{`_var_`}`. Their values must be defined before use in a separate
sequence item with key `definition_vars` and value that is a mapping
from variable names to values:

```yaml
- definition_vars:
    var1: value1
    var2: value2
```

Currently, the table information files support the following file name
and directory naming schemes (under `tests/data/db/`) for the various
types of tables:

- _tabletype_`/`_corpus_`.`_ext_
- _tabletype_`/`_corpus_`[_:+]`_tabletype\_detailed_`.`_ext_
- _corpus_`/`_tabletype\_detailed_`.`_ext_

Here:

- _corpus_ = corpus id (in lower case)
- _tabletype_ = high-level table type: one of `lemgram_index` (or
  `lemgrams`), `timedata` and `relations`
- _tabletype\_detailed_ = more detailed table type (mainly for TSV files):
   - `lemgram_index`: `lemgram_index` (or `lemgrams`) (the same as the
     high-level type)
   - `timedata`: `timedata` or `timedata_date`
   - `relations`: `relations`, `relations_strings`, `relations_rel`,
     `relations_head_rel`, `relations_dep_rel` or `relations_sentence`
- _ext_ = file type extension: `tsv` or `sql`
