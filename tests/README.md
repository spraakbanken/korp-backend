
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


## Running tests

To run tests, run
```
$ pytest
```

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
- [`conftest.py`](conftest.py): Pytest configuration; in particular,
  fixtures to be used by individual tests
- [`corpusutils.py`](corpusutils.py): utility functions for setting up
  CWB corpus data
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
- `app`: Create and configure a Korp Flask app instance
- `client`: Create and return a test client


### Functional tests

A typical functional test testing an endpoint uses the `client` and
`corpora` fixtures. For example:

```python
def test_corpus_info_single_corpus(self, client, corpora):
    corpus = corpora[0].upper()
    response = client.get(
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

_[To be added when database testing is implemented.]_
