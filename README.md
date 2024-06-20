# Korp Backend

This is the backend for [Korp](https://spraakbanken.gu.se/korp), a corpus search tool developed by
[Språkbanken](https://spraakbanken.gu.se) at the University of Gothenburg, Sweden.

The code is distributed under the [MIT license](https://opensource.org/licenses/MIT).

The Korp backend is a Python 3 WSGI application, acting as a wrapper for Corpus Workbench.

To see what has changed in recent versions, see the [CHANGELOG](CHANGELOG.md).

## Requirements

To use the basic features of the Korp backend you need the following:

* [Python 3.6+](https://python.org/)
* [Corpus Workbench](https://cwb.sourceforge.io/) (CWB) 3.4.12 or newer

To use the additional features such as the Word Picture you also need:

* [MariaDB](https://mariadb.org/) or [MySQL](https://www.mysql.com/)

For optional (but strongly recommended) caching you need:

* [Memcached](https://memcached.org/)

## Installing the required software

These instructions assume you are running a UNIX-like operating system (Linux, macOS, etc).

### Corpus Workbench

Download the current stable version of [Corpus Workbench](https://cwb.sourceforge.io/). Install by following the
[*Installing the CWB Core*](https://cwb.sourceforge.io/install.php) instructions, either by using the provided
packages or building from source. Refer to the included `INSTALL` text file for further instructions.

Once CWB is installed, by default you will find it under `/usr/local/cwb-X.X.X/bin` (where `X.X.X` is the version
number). Confirm that the installation was successful by running:

    /usr/local/cwb-X.X.X/bin/cqp -v

CWB needs two directories for storing the corpora. One for the data, and one for the corpus registry.
You may create these directories wherever you want, but from here on we will assume that you have created the
following two:

    /corpora/data
    /corpora/registry


## Setting up the Python environment and requirements

Optionally you may set up a virtual Python environment:

    $ python3 -m venv venv
    $ source venv/bin/activate

Install the required Python modules using `pip` with the included `requirements.txt`.

    $ pip3 install -r requirements.txt

If you wish to be able to run tests (useful mainly for Korp developers), install from `requirements-dev.txt` instead of `requirements.txt`.


## Configuring Korp

The supplied `config.py` contains the default configuration. To override the default configuration, make a copy
of `config.py` and place it in a directory named `instance` in the repo root directory, and edit that copy.

The following variables need to be set for Korp to work:

* `CQP_EXECUTABLE`  
The absolute path to the CQP binary. By default `/usr/local/cwb-X.X.X/bin/cqp`

* `CWB_SCAN_EXECUTABLE`  
The absolute path to the cwb-scan-corpus binary. By default `/usr/local/cwb-X.X.X/cwb-scan-corpus`

* `CWB_REGISTRY`  
The absolute path to the CWB registry files. This is the `/corpora/registry` folder you created before.

If you are planning on using functionality dependent on a database, you also need to set the following variables:

* `DBNAME`  
The name of the MySQL database where the corpus data will be stored.

* `DBUSER & DBPASSWORD`  
Username and password for accessing the database.

For caching to work you need to specify both a cache directory (`CACHE_DIR`) and a Memcached server address or socket
(`MEMCACHED_SERVER`).


## Running the backend

To run the backend, simply run `run.py`:

    python3 run.py

The backend should then be reachable in your web browser on the port you configured in `config.py`, for
example `http://localhost:1234`.

During development or while testing your configuration, use the flag `dev` for automatic reloading.

    python3 run.py dev

For deployment, [Gunicorn](http://gunicorn.org/) works well.

    gunicorn --worker-class gevent --bind 0.0.0.0:1234 --workers 4 --max-requests 250 --limit-request-line 0 'run:create_app()'


## Cache management

Most caching is done using Memcached, except for CWB query results which are temporarily saved to disk to speed up KWIC
pagination.
While Memcached handles removing old cache by itself, you will still have to tell it to invalidate parts of the cache
when one or more corpora are updated or added. This, and cleaning up the disk cache, is easily done by accessing the
`/cache` endpoint. It might be a good idea to set up a cronjob or similar to regularly do this, making the cache
maintenance fully automatic.


## API documentation

The API documentation is available as an OpenAPI specification in [docs/api.yaml](docs/api.yaml), or online at [https://ws.spraakbanken.gu.se/docs/korp](https://ws.spraakbanken.gu.se/docs/korp).


## Adding corpora

Korp works as a layer on top of Corpus Workbench for most corpus search functionality. See the [CWB corpus encoding tutorial](http://cwb.sourceforge.net/files/CWB_Encoding_Tutorial.pdf) for information regarding encoding corpora.
Note that Korp requires your corpora to be encoded in UTF-8. Values of structural attributes may not contain tab characters.
Once CWB is aware of your corpora they will be accessible through the Korp API.

### Adding additional info about the corpus

For Korp to show the number of sentences and the date when a corpus was last updated, you have to manually add this information.
Create a file called ".info" in the directory of the CWB data files for the corpus, and add to it the following lines (editing the values to match your material). Be sure to end the file with a blank line:

    Sentences: 12345
    Updated: 2019-11-30
    FirstDate: 2001-01-16 00:00:00
    LastDate: 2001-01-30 23:59:59

Once this file is in place, Korp will be able to access this information.


### Corpus structure requirements

To use the basic concordance features of Korp there are no particular requirements
regarding the markup of your corpora.

To use the **Word Picture** functionality your corpus must adhere to the following format:

* The structural annotation marking sentences must be named `sentence`.
* Every sentence annotation must have an attribute named `id` with a value that is unique within the corpus.

To use the **Trend Diagram** functionality, your corpus needs to be annotated with date information using
the following four structural attributes: `text_datefrom`, `text_timefrom`, `text_dateto`, `text_timeto`.
The date format should be *YYYYMMDD*, and the time format *hhmmss*.
A corpus dated 2006 would have the following values:

* `text_datefrom:  20060101`
* `text_timefrom:  000000`
* `text_dateto:    20061231`
* `text_timeto:    235959`


## Database tables

This section describes the database tables needed to use the Word Picture, Lemgram index and Trend Diagram features.
If you don't need any of these features, you can skip this section.


### Relations for the Word Picture

The Word Picture data consists of head-relation-dependent triplets and frequencies.
For every corpus, you need five database tables. The table structures are as follows:

    Table name: relations_CORPUSNAME  
    Charset:    UTF-8  

    Columns:  
        id             int                  A unique ID (within this table)  
        head           int                  Reference to an ID in the strings table (below). The head word in the relation  
        rel            enum(...)            The syntactic relation  
        dep            int                  Reference to an ID in the strings table (below). The dependent in the relation  
        freq           int                  Frequency of the triplet (head, rel, dep)  
        bfhead         bool                 True if head is a base form (or lemgram)  
        bfdep          bool                 True if dep  is a base form (or lemgram)  
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


In the main `relations_CORPUSNAME` table, each relation should be represented three times. Once with both dependent
and head as base forms, once with dependent as base form and head as word form, and once with dependent as
word form and head as base form. This is to allow searching for both base forms and word forms, giving different
results for different searched word forms, while the results are always displayed as base forms.
If the base form annotation is missing for a dependent, head or both, the word form can be used as both word form and
base form by setting both bfhead/bfdep and wfhead/wfdep to True. In such a case you won't need all three rows for
that relation.

The `sentences` table contains sentence IDs for sentences containing the relations, with start and end
values to point out exactly where in the sentences the relations occur (1 being the first word of the sentence).

### Lemgram Index

The lemgram index is an index of every lemgram in every corpus, along with the
number of occurrences. This is used by the frontend to grey out auto-completion
suggestions which would not give any results in the selected corpora. The lemgram
index consists of a single MySQL table, with the following layout:


    Table name: lemgram_index  
    Charset:    UTF-8  

    Columns:  
        lemgram      varchar(64)         The lemgram  
        freq         int                 Number of occurrences  
        freq_prefix  int                 Number of occurrences as a prefix  
        freq_suffix  int                 Number of occurrences as a suffix  
        corpus       varchar(64)         The corpus name  

    Indexes:  
        (lemgram, corpus, freq, freq_prefix, freq_suffix)


### Time data

For the Trend Diagram, you need to add token-per-time-span data to your database. For tokens without
date or time info, use the date 0000-00-00 00:00:00.
Use the following table layout:


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


## Corpus Configuration for the Korp Frontend

The corpus configuration used by the Korp frontend is served by the backend. In `config.py`, the variable
`CORPUS_CONFIG_DIR` should point to a directory having the following structure:

```
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
  attribute presets referred to by the corpus configurations.

For some inspiration, [here](https://github.com/spraakbanken/korp-config) are the config files used by the Korp instance
at Språkbanken Text.

**Note:**  
Most settings in these files referring to labels or descriptions can optionally be localized using ISO 639-3 language
codes. For example, a label can look both like this:

    label: author

... and like this:

    label:
      eng: author
      swe: författare

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
  ```
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
  refer to folders by using the prefix `__`, and dot-notation for refering to subfolders. Example:
  ```
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
- **modes**: A list of the modes in which the corpus will be included, optionally specifying a folder. Example:
  ```
  mode:
    - name: default
      folder: novels.classics
  ```

**Optional:**

- **within**: Use this to override **default_within** (set in the global or mode config). **within** is a list of
 structural elements to use as boundaries when searching, ordered from smaller to bigger. Example:
  ```
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
  ```
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
- **attribute_filters**: A list of structural attributes on which the user will be able to filter the search results,
  using menus in both simple and extended search.
- **pos_attributes** and **struct_attributes**: Lists of positional and structural attributes. Every item in each list
  should be an object with one key. The key should be the ID of the attribute, e.g. `msd` for positional attributes or
  `text_title` for structural. The value should be either 1) an object with a complete attribute definition, or 2) a
  string referring to an attribute preset containing such a definition, e.g. `msd` to refer to
  `attributes/positional/msd.yaml`. With option 1, you may also refer to a preset by using the key
  `preset` and then extend/override that preset. The attribute definition is what tells the Korp frontend how to handle
  each attribute, like how it should be presented in the sidebar and what interface widget to use in extended search.
  For more information about what options are available for attribute definitions, see the [Korp frontend
  documentation](https://github.com/spraakbanken/korp-frontend/blob/master/doc/frontend_devel.md#attribute-settings).
  Example:
  ```
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
- **limited_access**: Set to `true` to indicate that this corpus requires the user to be logged in and having the right
  permissions.

### Attribute Presets

See **pos_attributes** and **struct_attributes** above.
