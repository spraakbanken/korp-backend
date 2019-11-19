# Korp Backend

This is the backend for [Korp](https://spraakbanken.gu.se/korp), a corpus search tool developed by
[Språkbanken](https://spraakbanken.gu.se) at the University of Gothenburg, Sweden.

The code is distributed under the [MIT license](https://opensource.org/licenses/MIT).

The Korp backend is a Python 3 WSGI application, acting as a wrapper for Corpus Workbench.


## Requirements

To use the basic features of the Korp backend you need the following:

* [Python 3.3+](http://python.org/)
* [Corpus Workbench](http://cwb.sourceforge.net/beta.php) (CWB) 3.4.12 or newer

To use the additional features such as the Word Picture you also need:

* [MariaDB](https://mariadb.org/) or [MySQL](http://www.mysql.com/)

For optional caching you need:

* [Memcached](https://memcached.org/)
* libmemcached-dev


## Installing the required software

These instructions assume you are running a UNIX-like operating system (Linux, macOS, etc).


### Corpus Workbench

You will need the latest [beta version of CWB](http://cwb.sourceforge.net/beta.php). Install by following these steps:

Check out the latest version of the source code using Subversion by running the following command in a terminal:

    $ svn co http://svn.code.sf.net/p/cwb/code/cwb/trunk cwb

Refer to the INSTALL text file for instructions on how to build and install on your system. The source code comes with
a bunch of install scripts, so if you're lucky all you have to do is run one of them. For example:

    $ sudo ./install-scripts/install-linux

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

Install the required Python modules using `pip` with the included requirements.txt.

    $ pip3 install -r requirements.txt


## Configuring Korp

Normally nothing needs to be changed in the korp.py file, and all configuration is done by editing config.py.
The following variables need to be set:

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

For caching to work you need to specify both a cache directory (`CACHE_DIR`) and a list of Memcached servers
or sockets (`MEMCACHED_SERVERS`).


## Running the backend

To run the backend, simply run korp.py:

    python3 korp.py

The backend should then be reachable in your web browser on the port you configured in config.py, for
example `http://localhost:1234`.

During development or while testing your configuration, use the flag `dev` for automatic reloading.

    python3 korp.py dev

For deployment, [Gunicorn](http://gunicorn.org/) works well.

    gunicorn --worker-class gevent --bind 0.0.0.0:1234 --workers 4 --max-requests 250 --limit-request-line 0 korp:app


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
Note that Korp requires your corpora to be encoded in UTF-8.
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

For the Trend Diagram, you need to add token-per-time-span data to your database.
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
