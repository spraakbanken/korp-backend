# Korp Backend

This is the backend for [Korp](https://spraakbanken.gu.se/korp), a corpus search tool developed by [Språkbanken](https://spraakbanken.gu.se) at the University of Gothenburg, Sweden.

The code is distributed under the [MIT license](https://opensource.org/licenses/MIT).

The Korp backend is a Python 3 WSGI application, acting as a wrapper for Corpus Workbench.


## Requirements

To use the basic features of the Korp backend you need the following:

* [Python 3](http://python.org/)
* [Corpus Workbench](http://cwb.sourceforge.net/beta.php) (CWB) 3.4.12 or newer

To use the additional features such as the Word Picture you also need:

- [MariaDB](https://mariadb.org/) or [MySQL](http://www.mysql.com/)


## Installing the required software

These instructions assume you are running a UNIX-like operating system (Linux, macOS, etc).


### Corpus Workbench

You will need the latest [beta version of CWB](http://cwb.sourceforge.net/beta.php). Install by following these steps:

Check out the latest version of the source code using Subversion by running the following command in a terminal:

    $ svn co https://cwb.svn.sourceforge.net/svnroot/cwb/cwb/trunk cwb

Refer to the INSTALL text file for instructions on how to build and install on your system. The source code comes with a bunch of install scripts, so if you're lucky all you have to do is run one of them. For example on Ubuntu:

    $ sudo ./install-scripts/cwb-install-ubuntu

Once CWB is installed, by default you will find it under `/usr/local/cwb-X.X.X/bin` (where `X.X.X` is the version number).
Confirm that the installation was successful by running:

    /usr/local/cwb-X.X.X/bin/cqp -v

CWB needs two directories for storing the corpora. One for the data, and one for the corpus registry.
You may create these directories wherever you want, but from here on we will assume that you have created the following two:

    /corpora/data
    /corpora/registry

<!-- To make things easier you should add two environment variables:
export CWB_DATADIR=/corpora/data
export CORPUS_REGISTRY=/corpora/registry -->


## Setting up the Python environment and requirements

Optionally you may set up a virtual Python environment:

    $ python3 -m venv venv
    $ source venv/bin/activate

Install the required Python modules using `pip` with the included requirements.txt.

    $ pip3 install -r requirements.txt


## Configuring Korp

Normally nothing needs to be changed in the korp.py file, and all configuration is done by editing config.py. The following variables need to be set:

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


## Running the backend

To run the backend, simply run korp.py:

    python3 korp.py

The backend should then be reachable in your web browser on the port you configured in config.py, for example `http://localhost:1234`.
