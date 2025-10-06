
"""
tests/dbutils.py

Utilities used in pytest tests to create and populate a Korp MySQL
test database.

The test database should typically be different from the production
database, so this module contains facilities for creating a database
from scratch.

Individual database tables are created based on SQL or TSV files in
the specified test data directory. TSV file names are mapped to tables
and their definitions in YAML files in the subdirectory "tableinfo".
For more information, please see the documentation in tests/README.md.
"""


# TODO:
# - Specify default collation for db
# - Remove corpus data before importing to a multi-corpus table


import csv
import re

from collections import defaultdict
from string import Formatter

import MySQLdb
import yaml

from tests.configutils import get_korp_config


class KorpDatabase:

    """
    Class providing access to a Korp MySQL database for testing.

    A KorpDatabase object represents the configuration for a Korp
    MySQL database. An actual database is created with create() and
    dropped with drop().

    A KorpDatabase object should be created only after calling
    KorpDatabase.pytest_config_db_options(config) from
    pytest_configure(config) in conftest.py.
    """

    # Custom pytest command-line options (without the prefix "--db-")
    # affecting the Korp MySQL test database and their help strings
    # (or dicts of keyword arguments to argparse.addoption()), where
    # {} is replaced with the metavar
    _pytest_db_option_help = {
        "host": "Use host {} for the Korp MySQL test database",
        "port": dict(
            type=int,
            help="Use port {} for the Korp MySQL test database"
        ),
        "name": "Use database name {} for the Korp MySQL test database",
        "user": "Use user {} to access the Korp MySQL test database",
        "password": "Use password {} to access the Korp MySQL test database",
        "create-user": "Use user {} to create the Korp MySQL test database",
        "create-password": (
            "Use password {} to create the Korp MySQL test database"),
    }
    # The custom pytest command-line options
    _pytest_db_options = {}

    def __init__(self, datadir):
        """Initialize KorpDatabase but do not create an actual database yet.

        Use datadir as the database data directory.
        """
        # Database name; None if no database active
        self.dbname = None
        # Possible error that occurred when trying to create database:
        # a dict with keys "exception" (MySQLdb.Error object),
        # "message" (stringified error object) and "sql" (SQL
        # statement or None)
        self.create_error = None
        # Database data directory
        self._datadir = datadir
        # Database options: pytest command-line options combined with
        # options from the Korp configuration; keys are lowercase
        # without a "db" prefix
        self._db_options = {}
        # MySQL database connection parameters
        self._conn_params = {}
        # Table information
        self._tableinfo = []
        # Filename patterns by table type
        self._tabletype_patts = defaultdict(list)
        # Initialize self._tableinfo, self._tabletype_patts
        self._read_tableinfo()
        # If True, use an existing table in the database, so do not
        # drop it afterwards
        self._use_existing_table = False
        self._make_db_options(self._pytest_db_options)

    @classmethod
    def pytest_add_db_options(cls, parser):
        """Add database-related pytest command-line options via pytest parser

        To be called from pytest_addoption in conftest.py.
        """
        for opt, args in cls._pytest_db_option_help.items():
            if isinstance(args, str):
                args = dict(help=args)
            args["metavar"] = opt.replace("create-", "").upper()
            args["help"] = args["help"].replace("{}", "%(metavar)s")
            parser.addoption(f"--db-{opt}", **args)

    @classmethod
    def pytest_config_db_options(cls, config):
        """Get the values database-related pytest command-line options

        To be called from pytest_configure in conftest.py.
        """
        cls._pytest_db_options = dict([(opt, config.getoption(f"--db-{opt}"))
                                       for opt in cls._pytest_db_option_help])

    def _make_db_options(self, pytest_db_opts):
        """Set database options based on pytest_db_opts and Korp config

        Set database options (self._db_options) and connection
        parameters (self._conn_params) for creating a database.

        Take Korp configuration option values (DB*) as the basis and
        override them with possible values specified as custom pytest
        command-line options (in pytest_db_opts) --db-*. If
        --db-create-user or --db-create-password have not been
        specified, use the values of --db-user (DBUSER) and
        --db-password (DBPASSWORD), respectively.

        For connection options, user and password primarily those in
        create-user and create-password, and charset is taken from
        DBCHARSET in Korp configuration.
        """
        db_opts = pytest_db_opts.copy()
        korp_conf = get_korp_config()
        for key, val in db_opts.items():
            if val is None:
                if "create" in key:
                    db_opts[key] = db_opts.get(key.replace("create-", ""))
                elif key != "name":
                    db_opts[key] = korp_conf.get("DB" + key.upper(), "")
        self._conn_params = dict(
            [(key.rsplit("-")[-1], db_opts[key])
             for key in ["host", "port", "create-user", "create-password"]])
        self._conn_params["charset"] = korp_conf["DBCHARSET"]
        self._db_options = db_opts

    def get_config(self):
        """Return database configuration dict compatible with Korp config

        The keys in the returned dict are in uppercase, prefixed with
        "DB". Keys with value None are not included.
        """
        return dict([("DB" + name.upper(), val)
                     for name, val in self._db_options.items()
                     if val is not None])

    def _connect(self):
        """Return a MySQLdb Connection using the pre-specified parameters."""
        return MySQLdb.Connect(local_infile=True, **self._conn_params)

    def execute_get_cursor(self, sql, cursor=None, commit=True):
        """Execute SQL statements sql on cursor and commit if commit == True.

        sql can be str or an iterable of str, in which case all the
        items are concatenated and executed with a single call.
        If cursor is None, create a connection to the database and a
        cursor for it.
        Return the number of rows affected and the cursor.
        """
        if not isinstance(sql, str):
            sql = "".join(sql)
        if cursor is None:
            with self._connect() as conn:
                return self.execute_get_cursor(sql, conn.cursor())
        else:
            retval = cursor.execute(sql)
            if commit:
                # cursor.connection.commit() may result in
                # MySQLdb.ProgrammingError: (2014, "Commands out of
                # sync; you can't run this command now") (why?); if
                # that happens, use cursor.execute("COMMIT;"), which
                # however seems to reset the cursor.
                try:
                    cursor.connection.commit()
                except MySQLdb.ProgrammingError:
                    cursor.execute("COMMIT;")
            return retval, cursor

    def execute(self, sql, cursor=None, commit=True):
        """Execute SQL statements sql on cursor and commit if commit == True.

        The arguments and functionality are the same as for
        execute_get_cursor, but only return the number of rows
        affected.
        """
        return self.execute_get_cursor(sql, cursor, commit)[0]

    def execute_file(self, sqlfile, cursor=None, commit=True):
        """Execute SQL statements in sqlfile on cursor and commit if commit.

        If cursor is None, create a connection to the database and a
        cursor for it.
        Return the number of rows affected.
        """
        with open(sqlfile, "r") as sqlf:
            return self.execute(sqlf, cursor, commit=commit)

    def create(self):
        """Create a Korp MySQL database and grant privileges

        Create a Korp MySQL database using the pre-defined connection
        parameters, unless one has already been created (and not
        dropped) for self. Database name is generated in
        _make_db_name, user is taken from _db_options and host from
        _conn_params.
        """
        if self.dbname is not None:
            # If a database has already been created, do not create
            # another
            return
        korp_conf = get_korp_config()
        try:
            sql = None
            with self._connect() as conn:
                cursor = conn.cursor()
                dbname = self._make_db_name(cursor)
                charset = korp_conf['DBCHARSET']
                user = self._db_options['user']
                host = self._conn_params['host']
                for sql in [
                        f"CREATE DATABASE {dbname} CHARACTER SET {charset};",
                        f"GRANT ALL ON {dbname}.* TO '{user}'@'{host}';",
                ]:
                    self.execute(sql, cursor)
        except MySQLdb.Error as exc:
            self.create_error = {
                "exception": exc,
                "message": str(exc),
                "sql": sql,
            }
            return
        self._set_db_name(dbname)
        self.create_error = None

    def _set_db_name(self, dbname):
        """Set current database name to dbname."""
        self.dbname = self._conn_params["database"] = dbname

    def drop(self):
        """Drop the created database and set current database name to None."""
        if self.dbname and not self._use_existing_table:
            self.execute(f"DROP DATABASE {self.dbname};")
        self._set_db_name(None)

    def _make_db_name(self, cursor):
        """Return a name for the Korp test database

        If database options contains non-None value for "name", use it
        and set _use_existing_table to True. Otherwise, use the
        configured DBNAME with suffix "_pytest_N" where N is the
        smallest non-negative integer for which such a database does
        not yet exist. cursor is used to get a list of existing
        database names.
        """
        if self._db_options["name"] is not None:
            self._use_existing_table = True
            return self._db_options["name"]
        existing_db_names = self._get_db_names(cursor)
        db_name_base = get_korp_config().get("DBNAME", "korp") + "_pytest_"
        i = 0
        while db_name_base + str(i) in existing_db_names:
            i += 1
        db_name = db_name_base + str(i)
        self._db_options["name"] = db_name
        self._use_existing_table = False
        return db_name

    def _get_db_names(self, cursor):
        """Return a list of database names using MySQLdb cursor."""
        # commit=False to guarantee that we can retrieve database
        # names via cursor
        self.execute("SHOW DATABASES;", cursor, commit=False)
        return [item[0] for item in cursor]

    def _read_tableinfo(self):
        """Read table information YAML files and return the info as a list.

        Read table information YAML files in the "tableinfo"
        subdirectory of the data directory and return the information
        objects a list.
        """

        def compile_filenames(filenames):
            """Return a list of compiled regexps for the list filenames

            If a filename does not end in ".tsv", add the suffix. If a
            filename does not begin with ".*/", add the prefix.
            Replace corpus name placeholder "{corpus}” with
            "(?P<corpus>[a-zA-Z0-9_-]+?)".
            """
            filenames_re = []
            for regex in filenames:
                if not regex.endswith(r"\.tsv"):
                    regex = regex + r"\.tsv"
                if not regex.startswith(r".*/"):
                    regex = r".*/" + regex
                regex = regex.replace("{corpus}", "(?P<corpus>[a-zA-Z0-9_-]+?)")
                filenames_re.append(re.compile(regex))
            return filenames_re

        def add_tabletype(tableinfo_items, tabletype):
            """Add key "tabletype" with value tabletype to tableinfo_items.

            Skip items lacking key "filenames".
            Return tableinfo_items.
            """
            for item in tableinfo_items:
                if "filenames" in item:
                    item["tabletype"] = tabletype
            return tableinfo_items

        def expand_vars(tableinfo_items):
            """Expand variables in table definitions in tableinfo_items.

            Replace variable references "{var}" in the value of
            "definition" of tableinfo_items.

            Variable values are defined in separate sequence items
            that are mappings containing key "definition_vars", whose
            value is a mapping whose keys are variable names and
            values the replacement values.

            The returned result contains tableinfo_items with mappings
            containing key "definition_vars" removed and values for
            "definition" expanded in other mappings.
            """
            result = []
            vardefs = {}
            for item in tableinfo_items:
                if "definition_vars" in item:
                    vardefs.update(item["definition_vars"])
                else:
                    item["definition"] = item["definition"].format(**vardefs)
                    result.append(item)
            return result

        tableinfo_dir = self._datadir / "tableinfo"
        tableinfo = []
        for filepath in tableinfo_dir.glob("*.yaml"):
            with open(str(filepath), "r") as f:
                tableinfo_new = yaml.safe_load(f)
                tableinfo_new = add_tabletype(tableinfo_new, filepath.stem)
                tableinfo.extend(expand_vars(tableinfo_new))
        for info in tableinfo:
            # For filenames and exclude_filenames, add corresponding
            # *_re keys with compiled regular expressions
            for propname in ["filenames", "exclude_filenames"]:
                info[f"{propname}_re"] = compile_filenames(
                    info.get(propname, []))
            # Add filename patterns for the tabletype
            for filename in info["filenames"]:
                if not filename.startswith(".*/"):
                    filename = ".*/" + filename
                self._tabletype_patts[info["tabletype"]].append(filename)
        self._tableinfo = tableinfo

    def import_tables(self, corpora, tabletypes=None):
        """Import database tables of tabletypes (or all) for corpora.

        Import database tables in TSV or SQL files matching patterns
        in self._tabletype_patts for the tabletypes and corpora.
        corpora and tabletypes may be single strings or lists of
        strings. If tabletypes is None (default), import all types of
        tables for corpora.
        """
        files = self._find_table_files(corpora, tabletypes)
        self.import_table_files(files)

    def _find_table_files(self, corpora, tabletypes=None):
        """Return a list of table data file names for corpora and tabletypes."""
        if tabletypes is None:
            tabletypes = self._tabletype_patts.keys()
        elif isinstance(tabletypes, str):
            tabletypes = [tabletypes]
        if isinstance(corpora, str):
            corpora = [corpora]
        files = []
        for ext in ["sql", "tsv"]:
            for filename in self._datadir.rglob(f"*.{ext}"):
                filename = str(filename)
                for tabletype in tabletypes:
                    for corpus in corpora:
                        for patt in self._tabletype_patts[tabletype]:
                            patt = patt.replace("{corpus}", corpus) + f".{ext}"
                            if re.fullmatch(patt, filename):
                                files.append(filename)
        return files

    def import_table_files(self, tablefile_globs):
        """Import table data from files matched by tablefile_globs."""

        def find_files(tablefile_glob):
            """Find files in tablefile_globs or use directly if absolute.

            If tablefile_glob begins with a "/", it is an absolute and
            returned as a single-item list. Otherwise, return a glob
            generator for tablefile_glob.
            """
            if tablefile_glob and tablefile_glob[0] == "/":
                return [tablefile_glob]
            else:
                return self._datadir.glob(tablefile_glob)

        with self._connect() as conn:
            cursor = conn.cursor()
            for tablefile_glob in tablefile_globs:
                for tablefile in find_files(tablefile_glob):
                    tablefile = str(tablefile)
                    if tablefile.endswith(".sql"):
                        self.execute_file(tablefile, cursor)
                    else:
                        self._import_table(tablefile, cursor)

    def _import_table(self, tablefile, cursor):
        """Import table data from tablefile using cursor.

        Raise ValueError if no table info has a matching rule for file
        name tablefile.
        """
        tableinfo, corpus = self._find_tableinfo(tablefile)
        if tableinfo is None:
            raise ValueError(
                f"No table info matches file name \"{tablefile}\"")
        tablename = self._create_table(tableinfo, corpus, cursor)
        self._load_file(tablename, tablefile, cursor)

    def _find_tableinfo(self, tablefile):
        """Find and return table information for file tablefile.

        Find and return the first table information item in
        self._tableinfo for tablefile in which one of the file name
        regexps (filenames_re) match tablefile and none of excluded
        file name regexps (exclude_filename_re) match.

        Return a tuple (info, corpus) where corpus is the part
        matching "{corpus}" in the file name regexp, or (None, None)
        if no table info was found.
        """
        for info in self._tableinfo:
            for regex in info["filenames_re"]:
                mo = regex.fullmatch(tablefile)
                if mo and not any(exclude.fullmatch(tablefile)
                                  for exclude in info["exclude_filenames_re"]):
                    corpus = mo.groupdict().get("corpus")
                    return info, corpus
        return None, None

    def _create_table(self, tableinfo, corpus, cursor):
        """Create table based on tableinfo and corpus id using cursor.

        If the table already exists, do not do anything.
        Return the name of the created table.
        """
        tablename = self._make_tablename(tableinfo, corpus)
        self.execute(
            f"""CREATE TABLE IF NOT EXISTS `{tablename}` (
                {tableinfo["definition"]}
                );""",
            cursor
        )
        return tablename

    def _make_tablename(self, tableinfo, corpus):
        """Return table name based on tableinfo and corpus id.

        Take the table name from tableinfo["tablename"] and replace
        the possible format placeholders {corpus} and {CORPUS} in it
        with the corpus id in lower or upper case, respectively.
        """
        tablename = tableinfo["tablename"]
        return tablename.format(corpus=corpus.lower(), CORPUS=corpus.upper())

    def _load_file(self, tablename, tablefile, cursor):
        """Load the data from tablefile to table tablename using cursor.

        Load the data from TSV file tablefile using LOAD DATA LOCAL
        INFILE. This thus requires allowing LOAD DATA INFILE.
        """
        self.execute(
            f"""LOAD DATA LOCAL INFILE '{tablefile}' INTO TABLE `{tablename}`
                FIELDS ESCAPED BY '';""",
            cursor)
