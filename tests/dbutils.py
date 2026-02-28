"""Utilities used in pytest tests to create and populate a Korp MySQL test database.

The test database should typically be different from the production database, so this module contains facilities for
creating a database from scratch.

Individual database tables are created based on SQL or TSV files in the specified test data directory. TSV file names
are mapped to tables and their definitions in YAML files in the subdirectory "tableinfo". For more information, please
see the documentation in tests/README.md.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any, ClassVar

import pytest
import yaml
from sqlalchemy import create_engine
from sqlalchemy.engine import URL, Connection, Engine
from sqlalchemy.pool import NullPool

from tests.configutils import get_korp_config


class KorpDatabase:
    """Class providing access to a Korp MySQL database for testing.

    A `KorpDatabase` object represents the configuration for a Korp MySQL database. An actual database is created with
    `create()` and dropped with `drop()`.

    A `KorpDatabase` object should be created only after calling `KorpDatabase.pytest_config_db_options(config)` from
    `pytest_configure(config)` in `conftest.py`.
    """

    # Custom pytest command-line options (without the prefix "--db-") affecting the Korp MySQL test database and their
    # help strings (or dicts of keyword arguments to argparse.addoption()), where {} is replaced with the metavar
    _pytest_db_option_help: ClassVar[dict] = {
        "host": "Use host {} for the Korp MySQL test database",
        "port": {
            "type": int,
            "help": "Use port {} for the Korp MySQL test database",
        },
        "name": "Use database name {} for the Korp MySQL test database",
        "user": "Use user {} to access the Korp MySQL test database",
        "password": "Use password {} to access the Korp MySQL test database",
        "create-user": "Use user {} to create the Korp MySQL test database",
        "create-password": "Use password {} to create the Korp MySQL test database",
        "collate": (
            "Use {} as the Korp MySQL test database collation."
            " If not specified, use the collation of the Korp MySQL database,"
            " or if that cannot be accessed, the default collation for the"
            " Korp MySQL database character set."
        ),
    }
    # The custom pytest command-line options
    _pytest_db_options: ClassVar[dict] = {}

    def __init__(self, datadir: Path) -> None:
        """Initialize KorpDatabase but do not create an actual database yet.

        Args:
            datadir: The database data directory, with table information and table data files.
        """
        # Database name; None if no database active
        self.dbname = None
        # Possible error that occurred when trying to create database: a dict with keys "exception" (Exception object),
        # "message" (stringified error object) and "sql" (SQL statement or None)
        self.create_error = None
        # Database data directory
        self._datadir = datadir
        # Database options: pytest command-line options combined with options from the Korp configuration; keys are
        # lowercase without a "db" prefix
        self._db_options = {}
        # MySQL database connection parameters
        self._conn_params = {}
        # Table information
        self._table_info = []
        # Filename patterns by table type
        self._table_type_patts = defaultdict(list)
        # Table info by table type
        self._table_type_info = defaultdict(list)
        # Initialize self._table_info, self._table_type_patts
        self._read_table_info()
        # If True, use an existing table in the database, so do not drop it afterwards
        self._use_existing_table = False
        self._make_db_options(self._pytest_db_options)

    @classmethod
    def pytest_add_db_options(cls, parser: pytest.Parser) -> None:
        """Add database-related pytest command-line options via pytest parser.

        To be called from pytest_addoption in conftest.py.
        """
        for opt, args in cls._pytest_db_option_help.items():
            if isinstance(args, str):
                args = {"help": args}  # noqa: PLW2901
            args["metavar"] = opt.replace("create-", "").upper()
            args["help"] = args["help"].replace("{}", "%(metavar)s")
            parser.addoption(f"--db-{opt}", **args)

    @classmethod
    def pytest_config_db_options(cls, config: pytest.Config) -> None:
        """Process the database-related pytest command-line options from pytest config.

        To be called from pytest_configure in conftest.py.
        """
        cls._pytest_db_options = {opt: config.getoption(f"--db-{opt}") for opt in cls._pytest_db_option_help}

    @staticmethod
    def _split_sql_statements(sql: str) -> list[str]:
        """Split SQL script into statements, ignoring semicolons inside quotes.

        We need this since we're using exec_driver_sql, which does not support executing multiple statements at once.

        Args:
            sql: The SQL script to split into statements.

        Returns:
            A list of SQL statements.
        """
        statements: list[str] = []
        start = 0
        in_single = False
        in_double = False
        in_backtick = False
        escaped = False

        for i, char in enumerate(sql):
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == "'" and not in_double and not in_backtick:
                in_single = not in_single
                continue
            if char == '"' and not in_single and not in_backtick:
                in_double = not in_double
                continue
            if char == "`" and not in_single and not in_double:
                in_backtick = not in_backtick
                continue
            if char == ";" and not in_single and not in_double and not in_backtick:
                if statement := sql[start : i + 1].strip():
                    statements.append(statement)
                start = i + 1

        if tail := sql[start:].strip():
            statements.append(tail)

        return statements

    @staticmethod
    def _escape_sql_string(value: str) -> str:
        """Return a SQL-escaped version of `value`."""
        return value.replace("\\", "\\\\").replace("'", "\\'")

    @staticmethod
    def _make_engine(
        *,
        host: str,
        port: int,
        user: str,
        password: str,
        charset: str,
        database: str | None,
        local_infile: bool,
    ) -> Engine:
        url = URL.create(
            "mysql+pymysql",
            username=user or None,
            password=password or None,
            host=host,
            port=port,
            database=database or None,
            query={"charset": charset},
        )
        return create_engine(
            url,
            poolclass=NullPool,
            pool_pre_ping=True,
            connect_args={"local_infile": local_infile},
        )

    @contextmanager
    def _connection(self, *, include_database: bool, local_infile: bool = True) -> Iterator[Connection]:
        """Yield a database connection and always dispose its engine afterwards.

        Args:
            include_database: If `True`, connect to the database specified in the configuration. Otherwise, connect
                without specifying a database.
            local_infile: If `True`, allow loading data from local files with `LOAD DATA LOCAL INFILE`.

        Yields:
            A SQLAlchemy Connection to the database.
        """
        engine = self._make_engine(
            host=self._conn_params["host"],
            port=int(self._conn_params["port"]),
            user=self._conn_params["user"],
            password=self._conn_params["password"],
            charset=self._conn_params["charset"],
            database=self._conn_params.get("database") if include_database else None,
            local_infile=local_infile,
        )
        try:
            with engine.connect() as conn:
                yield conn
        finally:
            engine.dispose()

    def _make_db_options(self, pytest_db_opts: dict[str, Any]) -> None:
        """Set database options based on `pytest_db_opts` and Korp config.

        Set database options (`self._db_options`) and connection parameters (`self._conn_params`) for creating a
        database.

        Take Korp configuration option values (`DB_*`) as the basis and override them with possible values specified as
        custom pytest command-line options (in `pytest_db_opts`) `--db-*`. If `--db-create-user` or
        `--db-create-password` have not been specified, use the values of `--db-user` (`DB_USER`) and `--db-password`
        (`DB_PASSWORD`), respectively.

        For connection options, user and password primarily those in create-user and create-password, and charset is
        taken from `DB_CHARSET` in Korp configuration.
        """
        db_opts = pytest_db_opts.copy()
        korp_conf = get_korp_config()
        for key, val in db_opts.items():
            if val is None:
                if "create" in key:
                    db_opts[key] = db_opts.get(key.replace("create-", ""))
                elif key != "name":
                    db_opts[key] = korp_conf.get(f"DB_{key.upper()}", "")
        self._conn_params = {
            key.rsplit("-")[-1]: db_opts[key] for key in ["host", "port", "create-user", "create-password"]
        }
        self._conn_params["charset"] = korp_conf["DB_CHARSET"]
        self._db_options = db_opts

    def get_config(self) -> dict[str, str]:
        """Return database configuration dict compatible with Korp config.

        The keys in the returned dict use Settings names (`DB_*`). Keys with value `None` are not included.
        """
        key_map = {
            "host": "DB_HOST",
            "port": "DB_PORT",
            "name": "DB_NAME",
            "user": "DB_USER",
            "password": "DB_PASSWORD",
        }
        return {key_map[name]: val for name, val in self._db_options.items() if val is not None and name in key_map}

    def execute(self, sql: str | Iterable[str], conn: Connection | None = None, commit: bool = True) -> int:
        """Execute SQL statements `sql` on `conn` and commit if `commit == True`.

        Args:
            sql: A SQL statement or an iterable of SQL statements to execute.
            conn: A SQLAlchemy Connection to use for executing the SQL statements, or `None` to create a new connection.
            commit: If `True`, commit the transaction after executing the SQL statements.

        Returns:
            The total number of rows affected by the executed SQL statements.
        """
        statements = self._split_sql_statements(sql if isinstance(sql, str) else "".join(sql))

        def _run(connection: Connection) -> int:
            count = sum(connection.exec_driver_sql(stmt).rowcount for stmt in statements)
            if commit:
                connection.commit()
            return count

        if conn is not None:
            return _run(conn)

        with self._connection(include_database=True) as connection:
            return _run(connection)

    def execute_file(self, sqlfile: Path, conn: Connection | None = None, commit: bool = True) -> int:
        """Execute SQL statements in `sqlfile` on `conn` and commit if `commit`.

        Args:
            sqlfile: The file name of the SQL file to execute.
            conn: A SQLAlchemy Connection to use for executing the SQL statements, or `None` to create a new connection.
            commit: If `True`, commit the transaction after executing the SQL statements.

        Returns:
            The total number of rows affected by the executed SQL statements.
        """
        with sqlfile.open(encoding="utf-8") as sqlf:
            return self.execute(sqlf, conn, commit=commit)

    @staticmethod
    def _get_db_names(conn: Connection) -> list[str]:
        """Return a list of database names using SQLAlchemy connection."""
        result = conn.exec_driver_sql("SHOW DATABASES;")
        return [item[0] for item in result.fetchall()]

    def _make_db_name(self, conn: Connection) -> str:
        """Return a name for the Korp test database.

        If database options contains non-None value for "name", use it and set `_use_existing_table` to `True`.
        Otherwise, use the configured `DB_NAME` with suffix "_pytest_N" where N is the smallest non-negative integer for
        which such a database does not yet exist.
        """
        if self._db_options["name"] is not None:
            self._use_existing_table = True
            return self._db_options["name"]

        existing_db_names = self._get_db_names(conn)
        db_name_base = get_korp_config().get("DB_NAME", "korp") + "_pytest_"
        i = 0
        while db_name_base + str(i) in existing_db_names:
            i += 1
        db_name = db_name_base + str(i)
        self._db_options["name"] = db_name
        self._use_existing_table = False
        return db_name

    def _get_collation(self, korp_conf: dict[str, Any]) -> str:
        """Get the collation for the Korp test database.

        Args:
            korp_conf: The Korp configuration dict.

        Returns:
            The collation to use for the Korp test database, or an empty string if it cannot be determined.
        """
        collate = self._db_options["collate"] or ""
        if collate:
            return collate

        engine = None
        try:
            engine = self._make_engine(
                host=korp_conf["DB_HOST"],
                port=int(korp_conf["DB_PORT"]),
                user=korp_conf["DB_USER"],
                password=korp_conf["DB_PASSWORD"],
                charset=korp_conf["DB_CHARSET"],
                database=korp_conf.get("DB_NAME") or None,
                local_infile=False,
            )
            with engine.connect() as conn:
                result = conn.exec_driver_sql("SELECT @@collation_database;")
                row = result.first()
                if row:
                    return row[0]
        except Exception:
            return ""
        finally:
            if engine is not None:
                engine.dispose()

        return ""

    def create(self) -> None:
        """Create a Korp MySQL database and grant privileges.

        Create a Korp MySQL database using the pre-defined connection parameters, unless one has already been created
        (and not dropped) for self. Database name is generated in `_make_db_name`, user is taken from _db_options and
        host from `_conn_params`.
        """
        if self.dbname is not None:
            # If a database has already been created, do not create
            # another
            return

        korp_conf = get_korp_config()
        sql = None
        try:
            with self._connection(include_database=False) as conn:
                dbname = self._make_db_name(conn)
                charset = korp_conf["DB_CHARSET"]
                collate = self._get_collation(korp_conf)
                collate_sql = f" COLLATE {collate}" if collate else ""
                user = self._db_options["user"]
                host = self._conn_params["host"]
                sqls = [
                    f"CREATE DATABASE {dbname} CHARACTER SET {charset}{collate_sql};",
                    f"GRANT ALL ON {dbname}.* TO '{user}'@'{host}';",
                ]
                for sql in sqls:
                    conn.exec_driver_sql(sql)
                conn.commit()
        except Exception as exc:
            self.create_error = {
                "exception": exc,
                "message": str(exc),
                "sql": sql,
            }
            return

        self._set_db_name(dbname)
        self.create_error = None

    def _set_db_name(self, dbname: str | None) -> None:
        """Set current database name to `dbname`."""
        self.dbname = self._conn_params["database"] = dbname

    def drop(self) -> None:
        """Drop the created database and set current database name to None."""
        if self.dbname and not self._use_existing_table:
            with self._connection(include_database=False) as conn:
                conn.exec_driver_sql(f"DROP DATABASE {self.dbname};")
                conn.commit()
        self._set_db_name(None)

    def _read_table_info(self) -> None:
        """Read table information YAML files and initialize self._table_info and self._table_type_patts.

        The table information YAML files are in the "tableinfo" subdirectory of the data directory.
        """

        def compile_filenames(filenames: list[str]) -> list[re.Pattern]:
            """Return a list of compiled regexps for the list filenames.

            If a filename does not end in ".tsv", add the suffix. If a filename does not begin with ".*/", add the
            prefix. Replace corpus name placeholder "{corpus}" with "(?P<corpus>[a-zA-Z0-9_-]+?)".
            """
            filenames_re = []
            for regex in filenames:
                if not regex.endswith(r"\.tsv"):
                    regex += r"\.tsv"  # noqa: PLW2901
                if not regex.startswith(r".*/"):
                    regex = r".*/" + regex  # noqa: PLW2901
                regex = regex.replace("{corpus}", "(?P<corpus>[a-zA-Z0-9_-]+?)")  # noqa: PLW2901
                filenames_re.append(re.compile(regex))
            return filenames_re

        def process_table_info(table_info_items: list[dict], table_type: str) -> list[dict]:
            """Return processed table info items with `table_type` set and definition variables expanded.

            Adds the `table_type` key and expands definition variable references. Items containing key "definition_vars"
            define variable values and are removed from the result; their values are used to expand "{var}" references
            in the "definition" of subsequent items.

            Args:
                table_info_items: A list of table information dicts.
                table_type: The table type to add to the dicts in table_info_items.

            Returns:
                The processed list of table information dicts.
            """
            result = []
            vardefs = {}
            for item in deepcopy(table_info_items):
                if "definition_vars" in item:
                    vardefs.update(item["definition_vars"])
                else:
                    item["definition"] = item["definition"].format(**vardefs)
                    if "filenames" in item:
                        item["table_type"] = table_type
                    result.append(item)
            return result

        table_info_dir = self._datadir / "tableinfo"
        table_info = []
        for filepath in table_info_dir.glob("*.yaml"):
            with filepath.open("r") as f:
                table_info_new = yaml.safe_load(f)
                table_info.extend(process_table_info(table_info_new, filepath.stem))
        for info in table_info:
            # For filenames and exclude_filenames, add corresponding *_re keys with compiled regular expressions
            for propname in ["filenames", "exclude_filenames"]:
                info[f"{propname}_re"] = compile_filenames(info.get(propname, []))
            # Add filename patterns for the table type
            for filename in info["filenames"]:
                if not filename.startswith(".*/"):
                    filename = ".*/" + filename  # noqa: PLW2901
                self._table_type_patts[info["table_type"]].append(filename)
            self._table_type_info[info["table_type"]].append(info)
        self._table_info = table_info

    def import_tables(self, corpora: str | list[str], table_types: str | Iterable[str] | None = None) -> None:
        """Import database tables of `table_types` (or all) for corpora.

        Import database tables in TSV or SQL files matching patterns in `self._table_type_patts` for the `table_types`
        and corpora. Possibly existing tables are first dropped to avoid interference between tests.

        Args:
            corpora: A corpus id or a list of corpus ids whose data to import.
            table_types: A table type or an iterable of table types to import, or `None` to import all types.
        """
        if table_types is None:
            table_types = self._table_type_patts.keys()
        elif isinstance(table_types, str):
            table_types = [table_types]
        if isinstance(corpora, str):
            corpora = [corpora]
        files = self._find_table_files(corpora, table_types)
        # It would probably be more efficient to delete existing data than to drop and re-create tables, but the latter
        # is simpler to implement
        self.drop_tables(corpora, table_types)
        self.import_table_files(files)

    def _find_table_files(self, corpora: Iterable[str], table_types: Iterable[str]) -> list[str]:
        """Return a list of table data file names for corpora and table_types."""
        # Pre-compile all patterns for each corpus and table type combination
        table_types = list(table_types)
        compiled_patterns = []
        for table_type in table_types:
            for corpus in corpora:
                for patt in self._table_type_patts[table_type]:
                    for ext in ["sql", "tsv"]:
                        full_patt = patt.replace("{corpus}", corpus) + f".{ext}"
                        compiled_patterns.append(re.compile(full_patt))

        files = []
        for filename in self._datadir.rglob("*.sql"):
            filename_str = str(filename)
            if any(patt.fullmatch(filename_str) for patt in compiled_patterns):
                files.append(filename_str)
        for filename in self._datadir.rglob("*.tsv"):
            filename_str = str(filename)
            if any(patt.fullmatch(filename_str) for patt in compiled_patterns):
                files.append(filename_str)
        return files

    def drop_tables(self, corpora: Iterable[str], table_types: Iterable[str]) -> None:
        """Drop possibly existing tables for `table_types` and `corpora`.

        Args:
            corpora: An iterable of corpus ids whose tables to drop.
            table_types: An iterable of table types whose tables to drop.
        """
        tables: set[str] = set()
        for table_type in table_types:
            for info in self._table_type_info[table_type]:
                if "{" in info["tablename"]:
                    # Table name contains corpus id
                    tables.update(self._make_table_name(info, corpus) for corpus in corpora)
                else:
                    tables.add(info["tablename"])

        if not tables:
            return

        table_names = ", ".join(f"`{table}`" for table in tables)
        self.execute(f"DROP TABLE IF EXISTS {table_names};")

    @staticmethod
    def _resolve_table_files(datadir: Path, table_file_glob: str) -> Iterable[Path]:
        """Resolve a table file glob to concrete file paths.

        If `table_file_glob` begins with a "/", it is treated as an absolute file name. Otherwise, it is resolved as a
        glob pattern in `datadir`.

        Args:
            datadir: The data directory in which to resolve `table_file_glob` if it is not an absolute file name.
            table_file_glob: An absolute file name or a glob pattern to be resolved in `datadir`.

        Returns:
            An iterable of Paths matching `table_file_glob`, or a single Path if `table_file_glob` is an absolute file
                name.
        """
        if table_file_glob and table_file_glob.startswith("/"):
            return [Path(table_file_glob)]
        return datadir.glob(table_file_glob)

    def import_table_files(self, table_file_globs: Iterable[str]) -> None:
        """Import table data from files matched by table_file_globs.

        Note that unlike `import_tables`, `import_table_files` does *not* first drop possibly existing tables.

        Args:
            table_file_globs: An iterable of file name globs to import, where a glob is either an absolute file name or
                a relative glob pattern to be resolved in the data directory.
        """
        with self._connection(include_database=True) as conn:
            for table_file_glob in table_file_globs:
                for table_file in self._resolve_table_files(self._datadir, table_file_glob):
                    table_file: Path
                    if table_file.suffix == ".sql":
                        self.execute_file(table_file, conn=conn, commit=False)
                    else:
                        self._import_table(conn, table_file)
            conn.commit()

    def _import_table(self, conn: Connection, table_file: Path) -> None:
        """Import table data from `table_file` using `conn`.

        Raises:
            ValueError: If no table info has a matching rule for file name `table_file`.
        """
        table_info, corpus = self._find_table_info(table_file)
        if table_info is None or corpus is None:
            raise ValueError(f'No table info matches file name "{table_file}"')
        table_name = self._create_table(conn, table_info, corpus)
        self._load_file(conn, table_name, table_file)

    def _find_table_info(self, table_file: Path) -> tuple[dict | None, str | None]:
        """Find and return table information for file `table_file`.

        Find and return the first table information item in `self._table_info` for `table_file` in which one of the file
        name regexps (filenames_re) match `table_file` and none of excluded file name regexps (exclude_filename_re)
        match.

        Args:
            table_file: The file name to find the table information for.

        Returns:
            A tuple of the table information dict and the corpus id if a matching table information item is found, or
                `(None, None)` otherwise.
        """
        for info in self._table_info:
            for regex in info["filenames_re"]:
                mo = regex.fullmatch(str(table_file))
                if mo and not any(exclude.fullmatch(str(table_file)) for exclude in info["exclude_filenames_re"]):
                    corpus = mo.groupdict().get("corpus")
                    return info, corpus
        return None, None

    def _create_table(self, conn: Connection, table_info: dict, corpus: str) -> str:
        """Create table based on `table_info` and `corpus` id and return table name.

        Args:
            conn: A SQLAlchemy Connection to use for executing the CREATE TABLE statement.
            table_info: A dict containing the table information, including the table definition and a table name
                template.
            corpus: The corpus id to replace the placeholders in the table name template.

        Returns:
            The name of the created table.
        """
        table_name = self._make_table_name(table_info, corpus)
        conn.exec_driver_sql(
            f"""CREATE TABLE IF NOT EXISTS `{table_name}` (
                {table_info["definition"]}
                );"""
        )
        return table_name

    @staticmethod
    def _make_table_name(table_info: dict, corpus: str) -> str:
        """Return table name based on `table_info` and `corpus` id.

        Take the table name from `table_info["tablename"]` and replace the possible format placeholders {corpus} and
        {CORPUS} in it with the corpus id in lower or upper case, respectively.
        """
        table_name = table_info["tablename"]
        return table_name.format(corpus=corpus.lower(), CORPUS=corpus.upper())

    def _load_file(self, conn: Connection, table_name: str, table_file: Path) -> None:
        """Load data from `table_file` to table `table_name` using `conn`."""
        escaped_file = self._escape_sql_string(str(table_file))
        conn.exec_driver_sql(f"LOAD DATA LOCAL INFILE '{escaped_file}' INTO TABLE `{table_name}` FIELDS ESCAPED BY '';")
