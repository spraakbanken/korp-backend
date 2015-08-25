# -*- coding: utf-8 -*-
"""
This is the configuration file, used by the main korp.cgi script.
"""

# The absolute path to the CQP binaries
CQP_EXECUTABLE = ""
CWB_SCAN_EXECUTABLE = ""

# The absolute path to the CWB registry files
CWB_REGISTRY = ""

# The default encoding for the cqp binary
# (this can be changed by the CGI parameter 'encoding')
CQP_ENCODING = "UTF-8"

# Locale to use when sorting
LC_COLLATE = "sv_SE.UTF-8"

# The maximum number of search results that can be returned per query (0 = no limit)
MAX_KWIC_ROWS = 0

# Number of threads to use during parallel processing
PARALLEL_THREADS = 3

# The name of the MySQL database and table prefix
DBNAME = ""
DBTABLE = ""
# Username and password for database access
DBUSER = ""
DBPASSWORD = ""

# URL to authentication server
AUTH_SERVER = ""
# Secret string used when communicating with authentication server
AUTH_SECRET = ""

# A text file with names of corpora needing authentication, one per line
PROTECTED_FILE = ""

# Cache path (optional). Script must have read and write access. Cache needs to be cleared manually when corpus data is updated.
CACHE_DIR = ""

# Max number of rows from count command to cache
CACHE_MAX_STATS = 5000
