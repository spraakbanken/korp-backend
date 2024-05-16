"""
Default configuration file.

Settings can be overridden by placing a copy of this file in a directory named 'instance', and editing that copy.
"""

# Host and port for the WSGI server
WSGI_HOST = "0.0.0.0"
WSGI_PORT = 1234

# The absolute path to the CQP binaries
CQP_EXECUTABLE = ""
CWB_SCAN_EXECUTABLE = ""

# The absolute path to the CWB registry files
CWB_REGISTRY = ""

# The default encoding for the cqp binary
CQP_ENCODING = "UTF-8"

# Locale to use when sorting
LC_COLLATE = "sv_SE.UTF-8"

# The maximum number of search results that can be returned per query (0 = no limit)
MAX_KWIC_ROWS = 0

# Number of threads to use during parallel processing
PARALLEL_THREADS = 3

# Database host and port
DBHOST = "0.0.0.0"
DBPORT = 3306

# Database name
DBNAME = ""

# Word Picture table prefix
DBWPTABLE = "relations"

# Username and password for database access
DBUSER = ""
DBPASSWORD = ""

# Cache path (optional). Script must have read and write access.
CACHE_DIR = ""

# Disk cache lifespan in minutes
CACHE_LIFESPAN = 20

# Memcached server IP address and port, or path to socket file (socket path must start with slash)
MEMCACHED_SERVER = None

# Max number of rows from count command to cache
CACHE_MAX_STATS = 50

# Max size in bytes per cached query data file (0 = no limit)
CACHE_MAX_QUERY_DATA = 0

# Corpus configuration directory
CORPUS_CONFIG_DIR = ""

# Set to True to enable "lab mode", potentially enabling experimental features and access to lab-only corpora
LAB_MODE = False

# When checking the availability of corpora, if True, try to select
# each corpus in CQP (slower), otherwise only check the files in the
# CWB registry directory (somewhat faster)
CHECK_AVAILABLE_CORPORA_STRICTLY = True

# Plugins to load
PLUGINS = []

# Plugin configuration
PLUGINS_CONFIG = {}
