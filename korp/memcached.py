import pylibmc


class Memcached:

    def __init__(self):
        self.pool = None

    def init(self, servers, pool_size):
        """Initialize memcached connection."""
        mc_client = pylibmc.Client(servers)
        self.pool = pylibmc.ClientPool(mc_client, pool_size or 1)
        with self.pool.reserve() as mc:
            try:
                mc.get("test_connection")
            except:
                print("Could not connect to Memcached. Caching will be disabled.")
                self.pool = None


memcached = Memcached()
