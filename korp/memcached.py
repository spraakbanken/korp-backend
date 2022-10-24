from contextlib import contextmanager
from typing import Generator

from pymemcache import serde
from pymemcache.client.base import Client


class Memcached:

    def __init__(self):
        self.server = None
        self.active = False

    def init(self, server):
        """Initialize memcached class."""
        self.server = server

        try:
            with self.get_client() as mc:
                mc.get("test_connection")
            self.active = True
        except:
            print("Could not connect to Memcached. Caching will be disabled.")

    @contextmanager
    def get_client(self) -> Generator[Client, None, None]:
        """Get a connected Memcached client."""
        client = Client(self.server, serde=serde.pickle_serde)
        try:
            yield client
        finally:
            client.close()


memcached = Memcached()
