"""Server Pool for Load Balancing."""

from threading import Lock
from typing import List


class ServerPool:
    """Thread-safe pool of server URLs with least-busy load balancing."""

    def __init__(self, base_urls: List[str]):
        if not base_urls:
            raise ValueError("base_urls cannot be empty")

        self.servers = [
            {'url': url, 'active': 0}
            for url in base_urls
        ]
        self.lock = Lock()

    def acquire(self) -> str:
        """Acquire the least busy server URL."""
        with self.lock:
            server = min(self.servers, key=lambda s: s['active'])
            server['active'] += 1
            return server['url']

    def release(self, url: str) -> None:
        """Release a server back to the pool."""
        with self.lock:
            for server in self.servers:
                if server['url'] == url:
                    if server['active'] <= 0:
                        raise RuntimeError(
                            f"Release called on server with active={server['active']}: {url}. "
                            "Bug: release() called more times than acquire()."
                        )
                    server['active'] -= 1
                    return

            raise RuntimeError(
                f"Unknown server URL: {url}. "
                "Bug: releasing a URL that was never in the pool."
            )

    def get_loads(self) -> List[int]:
        with self.lock:
            return [s['active'] for s in self.servers]

    def __len__(self) -> int:
        return len(self.servers)
