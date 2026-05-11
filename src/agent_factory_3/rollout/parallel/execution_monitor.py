"""Execution Monitor for Batch Processing."""

from contextlib import contextmanager
from threading import Lock
from typing import TYPE_CHECKING, Optional

try:
    import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

if TYPE_CHECKING:
    from .server_pool import ServerPool


class ExecutionMonitor:
    """Thread-safe execution monitor with tqdm progress bar."""

    def __init__(
        self,
        total_tasks: int,
        description: str = "Processing tasks",
        enabled: bool = True,
        position: Optional[int] = None,
        server_pool: Optional["ServerPool"] = None,
    ) -> None:
        self.total_tasks = total_tasks
        self.description = description
        self.enabled = enabled and TQDM_AVAILABLE
        self.position = position
        self.server_pool = server_pool

        self.lock = Lock()
        self.active_count = 0
        self.done = 0
        self.pbar: Optional[tqdm.tqdm] = None

        if enabled and not TQDM_AVAILABLE:
            import sys
            print("Warning: tqdm not available, progress bar disabled", file=sys.stderr)

    def __enter__(self) -> "ExecutionMonitor":
        if self.enabled:
            self.pbar = tqdm.tqdm(
                total=self.total_tasks,
                desc=self.description,
                unit="task",
                dynamic_ncols=True,
                leave=False,
                smoothing=0.03,
                position=self.position,
            )
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self.pbar:
            self.pbar.close()
            self.pbar = None
        return False

    @contextmanager
    def track(self, task_id: str):
        with self.lock:
            self.active_count += 1
            self._update_postfix_locked()

        try:
            yield
        finally:
            with self.lock:
                self.active_count -= 1
                self.done += 1
                if self.pbar:
                    self.pbar.update(1)
                self._update_postfix_locked()

    def _update_postfix_locked(self) -> None:
        if not self.pbar:
            return

        postfix: dict = {"active": self.active_count}

        if self.server_pool:
            loads = self.server_pool.get_loads()
            postfix["servers"] = str(loads)

        self.pbar.set_postfix(postfix)
