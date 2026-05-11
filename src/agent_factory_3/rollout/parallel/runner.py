"""Parallel Runner for Batch Rollout Execution."""

import asyncio
from typing import AsyncIterator, Awaitable, Callable, List, Optional

from .backend import AsyncBackend
from .config import RolloutConfig, RolloutResult
from .execution_monitor import ExecutionMonitor
from .executor import execute_rollout
from .server_pool import ServerPool


class ParallelRunner:
    """Parallel rollout executor.

    Features:
    - Automatic server selection via ServerPool (least-busy first)
    - Real-time progress tracking via tqdm
    - Sync/async API + streaming API
    """

    def __init__(
        self,
        server_urls: List[str],
        max_concurrency: int = 20,
        show_progress: bool = True,
    ):
        if not server_urls:
            raise ValueError("server_urls cannot be empty")

        self.server_pool = ServerPool(server_urls)
        self.show_progress = show_progress
        self._backend = AsyncBackend(max_concurrency)

    def run(
        self,
        configs: List[RolloutConfig],
        stop_event: Optional[asyncio.Event] = None,
    ) -> List[RolloutResult]:
        """Execute rollouts synchronously."""
        return asyncio.run(self.run_async(configs, stop_event))

    async def run_async(
        self,
        configs: List[RolloutConfig],
        stop_event: Optional[asyncio.Event] = None,
    ) -> List[RolloutResult]:
        """Execute rollouts asynchronously. Returns results in input order."""
        if not configs:
            return []

        with ExecutionMonitor(
            total_tasks=len(configs),
            description="Rollouts",
            enabled=self.show_progress,
            server_pool=self.server_pool,
        ) as monitor:
            tasks = [
                self._make_task(config, stop_event, monitor) for config in configs
            ]
            results = await self._backend.execute_batch(tasks)

        return results

    async def run_async_streaming(
        self,
        configs: List[RolloutConfig],
        stop_event: Optional[asyncio.Event] = None,
    ) -> AsyncIterator[tuple[int, RolloutResult]]:
        """Execute rollouts, yielding (index, result) as each completes."""
        if not configs:
            return

        with ExecutionMonitor(
            total_tasks=len(configs),
            description="Rollouts",
            enabled=self.show_progress,
            server_pool=self.server_pool,
        ) as monitor:
            tasks = [
                self._make_task(config, stop_event, monitor) for config in configs
            ]
            async for idx, result in self._backend.execute_batch_streaming(tasks):
                yield idx, result

    def _make_task(
        self,
        config: RolloutConfig,
        stop_event: Optional[asyncio.Event],
        monitor: ExecutionMonitor,
    ) -> Callable[[], Awaitable[RolloutResult]]:
        async def task() -> RolloutResult:
            server_url = self.server_pool.acquire()
            try:
                with monitor.track(config.rollout_id):
                    return await execute_rollout(config, server_url, stop_event)
            finally:
                self.server_pool.release(server_url)

        return task

    def __repr__(self) -> str:
        return (
            f"ParallelRunner("
            f"servers={len(self.server_pool)}, "
            f"concurrency={self._backend.max_concurrency})"
        )
