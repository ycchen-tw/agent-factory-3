"""Async execution backend for parallel rollouts."""

import asyncio
from typing import AsyncIterator, Awaitable, Callable, Generic, List, TypeVar

T = TypeVar("T")
Task = Callable[[], Awaitable[T]]


class AsyncBackend(Generic[T]):
    """Async backend using asyncio.Semaphore for concurrency control.

    All tasks run in the same event loop, controlled by a semaphore.
    """

    def __init__(self, max_concurrency: int = 20):
        if max_concurrency <= 0:
            raise ValueError("max_concurrency must be positive")
        self.max_concurrency = max_concurrency

    async def execute_batch(self, tasks: List[Task[T]]) -> List[T]:
        """Execute tasks using asyncio.gather with semaphore control.

        Returns results in input order.
        """
        if not tasks:
            return []

        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def run_with_semaphore(task: Task[T]) -> T:
            async with semaphore:
                return await task()

        coros = [run_with_semaphore(t) for t in tasks]
        return await asyncio.gather(*coros)

    async def execute_batch_streaming(self, tasks: List[Task[T]]) -> AsyncIterator[tuple[int, T]]:
        """完成一個就 yield 一個，不等全部完成。

        Yields (index, result) tuples.
        """
        if not tasks:
            return

        semaphore = asyncio.Semaphore(self.max_concurrency)
        pending: set[asyncio.Task] = set()

        async def run_with_semaphore(task: Task[T], idx: int) -> tuple[int, T]:
            async with semaphore:
                return idx, await task()

        for i, task in enumerate(tasks):
            t = asyncio.create_task(run_with_semaphore(task, i))
            pending.add(t)

        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for t in done:
                yield t.result()
