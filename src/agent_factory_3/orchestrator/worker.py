"""Worker process — dumb executor that runs rollout tasks.

Key design: sem.acquire() BEFORE queue.get().
This ensures a worker never holds more tasks than its concurrency limit.
Tasks stay in the shared queue for other workers to pick up.
"""

import asyncio
import logging
import multiprocessing as mp
import time
import traceback

from ..rollout.parallel.config import RolloutResult
from ..rollout.parallel.executor import execute_rollout
from .types import RolloutTask

logger = logging.getLogger(__name__)


def worker_main(
    input_queue: mp.Queue,
    output_queue: mp.Queue,
    max_concurrent: int,
    worker_id: int,
) -> None:
    """Worker process entry point."""
    asyncio.run(_worker_loop(input_queue, output_queue, max_concurrent, worker_id))


async def _worker_loop(
    input_queue: mp.Queue,
    output_queue: mp.Queue,
    max_concurrent: int,
    worker_id: int,
) -> None:
    sem = asyncio.Semaphore(max_concurrent)
    loop = asyncio.get_event_loop()
    pending: set[asyncio.Task] = set()

    logger.info(f"Worker {worker_id} started (concurrency={max_concurrent})")

    async def run_task(task: RolloutTask) -> None:
        try:
            try:
                result = await execute_rollout(task.config, task.server_url)
            except Exception as e:
                logger.exception(f"Worker {worker_id}: unhandled error for {task.config.rollout_id}")
                result = RolloutResult(
                    rollout_id=task.config.rollout_id,
                    result=None,
                    start_time=time.time(),
                    end_time=time.time(),
                    elapsed_time=0,
                    server_url=task.server_url,
                    success=False,
                    error=str(e),
                    traceback=traceback.format_exc(),
                )
            await loop.run_in_executor(
                None, output_queue.put, (task.group_id, task.index_in_group, result),
            )
        finally:
            sem.release()

    while True:
        # Wait for capacity BEFORE pulling from queue.
        # This ensures we never hold more tasks than max_concurrent.
        # Excess tasks stay in the shared queue for other workers.
        await sem.acquire()
        raw = await loop.run_in_executor(None, input_queue.get)
        if raw is None:  # shutdown signal
            sem.release()
            break
        task: RolloutTask = raw
        t = asyncio.create_task(run_task(task))
        pending.add(t)
        t.add_done_callback(pending.discard)

    # Drain pending tasks
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)

    logger.info(f"Worker {worker_id} shutdown complete")
