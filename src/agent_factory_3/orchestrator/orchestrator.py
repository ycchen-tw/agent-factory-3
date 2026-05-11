"""Orchestrator — dispatches rollouts to workers, collects results, yields batches.

Design:
- Dispatch unit = group (all rollouts of a group enter queue atomically)
- Individual rollouts are assigned to servers independently (least-loaded)
- Load metric = real KV token usage from sglang /v1/loads + dispatch-time estimates
- Individual rollouts are pulled by workers (sem.acquire before get)
- Dispatch policy = push new group only when in_flight + group_size <= max_in_flight
"""

import asyncio
import logging
import math
import multiprocessing as mp
import queue
import time
from typing import AsyncIterator, Dict, Iterable, List, Optional

import aiohttp

from ..rollout.parallel.config import RolloutResult
from ..trainer.sglang_metrics import parse_loads_response, server_id_from_url
from .types import GroupConfig, GroupResult, RewardFn, RolloutTask, per_rollout
from .worker import worker_main

logger = logging.getLogger(__name__)


# How often the background poller hits each sglang /v1/loads endpoint.
# Used both for load-balancing (real KV token counts) and for the
# per-server wandb metrics stream. EWMA-style `in_flight_count[url] *
# est_tokens_per_rollout` compensates for staleness between polls, so a
# longer cadence keeps dispatch quality while reducing HTTP and wandb load.
_POLL_INTERVAL_S: float = 10.0


def _fmt_load(load: Dict[str, float]) -> str:
    """Format adjusted_load dict for logging: strip scheme, show integer tokens."""
    parts = []
    for url, tokens in load.items():
        host = url.split("//")[-1] if "//" in url else url
        parts.append(f"{host}: {tokens:.0f}")
    return "{" + ", ".join(parts) + "}"


class Orchestrator:
    """Multi-process rollout orchestrator.

    Architecture:
    - N worker processes, each with semaphore-limited concurrency
    - Workers pull individual tasks from shared queue (acquire sem first)
    - Per-rollout server assignment using real KV token load from sglang /v1/loads
    - Background poller refreshes server metrics every ~10s; between polls,
      dispatches add estimated token cost to prevent over-dispatching
    - Sliding window: new group dispatched only when capacity >= group_size
    - Groups complete → reward + advantage → pool → yield batch
    """

    def __init__(
        self,
        server_urls: List[str],
        num_workers: int = 8,
        worker_concurrency: int = 8,
        batch_size: int = 128,
        reward_fn: Optional[RewardFn] = None,
        metrics_queue: Optional[mp.Queue] = None,
    ):
        if not server_urls:
            raise ValueError("server_urls cannot be empty")

        self.server_urls = server_urls
        self.num_workers = num_workers
        self.worker_concurrency = worker_concurrency
        self.batch_size = batch_size
        self.reward_fn = reward_fn or per_rollout(lambda r, m: 0.0)
        self.metrics_queue = metrics_queue

    async def run(
        self,
        groups: Iterable[GroupConfig],
        stop_event: Optional[asyncio.Event] = None,
    ) -> AsyncIterator[List[GroupResult]]:
        """Run rollouts and yield batches of completed groups.

        Args:
            groups: Iterable of GroupConfig (can be infinite).
            stop_event: Set to gracefully stop accepting new groups.

        Yields:
            List[GroupResult] when signal-bearing rollouts reach batch_size,
            or when total rollouts reach 3x batch_size (overflow safety valve).
            Final flush may be smaller than batch_size.
        """
        input_queue: mp.Queue = mp.Queue()
        output_queue: mp.Queue = mp.Queue()

        workers: List[mp.Process] = []
        for i in range(self.num_workers):
            p = mp.Process(
                target=worker_main,
                args=(input_queue, output_queue, self.worker_concurrency, i),
                daemon=True,
            )
            p.start()
            workers.append(p)
            logger.info(f"Started worker {i} (pid={p.pid})")

        stopped = False
        try:
            async for batch in self._main_loop(
                groups, input_queue, output_queue, workers, stop_event,
            ):
                yield batch
                # Check if stop_event was set during yield (consumer broke out)
                if stop_event is not None and stop_event.is_set():
                    stopped = True
                    break
        finally:
            self._shutdown_workers(input_queue, workers, force=stopped)

    async def _main_loop(
        self,
        groups: Iterable[GroupConfig],
        input_queue: mp.Queue,
        output_queue: mp.Queue,
        workers: List[mp.Process],
        stop_event: Optional[asyncio.Event],
    ) -> AsyncIterator[List[GroupResult]]:
        loop = asyncio.get_event_loop()

        max_in_flight = self.num_workers * self.worker_concurrency

        # ---- Metrics-driven load balancing ----
        # adjusted_load = real KV tokens (from /v1/loads poll) + estimated tokens
        # for rollouts dispatched since the last poll.  Poll resets to:
        #   ground_truth + in_flight_count * est_tokens  (compensate in-transit).
        # Concurrency note: poller task and main loop share adjusted_load, but
        # both run on the same asyncio event loop (single-threaded), so dict
        # mutations between await points are atomic — no lock needed.
        adjusted_load: Dict[str, float] = {url: 0.0 for url in self.server_urls}
        in_flight_count: Dict[str, int] = {url: 0 for url in self.server_urls}
        est_tokens_per_rollout: float = 10000.0   # EWMA of tokens per rollout
        _EWMA_ALPHA = 0.05

        group_iter = iter(groups)
        group_tracker: Dict[str, _GroupState] = {}
        completed_pool: List[GroupResult] = []
        pool_rollout_count = 0
        signal_rollout_count = 0

        in_flight = 0
        exhausted = False
        pending_group: Optional[GroupConfig] = None  # look-ahead buffer

        # Stats
        total_dispatched_groups = 0
        total_completed_groups = 0

        def assign_server() -> str:
            """Pick the server with lowest estimated KV token load."""
            url = min(self.server_urls, key=lambda u: adjusted_load[u])
            adjusted_load[url] += est_tokens_per_rollout
            in_flight_count[url] += 1
            return url

        def dispatch_group(group: GroupConfig) -> None:
            nonlocal in_flight, total_dispatched_groups

            group_size = len(group.rollout_configs)

            group_tracker[group.group_id] = _GroupState(
                group_size=group_size, metadata=group.metadata,
            )

            server_counts: Dict[str, int] = {}

            if group.server_affinity:
                # All rollouts in this group go to the same server (prefix cache sharing)
                affinity_url = min(self.server_urls, key=lambda u: adjusted_load[u])
                adjusted_load[affinity_url] += est_tokens_per_rollout * group_size
                in_flight_count[affinity_url] += group_size
                for idx, config in enumerate(group.rollout_configs):
                    task = RolloutTask(
                        config=config,
                        server_url=affinity_url,
                        group_id=group.group_id,
                        index_in_group=idx,
                    )
                    input_queue.put(task)
                server_counts[affinity_url] = group_size
            else:
                for idx, config in enumerate(group.rollout_configs):
                    server_url = assign_server()
                    task = RolloutTask(
                        config=config,
                        server_url=server_url,
                        group_id=group.group_id,
                        index_in_group=idx,
                    )
                    input_queue.put(task)
                    server_counts[server_url] = server_counts.get(server_url, 0) + 1

            in_flight += group_size
            total_dispatched_groups += 1
            logger.debug(
                f"Dispatched group {group.group_id} ({group_size} rollouts) "
                f"spread={dict(server_counts)} [in_flight={in_flight}]"
            )

        def fill_pipeline() -> None:
            """Push groups into queue while there's capacity for a full group."""
            nonlocal pending_group, exhausted
            stopped = stop_event is not None and stop_event.is_set()
            while not exhausted and not stopped:
                # Get next group (look-ahead buffer)
                if pending_group is None:
                    pending_group = next(group_iter, None)
                    if pending_group is None:
                        exhausted = True
                        break
                # Check capacity: room for the entire group?
                next_size = len(pending_group.rollout_configs)
                if in_flight + next_size <= max_in_flight:
                    dispatch_group(pending_group)
                    pending_group = None
                else:
                    break  # no room, wait for completions

        # ---- Background metrics poller ----
        poller_stop = asyncio.Event()

        async def _metrics_poller() -> None:
            timeout = aiohttp.ClientTimeout(total=1.5)
            while not poller_stop.is_set():
                try:
                    async with aiohttp.ClientSession(timeout=timeout) as session:
                        while not poller_stop.is_set():
                            # NOTE: do NOT pass include=core — that strips
                            # memory/speculative/queues which we publish as
                            # per-server wandb metrics. The full /v1/loads
                            # payload still contains num_used_tokens for the
                            # load-balancing path below.
                            reqs = [
                                session.get(f"{url}/v1/loads")
                                for url in self.server_urls
                            ]
                            results = await asyncio.gather(
                                *reqs, return_exceptions=True,
                            )
                            for url, resp_or_exc in zip(self.server_urls, results):
                                if isinstance(resp_or_exc, BaseException):
                                    continue  # keep stale value
                                try:
                                    data = await resp_or_exc.json()
                                    real_tokens = float(data["loads"][0]["num_used_tokens"])
                                    # Reset to ground truth + estimate for
                                    # in-transit rollouts not yet seen by sglang
                                    adjusted_load[url] = (
                                        real_tokens
                                        + in_flight_count[url] * est_tokens_per_rollout
                                    )
                                    if self.metrics_queue is not None:
                                        flat = parse_loads_response(data)
                                        if flat:
                                            sid = server_id_from_url(url)
                                            try:
                                                self.metrics_queue.put_nowait(
                                                    (time.time(), sid, flat),
                                                )
                                            except queue.Full:
                                                pass  # never block load-balancing
                                        else:
                                            logger.warning(
                                                "sglang /v1/loads from %s returned "
                                                "unexpected shape (keys=%s)",
                                                url, list(data.keys())[:6],
                                            )
                                except Exception:
                                    logger.debug(
                                        "Failed to parse /v1/loads from %s", url,
                                        exc_info=True,
                                    )
                                finally:
                                    resp_or_exc.close()
                            try:
                                await asyncio.wait_for(
                                    poller_stop.wait(), timeout=_POLL_INTERVAL_S,
                                )
                                return  # stop event set
                            except asyncio.TimeoutError:
                                pass  # next poll cycle
                except Exception:
                    logger.warning(
                        "Metrics poller error, retrying in 5s", exc_info=True,
                    )
                    try:
                        await asyncio.wait_for(poller_stop.wait(), timeout=5.0)
                        return
                    except asyncio.TimeoutError:
                        pass  # retry

        poller_task = asyncio.create_task(_metrics_poller())

        # Initial fill
        fill_pipeline()

        # Main loop: collect results, refill, yield batches
        while in_flight > 0:
            # If stop requested, abandon in-flight rollouts and exit immediately.
            # Workers will be killed in _shutdown_workers via finally block.
            if stop_event is not None and stop_event.is_set():
                logger.info(
                    f"Stop event set, abandoning {in_flight} in-flight rollouts"
                )
                break

            try:
                raw = await asyncio.wait_for(
                    loop.run_in_executor(
                        None, lambda: output_queue.get(timeout=10),
                    ),
                    timeout=15,
                )
            except (asyncio.TimeoutError, queue.Empty):
                alive = sum(1 for p in workers if p.is_alive())
                if alive == 0:
                    raise RuntimeError(
                        f"All {self.num_workers} workers died. "
                        f"in_flight={in_flight}"
                    )
                logger.debug(
                    f"Waiting... in_flight={in_flight}, alive_workers={alive}, "
                    f"load={_fmt_load(adjusted_load)}"
                )
                continue

            group_id, index, result = raw
            in_flight -= 1

            # Update in-flight count (adjusted_load is corrected by poller)
            in_flight_count[result.server_url] -= 1

            # Update EWMA of tokens per rollout
            if result.num_generated_tokens > 0:
                est_tokens_per_rollout = (
                    _EWMA_ALPHA * result.num_generated_tokens
                    + (1 - _EWMA_ALPHA) * est_tokens_per_rollout
                )

            # Track group completion
            state = group_tracker[group_id]
            state.results[index] = result
            state.completed += 1

            if state.completed == state.group_size:
                group_result = self._finalize_group(group_id, state)
                completed_pool.append(group_result)
                n_rollouts = len(group_result.results)
                pool_rollout_count += n_rollouts
                if self._has_signal(group_result):
                    signal_rollout_count += n_rollouts
                del group_tracker[group_id]
                total_completed_groups += 1

                logger.info(
                    f"Group {group_id} complete "
                    f"(rewards={[f'{r:.1f}' for r in group_result.rewards]}) "
                    f"[signal={signal_rollout_count}/{self.batch_size}, "
                    f"pool={pool_rollout_count}, "
                    f"groups={total_completed_groups}/{total_dispatched_groups}, "
                    f"load={_fmt_load(adjusted_load)}]"
                )

                overflow = pool_rollout_count >= 3 * self.batch_size
                if signal_rollout_count >= self.batch_size or overflow:
                    if overflow and signal_rollout_count < self.batch_size:
                        logger.warning(
                            f"Overflow yield: {pool_rollout_count} total rollouts "
                            f"but only {signal_rollout_count} signal rollouts "
                            f"(batch_size={self.batch_size})"
                        )
                    yield list(completed_pool)
                    completed_pool.clear()
                    pool_rollout_count = 0
                    signal_rollout_count = 0

            # Refill pipeline (may dispatch if enough capacity freed)
            fill_pipeline()

        # Stop metrics poller
        poller_stop.set()
        poller_task.cancel()
        try:
            await poller_task
        except (asyncio.CancelledError, Exception):
            pass

        # Flush remaining
        if completed_pool:
            yield list(completed_pool)

        logger.info(
            f"Orchestrator done: {total_dispatched_groups} groups dispatched, "
            f"{total_completed_groups} completed, "
            f"est_tokens_per_rollout={est_tokens_per_rollout:.0f}"
        )

    @staticmethod
    def _has_signal(gr: GroupResult) -> bool:
        """Quick check: does this group have reward variance (training signal)?

        A group has no signal when all rewards are identical — either all_failed
        (all zero) or all_solved (all same non-zero). Uses raw rewards as a fast
        heuristic; SampleProcessor does the precise trainable-reward check later.
        """
        rewards = gr.rewards
        if not rewards or any(math.isnan(r) for r in rewards):
            return False
        return len(set(rewards)) > 1

    def _finalize_group(self, group_id: str, state: "_GroupState") -> GroupResult:
        results = [state.results[i] for i in range(state.group_size)]
        rewards = self.reward_fn(results, state.metadata)
        assert len(rewards) == len(results), (
            f"RewardFn returned {len(rewards)} rewards for {len(results)} results"
        )
        return GroupResult(
            group_id=group_id,
            results=results,
            rewards=rewards,
            metadata=state.metadata,
        )

    def _shutdown_workers(
        self,
        input_queue: mp.Queue,
        workers: List[mp.Process],
        force: bool = False,
    ) -> None:
        if force:
            # Force kill — workers may have in-flight work, don't wait for sentinel
            for i, p in enumerate(workers):
                if p.is_alive():
                    logger.info(f"Force-killing worker {i} (pid={p.pid})")
                    p.kill()
            for p in workers:
                p.join(timeout=5)
            return

        # Graceful: send sentinel to each worker
        for _ in workers:
            try:
                input_queue.put_nowait(None)
            except Exception:
                pass
        for i, p in enumerate(workers):
            p.join(timeout=5)
            if p.is_alive():
                logger.info(f"Killing worker {i} (pid={p.pid})")
                p.kill()
                p.join(timeout=5)


class _GroupState:
    __slots__ = ("group_size", "metadata", "results", "completed")

    def __init__(self, group_size: int, metadata: dict):
        self.group_size = group_size
        self.metadata = metadata
        self.results: Dict[int, RolloutResult] = {}
        self.completed = 0
