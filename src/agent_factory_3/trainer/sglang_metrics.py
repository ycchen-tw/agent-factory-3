"""Per-server sglang metric publishing.

This module owns three responsibilities:

1. **Schema** — a single declarative table (`_FIELDS`) of which fields to
   pluck from sglang's `/v1/loads` JSON. Adding a metric is one line.
2. **Parsing** — `parse_loads_response` extracts those fields. Missing
   keys become NaN (not 0.0) so a sglang upstream rename surfaces as a
   broken-line wandb plot instead of silently flatlining at zero.
3. **Drain** — `SglangMetricsDrainer` is a background thread on rank 0
   that pulls metric snapshots off the cross-process queue and feeds
   them into wandb. Decoupling from the training loop's batch tick keeps
   the queue from falling behind under multi-server / long-batch loads.

Layering: this module is a leaf — it imports nothing from `rl_flow` or
`orchestrator`, so both can use it without circular imports.

Created: 2026-05-08
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


# ---- Field schema ------------------------------------------------------

_NAN = float("nan")


@dataclass(frozen=True)
class _Field:
    """Declarative spec: how to pluck one metric from /v1/loads payload.

    `path` is a JSON-pointer-style tuple walked inside `loads[0]`. The
    name becomes the metric key (`sglang/<name>/<sid>`).
    """
    name: str
    path: tuple[str, ...]


# Add a new sglang metric here; nothing else changes.
_FIELDS: tuple[_Field, ...] = (
    _Field("throughput",         ("gen_throughput",)),
    _Field("kv_usage",           ("token_usage",)),
    _Field("kv_used_tokens",     ("num_used_tokens",)),
    _Field("kv_cache_gb",        ("memory", "kv_cache_gb")),
    _Field("cache_hit_rate",     ("cache_hit_rate",)),
    _Field("spec_accept_length", ("speculative", "accept_length")),
    _Field("spec_accept_rate",   ("speculative", "accept_rate")),
    _Field("queue_running",      ("num_running_reqs",)),
    _Field("queue_waiting",      ("num_waiting_reqs",)),
    _Field("queue_paused",       ("queues", "paused")),
    _Field("queue_retracted",    ("queues", "retracted")),
    _Field("queue_grammar",      ("queues", "grammar")),
    _Field("utilization",        ("utilization",)),
)


def field_names() -> tuple[str, ...]:
    """Public metric short-names (used by setup_wandb_workspace.py)."""
    return tuple(f.name for f in _FIELDS)


def _walk(d: Any, path: tuple[str, ...]) -> Any:
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def parse_loads_response(data: dict) -> dict[str, float]:
    """Parse one /v1/loads JSON response into a flat metric dict.

    Returns {} if `loads[0]` is missing or not a dict (caller should
    skip enqueue and log). Missing fields inside a valid dict default
    to NaN so wandb renders them as gaps rather than silent zeros.
    """
    try:
        load = data["loads"][0]
    except (KeyError, IndexError, TypeError):
        return {}
    if not isinstance(load, dict):
        return {}
    out: dict[str, float] = {}
    for f in _FIELDS:
        v = _walk(load, f.path)
        out[f.name] = float(v) if v is not None else _NAN
    return out


def server_id_from_url(url: str) -> str:
    """Stable wandb metric suffix derived from a server URL.

    Replaces characters that interfere with wandb metric paths or regex
    matching: '.', '-', ':' (the last for IPv6 hostnames after urlparse).
    """
    p = urlparse(url)
    host = (p.hostname or "unknown")
    for bad in ".-:":
        host = host.replace(bad, "_")
    return f"{host}_{p.port or 0}"


# ---- Drainer -----------------------------------------------------------

@dataclass
class DrainerStats:
    points_logged: int = 0
    log_errors: int = 0
    final_pump: int = 0


class SglangMetricsDrainer:
    """Background thread: drains the metrics queue and forwards to wandb.

    Why a thread instead of inline drain in the training loop:
      - Sglang polls every ~10s × N servers; even at modest N a long
        batch (minutes) could let the queue grow faster than one
        batch-tick drain could keep up.
      - A thread keeps the upload cadence steady regardless of training
        pace, so wandb plots stay smooth across weight syncs and stalls.

    Step axis (canonical wandb pattern, see _init_wandb in rl_flow.py):
      - We do NOT pass step= to wandb.log. wandb auto-increments its
        global _step counter with each log call, which is always
        monotonic by construction → no dropped data, no ordering races.
      - The panel-time x axis comes from `sglang/_seq` carried inside
        the payload; `define_metric("sglang/*", step_metric="sglang/_seq")`
        in _init_wandb wires panels to use _seq for the x axis.
      - This means the drainer needs no atomic step counter and no
        coordination with the train-side log path.

    Thread-safety contract:
      - `wandb.log` is thread-safe (it forwards to wandb's worker process
        via mp.Queue).
      - `_seq` is mutated by the drainer thread during normal operation
        and by the main thread during `pump_until_empty()`. There is a
        happens-before edge: stop() joins the thread before pump_until_empty()
        runs, so the main thread sees the final value with no race.

    Lifecycle:
      drainer = SglangMetricsDrainer(q)
      drainer.start()
      ...
      drainer.stop()                 # signal + join, ~bounded by IDLE_SLEEP
      drainer.pump_until_empty()     # final straggler pump after producer dies
    """

    _IDLE_SLEEP_S: float = 1.0
    _PUMP_HARD_LIMIT: int = 100_000  # safety cap on final pump

    def __init__(self, metrics_queue: "mp.Queue") -> None:
        self._queue = metrics_queue
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._seq = 0
        self.stats = DrainerStats()

    # --- public lifecycle ---

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("SglangMetricsDrainer already started")
        self._thread = threading.Thread(
            target=self._run, name="sglang-metrics-drainer", daemon=True,
        )
        self._thread.start()
        logger.info("sglang metrics drainer started")

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the drainer to exit and wait for it to wind down."""
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            logger.warning("sglang metrics drainer did not exit in %.1fs", timeout)
        self._thread = None

    def pump_until_empty(self) -> int:
        """Synchronously drain remaining items.

        Use AFTER the orchestrator process has fully exited — only then
        can mp.Queue's feeder thread (in the producer) be guaranteed to
        have flushed its in-memory buffer to the OS pipe, so the consumer
        side actually sees every enqueued item.
        """
        n = 0
        while n < self._PUMP_HARD_LIMIT:
            if not self._drain_one():
                break
            n += 1
        self.stats.final_pump += n
        if n:
            logger.info("Final sglang drain: %d straggler points", n)
        return n

    # --- internal ---

    def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                if not self._drain_one():
                    # Queue empty: idle until something arrives or we're stopped
                    self._stop_event.wait(timeout=self._IDLE_SLEEP_S)
        except Exception:
            logger.exception("sglang metrics drainer crashed")

    def _drain_one(self) -> bool:
        """Drain at most one snapshot. Returns False iff queue is empty."""
        try:
            wallclock, sid, flat = self._queue.get_nowait()
        except queue.Empty:
            return False
        self._publish(wallclock, sid, flat)
        return True

    def _publish(self, wallclock: float, sid: str, flat: dict[str, float]) -> None:
        try:
            import wandb
        except ImportError:
            return
        payload: dict[str, float] = {
            f"sglang/{name}/{sid}": value for name, value in flat.items()
        }
        # Two panel-x-axis hints carried alongside the metrics:
        #   _wallclock — epoch seconds at poll time (absolute)
        #   _seq       — strictly-monotonic poll index, used as the panel
        #                x metric via define_metric in rl_flow._init_wandb
        payload[f"sglang/_wallclock/{sid}"] = wallclock
        payload["sglang/_seq"] = float(self._seq)
        try:
            wandb.log(payload)  # no step= — wandb auto-advances its global _step
            self.stats.points_logged += 1
        except Exception:
            logger.exception("Failed to log sglang metrics for %s", sid)
            self.stats.log_errors += 1
        self._seq += 1
