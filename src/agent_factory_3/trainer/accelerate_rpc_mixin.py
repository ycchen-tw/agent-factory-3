from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, List, Tuple

import torch
import torch.distributed as dist
from accelerate import Accelerator


# =============================================================================
# Core Data Structures
# =============================================================================

@dataclass
class Command:
    """RPC command to be broadcast across processes."""
    name: str
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]


@dataclass
class RPCMeta:
    """Metadata for RPC method configuration."""
    name: str
    reducer: str | Callable[[List[Any]], Any] | None = "rank0"
    gather: bool = True
    broadcast_args: bool = True


# =============================================================================
# Decorator
# =============================================================================

def rpc_method(
    *,
    reducer: str | Callable[[List[Any]], Any] | None = "rank0",
    gather: bool = True,
    broadcast_args: bool = True,
):
    """
    Decorator to convert instance method into RPC method.

    Args:
        reducer: Strategy to aggregate results from all ranks.
                 Options: "rank0", "mean_dict", "mean_list_dict", "concat",
                 "merge_dict", or custom callable.
        gather: Whether to gather results from all ranks.
        broadcast_args: Whether to broadcast args/kwargs to workers.

    Example:
        @rpc_method(reducer="mean_dict")
        def train_epoch(self, dataloader, steps: int) -> dict:
            ...
    """
    def decorator(fn: Callable):
        meta = RPCMeta(
            name=fn.__name__,
            reducer=reducer,
            gather=gather,
            broadcast_args=broadcast_args,
        )

        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            # Non-distributed mode: execute locally
            if not hasattr(self, "_rpc_is_distributed") or not self._rpc_is_distributed():
                return fn(self, *args, **kwargs)

            # Already inside RPC dispatch: execute local implementation
            if getattr(self, "_rpc_in_dispatch", False):
                return fn(self, *args, **kwargs)

            # RPC can only be triggered from main process
            if not self.is_main:
                raise RuntimeError(
                    f"RPC method '{meta.name}' must be called on main process"
                )

            # Delegate to mixin's generic RPC logic
            return self._rpc_invoke(meta, fn, *args, **kwargs)

        wrapper.__rpc_meta__ = meta
        return wrapper

    return decorator


# =============================================================================
# Accelerate-based RPC Mixin
# =============================================================================

class AccelerateRPCMixin:
    """
    Mixin for distributed RPC using Accelerate framework.

    Requirements:
        Subclass must have `self.accelerator: Accelerator` attribute.
    """

    accelerator: Accelerator  # type: ignore[assignment]
    _gloo_group = None  # Class-level cache for CPU communication group

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def world_size(self) -> int:
        return getattr(self.accelerator, "num_processes", 1)

    @property
    def rank(self) -> int:
        return getattr(self.accelerator, "process_index", 0)

    @property
    def is_main(self) -> bool:
        return getattr(self.accelerator, "is_main_process", True)

    def _rpc_is_distributed(self) -> bool:
        return (
            self.world_size > 1
            and dist.is_available()
            and dist.is_initialized()
        )

    def _get_or_create_gloo_group(self):
        """
        Get or create gloo group for CPU-based communication.

        Returns:
            Gloo process group, or None if not in distributed mode.

        Raises:
            RuntimeError: If gloo backend is not available.
        """
        if not self._rpc_is_distributed():
            return None

        # If main backend is already gloo, use WORLD group directly
        if dist.get_backend() == "gloo":
            return dist.group.WORLD

        # Otherwise, create a dedicated gloo CPU group
        if AccelerateRPCMixin._gloo_group is None:
            # FAIL FAST: gloo must be available
            if not dist.is_gloo_available():
                raise RuntimeError(
                    "Gloo backend is required for CPU-based RPC communication "
                    "but is not available. Please ensure PyTorch is compiled "
                    "with gloo support."
                )
            import datetime
            AccelerateRPCMixin._gloo_group = dist.new_group(
                backend="gloo", timeout=datetime.timedelta(hours=6),
            )

        return AccelerateRPCMixin._gloo_group

    # -------------------------------------------------------------------------
    # Controller Side: Invoked by @rpc_method wrapper
    # -------------------------------------------------------------------------

    def _rpc_invoke(
        self,
        meta: RPCMeta,
        fn: Callable,
        *args,
        **kwargs,
    ) -> Any:
        """
        Controller-side RPC logic:
        1. Broadcast command to all ranks
        2. Execute local implementation
        3. Gather results from all ranks (if needed)
        4. Reduce and return aggregated result
        """
        if not self._rpc_is_distributed():
            return fn(self, *args, **kwargs)

        # Prepare command
        if meta.broadcast_args:
            cmd = Command(meta.name, args, kwargs)
        else:
            cmd = Command(meta.name, (), {})

        # Step 1: Broadcast command
        self._rpc_broadcast_command(cmd)

        # Step 2: Execute local implementation
        try:
            self._rpc_in_dispatch = True
            local_result = fn(self, *args, **kwargs)
        finally:
            self._rpc_in_dispatch = False

        # Step 3: Early return if no gathering needed
        if not meta.gather:
            return local_result

        # Step 4: Gather results from all ranks
        per_rank_results = self._rpc_all_gather_object(local_result)

        # Step 5: Reduce results
        return self._rpc_reduce_result(meta, per_rank_results)

    # -------------------------------------------------------------------------
    # Worker Side: Event loop on non-main ranks
    # -------------------------------------------------------------------------

    def rpc_worker_loop(self) -> None:
        """
        Worker event loop for non-main processes.

        Usage:
            if not accelerator.is_main_process:
                trainer.rpc_worker_loop()
                return
        """
        if not self._rpc_is_distributed() or self.is_main:
            return

        while True:
            # Wait for command from controller
            cmd = self._rpc_broadcast_command(None)

            if cmd.name == "__exit__":
                break

            # Retrieve corresponding method
            method = getattr(self, cmd.name, None)
            if method is None:
                raise AttributeError(
                    f"[rank {self.rank}] Unknown RPC method: {cmd.name}"
                )

            meta: RPCMeta | None = getattr(method, "__rpc_meta__", None)

            # Execute local implementation
            try:
                self._rpc_in_dispatch = True
                local_result = method(*cmd.args, **cmd.kwargs)
            finally:
                self._rpc_in_dispatch = False

            # Synchronize with controller's all_gather
            if meta and meta.gather:
                _ = self._rpc_all_gather_object(local_result)

        self.accelerator.wait_for_everyone()

    def rpc_shutdown(self) -> None:
        """Signal all workers to exit their event loop."""
        if not self._rpc_is_distributed() or not self.is_main:
            return

        cmd = Command("__exit__", (), {})
        self._rpc_broadcast_command(cmd)
        self.accelerator.wait_for_everyone()

    # -------------------------------------------------------------------------
    # Low-level Communication Primitives
    # -------------------------------------------------------------------------

    def _rpc_broadcast_command(self, cmd: Command | None) -> Command:
        """
        Broadcast command using CPU-based gloo backend.

        This method uses gloo backend for CPU communication, ensuring workers
        do not occupy GPU resources while waiting for commands.

        Args:
            cmd: Command to broadcast (main process), or None (worker process)

        Returns:
            Original command (main process) or received command (worker process)

        Raises:
            RuntimeError: If main process provides None, or if gloo is unavailable
        """
        # Non-distributed mode: return immediately
        if not self._rpc_is_distributed():
            if cmd is None:
                raise RuntimeError("No Command to return on single-process run")
            return cmd

        # FAIL FAST: Main process must provide command
        if self.is_main and cmd is None:
            raise RuntimeError(
                "Main process must provide Command object, got None"
            )

        # Get gloo group for CPU communication
        gloo_group = self._get_or_create_gloo_group()

        # Use broadcast_object_list (auto handles pickle/unpickle)
        if self.is_main:
            object_list = [cmd]
        else:
            object_list = [None]

        dist.broadcast_object_list(
            object_list,
            src=0,
            group=gloo_group,
            device=torch.device('cpu')
        )

        result = object_list[0]

        # FAIL FAST: Ensure result is valid
        if result is None:
            raise RuntimeError("Received None Command from broadcast")

        return result

    def _rpc_all_gather_object(self, local_obj: Any) -> List[Any]:
        """Gather objects from all ranks."""
        if not self._rpc_is_distributed():
            return [local_obj]

        out: List[Any] = [None for _ in range(self.world_size)]
        dist.all_gather_object(out, local_obj)
        return out

    # -------------------------------------------------------------------------
    # Result Reduction Strategies
    # -------------------------------------------------------------------------

    def _rpc_reduce_result(self, meta: RPCMeta, per_rank_results: List[Any]) -> Any:
        """
        Aggregate results from all ranks using specified reducer.

        Built-in reducers:
            - "rank0": Return result from rank 0
            - "mean_dict": Average numeric values in dict
            - "concat"/"concat_list": Concatenate lists
            - None: Return raw list of results
            - Callable: Custom reduction function
        """
        if meta.reducer is None:
            return per_rank_results

        if callable(meta.reducer):
            return meta.reducer(per_rank_results)

        # Built-in reducers
        if meta.reducer == "rank0":
            return per_rank_results[0]

        if meta.reducer == "mean_dict":
            first = per_rank_results[0]
            if not isinstance(first, dict):
                return first

            keys = set().union(
                *[r.keys() for r in per_rank_results if isinstance(r, dict)]
            )
            merged: Dict[str, float] = {}
            for k in keys:
                vals = [
                    float(r[k])
                    for r in per_rank_results
                    if isinstance(r, dict) and isinstance(r.get(k), (int, float))
                ]
                if vals:
                    merged[k] = sum(vals) / len(vals)
            return merged

        if meta.reducer in ("concat", "concat_list"):
            merged: List[Any] = []
            for r in per_rank_results:
                if isinstance(r, list):
                    merged.extend(r)
            return merged
        
        if meta.reducer == "merge_dict":
            merged: Dict[str, Any] = {}
            for r in per_rank_results:
                merged.update(r)
            return merged

        if meta.reducer == "mean_list_dict":
            if not all(isinstance(r, list) for r in per_rank_results):
                raise TypeError(
                    f"mean_list_dict reducer expects all results to be lists, "
                    f"got types: {[type(r).__name__ for r in per_rank_results]}"
                )

            step_counts = [len(r) for r in per_rank_results]
            if len(set(step_counts)) > 1:
                raise ValueError(
                    f"mean_list_dict requires all ranks to have same number of steps, "
                    f"got step counts by rank: {step_counts}"
                )

            num_steps = step_counts[0] if step_counts else 0
            averaged: List[Dict[str, float]] = []

            for step_idx in range(num_steps):
                step_dicts = [r[step_idx] for r in per_rank_results]

                if not all(isinstance(d, dict) for d in step_dicts):
                    raise TypeError(
                        f"mean_list_dict expects all list elements to be dicts at step {step_idx}, "
                        f"got types: {[type(d).__name__ for d in step_dicts]}"
                    )

                all_keys = set().union(*[d.keys() for d in step_dicts])
                averaged_dict: Dict[str, float] = {}

                for key in all_keys:
                    values = [
                        d[key]
                        for d in step_dicts
                        if key in d and isinstance(d[key], (int, float))
                    ]
                    if values:
                        averaged_dict[key] = sum(values) / len(values)

                averaged.append(averaged_dict)

            return averaged

        raise ValueError(f"Unknown reducer: {meta.reducer}")