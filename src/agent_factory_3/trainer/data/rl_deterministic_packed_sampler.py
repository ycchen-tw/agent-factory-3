"""
Deterministic RL packed sampler (single-step oriented).

Intended usage:
- Upstream (`rl_flow`) decides PPO steps / minibatches and shuffles samples if desired.
- This sampler receives a dataset in that order and deterministically packs it into
  variable-sized micro-batches ("buckets") under `max_capacity`.
- Buckets are emitted as a *global plan* that is then sharded by Accelerate/torch
  DataLoaderShard across DDP/FSDP ranks (interleaving sharding).

Design goals vs `rl_packed_dataloader.py`:
- No internal randomness (no random.shuffle, epoch seed)
- No "big batches" and no sentinel markers (the trainer can step once after the
  dataloader is exhausted, and can detect "last microbatch" via `len(dataloader)`).
- Optional per-rank load de-biasing: a simple "snake" (zig-zag) ordering of bucket
  groups so rank0 isn't systematically assigned the heaviest buckets.

Notes:
- This file is standalone and does not modify existing code. Wire-up can be done later
  by swapping imports and/or updating the training loop.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import NamedTuple, Optional

from torch.utils.data import BatchSampler, Dataset


class SampleInfo(NamedTuple):
    index: int
    length: int


@dataclass(frozen=True)
class PackingStats:
    num_samples: int
    num_buckets: int
    num_groups: int
    avg_bucket_load: float
    max_bucket_load: int
    min_bucket_load: int
    utilization_rate: float

    def __repr__(self) -> str:
        return (
            f"PackingStats(samples={self.num_samples}, buckets={self.num_buckets}, "
            f"groups={self.num_groups}, utilization={self.utilization_rate:.2%})"
        )


def _get_sequence_lengths(dataset: Dataset) -> list[int]:
    """
    Get per-sample sequence lengths for packing.

    Prefer dataset.get_sequence_lengths() if available. Otherwise, fall back to reading
    samples via __getitem__ to infer length, and emit a warning (packing may be slower).
    """
    if hasattr(dataset, "get_sequence_lengths"):
        lengths = dataset.get_sequence_lengths()  # type: ignore[attr-defined]
        if not isinstance(lengths, list):
            raise TypeError(f"get_sequence_lengths() must return list[int], got {type(lengths)}")
        return [int(x) for x in lengths]

    warnings.warn(
        "Dataset has no get_sequence_lengths(); falling back to reading samples to infer lengths. "
        "Consider implementing get_sequence_lengths() for faster packing.",
        RuntimeWarning,
        stacklevel=2,
    )

    lengths: list[int] = []
    for i in range(len(dataset)):
        sample = dataset[i]
        if hasattr(sample, "length"):
            lengths.append(int(sample.length))  # type: ignore[attr-defined]
        elif isinstance(sample, dict):
            if "length" in sample:
                lengths.append(int(sample["length"]))
            elif "input_ids" in sample:
                lengths.append(len(sample["input_ids"]))
            else:
                raise AttributeError(f"Sample dict at index {i} missing 'length' and 'input_ids' keys")
        elif hasattr(sample, "input_ids"):
            lengths.append(len(sample.input_ids))  # type: ignore[attr-defined]
        else:
            raise AttributeError(f"Cannot infer length for sample at index {i}; add get_sequence_lengths()")
    return lengths


def _bucket_load(bucket: list[SampleInfo]) -> int:
    return sum(s.length for s in bucket)


class _BucketManager:
    def __init__(self, *, max_capacity: int, world_size: int):
        self.max_capacity = max_capacity
        self.world_size = world_size
        self.buckets: list[list[SampleInfo]] = []
        self.loads: list[int] = []

    def add_bucket_group(self) -> int:
        start_idx = len(self.buckets)
        self.buckets.extend([] for _ in range(self.world_size))
        self.loads.extend([0] * self.world_size)
        return start_idx

    def can_fit(self, bucket_idx: int, sample_length: int) -> bool:
        return self.loads[bucket_idx] + sample_length <= self.max_capacity

    def add_sample(self, bucket_idx: int, sample: SampleInfo) -> None:
        self.buckets[bucket_idx].append(sample)
        self.loads[bucket_idx] += sample.length


def _redistribute_to_fill_empty_buckets(manager: _BucketManager) -> None:
    """
    Ensure no empty buckets exist (required for synchronous multi-rank iteration).

    Strategy: for each empty bucket, move the shortest movable sample from the heaviest donor bucket.
    """
    empty = [i for i, b in enumerate(manager.buckets) if not b]
    if not empty:
        return

    for empty_idx in empty:
        # donors: non-empty buckets sorted by load desc
        donors = [i for i, b in enumerate(manager.buckets) if b]
        donors.sort(key=lambda i: (-manager.loads[i], i))

        placed = False
        for donor_idx in donors:
            donor_bucket = manager.buckets[donor_idx]
            if len(donor_bucket) <= 1:
                continue

            # move the shortest candidate that fits
            candidates = sorted(donor_bucket, key=lambda s: (s.length, s.index))
            for cand in candidates:
                if manager.can_fit(empty_idx, cand.length):
                    donor_bucket.remove(cand)
                    manager.loads[donor_idx] -= cand.length
                    manager.add_sample(empty_idx, cand)
                    placed = True
                    break

            if placed:
                break

        if not placed:
            raise RuntimeError(
                f"Unable to redistribute to fill empty bucket #{empty_idx}. "
                f"Consider reducing world_size or increasing max_capacity (max_capacity={manager.max_capacity})."
            )


def pack_samples_into_bucket_groups(
    samples: list[SampleInfo],
    *,
    max_capacity: int,
    world_size: int,
) -> tuple[list[list[list[SampleInfo]]], PackingStats]:
    """
    Deterministic bin-packing (best-fit decreasing) into bucket groups.

    Output:
      bucket_groups: list of groups; each group has exactly `world_size` non-empty buckets.
    """
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if max_capacity <= 0:
        raise ValueError(f"max_capacity must be positive, got {max_capacity}")
    if len(samples) < world_size:
        raise RuntimeError(f"Need at least world_size samples ({world_size}), got {len(samples)}")

    for s in samples:
        if s.length > max_capacity:
            raise ValueError(f"Sample {s.index} length {s.length} exceeds max_capacity {max_capacity}.")

    # Global sort (deterministic): descending length, ascending index.
    sorted_samples = sorted(samples, key=lambda s: (-s.length, s.index))

    manager = _BucketManager(max_capacity=max_capacity, world_size=world_size)
    manager.add_bucket_group()

    # Best-fit: place into the least-loaded bucket that can still fit; create a new group if needed.
    for s in sorted_samples:
        best_idx: int | None = None
        best_load: int | None = None

        for i, load in enumerate(manager.loads):
            if load + s.length > max_capacity:
                continue
            if best_load is None or load < best_load or (load == best_load and i < best_idx):  # type: ignore[operator]
                best_idx = i
                best_load = load

        if best_idx is None:
            manager.add_bucket_group()
            best_idx = len(manager.loads) - world_size  # first bucket in the new group

        manager.add_sample(best_idx, s)

    _redistribute_to_fill_empty_buckets(manager)

    # Order buckets by load descending for stable grouping.
    bucket_pairs = list(zip(manager.buckets, manager.loads))
    bucket_pairs.sort(key=lambda x: (-x[1], [s.index for s in x[0]]))
    ordered_buckets = [b for b, _ in bucket_pairs]
    ordered_loads = [l for _, l in bucket_pairs]

    # Group into world_size buckets per group.
    if len(ordered_buckets) % world_size != 0:
        raise RuntimeError(
            f"Internal error: num_buckets={len(ordered_buckets)} not divisible by world_size={world_size}"
        )

    bucket_groups = [
        ordered_buckets[i : i + world_size]
        for i in range(0, len(ordered_buckets), world_size)
    ]

    total_load = sum(ordered_loads)
    stats = PackingStats(
        num_samples=len(samples),
        num_buckets=len(ordered_buckets),
        num_groups=len(bucket_groups),
        avg_bucket_load=total_load / len(ordered_buckets) if ordered_buckets else 0.0,
        max_bucket_load=max(ordered_loads) if ordered_loads else 0,
        min_bucket_load=min(ordered_loads) if ordered_loads else 0,
        utilization_rate=total_load / (len(ordered_buckets) * max_capacity) if ordered_buckets else 0.0,
    )

    return bucket_groups, stats


class DeterministicPackedBatchSampler(BatchSampler):
    """
    Build a deterministic global bucket plan for one training step.

    This sampler yields a list of sample indices per bucket.
    """

    def __init__(
        self,
        *,
        dataset: Dataset,
        max_capacity: int,
        world_size: int,
        balance_ranks: bool = True,
        verbose: bool = False,
    ) -> None:
        self.dataset = dataset
        self.max_capacity = max_capacity
        self.world_size = world_size
        self.balance_ranks = balance_ranks
        self.verbose = verbose

        self._global_plan: list[list[int]] = []
        self._stats: PackingStats | None = None

        # Deterministic, step-local sampler: build the plan once at construction time.
        # (Upstream is expected to create a new dataset+sampler per PPO step.)
        self._create_plan()

    def set_epoch(self, epoch: int) -> None:
        # Kept for API compatibility; packing is deterministic and step-local.
        # No-op to avoid rebuilding the plan unexpectedly.
        return

    def get_stats(self) -> Optional[PackingStats]:
        return self._stats

    def get_rank_loads(self) -> Optional[list[int]]:
        return None

    def _create_plan(self) -> None:
        lengths = _get_sequence_lengths(self.dataset)
        if len(lengths) != len(self.dataset):
            raise ValueError(f"get_sequence_lengths() returned {len(lengths)} lengths for dataset of size {len(self.dataset)}")

        samples = [SampleInfo(index=i, length=int(lengths[i])) for i in range(len(lengths))]

        bucket_groups, stats = pack_samples_into_bucket_groups(
            samples,
            max_capacity=self.max_capacity,
            world_size=self.world_size,
        )

        plan: list[list[int]] = []
        for group_idx, group in enumerate(bucket_groups):
            # Simple zig-zag ordering so rank0 isn't always assigned the heaviest bucket.
            if self.balance_ranks and (group_idx % 2 == 1):
                group = list(reversed(group))
            for bucket in group:
                plan.append([s.index for s in bucket])

        self._global_plan = plan
        self._stats = stats

        if self.verbose:
            print(f"[DeterministicPackedBatchSampler] {self._stats}")

    def __iter__(self):
        yield from self._global_plan

    def __len__(self) -> int:
        return len(self._global_plan)
