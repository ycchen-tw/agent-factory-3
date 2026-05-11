"""Reader for training data npz files saved by RolloutSaver.

Usage:
    from agent_factory_3.trainer.training_data_reader import load_batch, iter_samples

    # Load a single batch
    batch = load_batch("rollouts/batch_0000.npz")
    print(batch["num_samples"], "rollouts,", batch["total_tokens"], "tokens")

    # Iterate over individual samples
    for sample in iter_samples(batch):
        print(sample["rollout_id"], sample["reward"], sample["token_ids"].shape)
        if sample["routing_indices"] is not None:
            print("  routing:", sample["routing_indices"].shape)

    # Per-round reconstruction (using step boundaries from metadata)
    for sample in iter_samples(batch):
        for step in sample["steps"]:
            s, e = step["start"], step["end"]
            if step["type"] == "assistant":
                round_logprobs = sample["logprobs"][s:e]
                round_routing  = sample["routing_indices"][s:e]

    # Load all batches in a directory
    for batch in load_all_batches("rollouts/"):
        for sample in iter_samples(batch):
            ...

Notes:
    - routing_indices uses sentinel value 255 for positions with no routing data
      (prefix cache hits, tool output tokens). Valid expert indices are 0-254.
    - Metadata "steps" field contains per-step (type, start, end) for per-round
      reconstruction. Token indices are relative to the rollout, matching the
      sliced arrays from iter_samples().
    - To cross-reference with the JSON file, join on (group_id, rollout_id).
      The NPZ may contain fewer rollouts than the JSON (skips those without logprobs).
"""

import json
import logging
from pathlib import Path
from typing import Iterator

import numpy as np

logger = logging.getLogger(__name__)

ROUTING_SENTINEL = 255  # Positions with no routing data


def load_batch(path: str | Path) -> dict:
    """Load a batch npz file.

    Returns dict with:
        token_ids:       np.ndarray[total_T] int32
        logprobs:        np.ndarray[total_T] float16  (NaN = non-generated position)
        completion_mask:  np.ndarray[total_T] uint8
        advantages:      np.ndarray[total_T] float16
        routing_indices: np.ndarray[total_T, L, K] uint8 or None  (255 = no data)
        cu_seqlens:      np.ndarray[N+1] int32
        metadata:        list[dict]  (per-rollout, includes "steps" for per-round access)
        num_samples:     int
        total_tokens:    int
    """
    with np.load(path) as data:
        cu_seqlens = data["cu_seqlens"].copy()
        meta = json.loads(data["metadata"].tobytes().decode("utf-8"))
        routing = data["routing_indices"].copy() if "routing_indices" in data else None

        result = {
            "token_ids": data["token_ids"].copy(),
            "logprobs": data["logprobs"].copy(),
            "completion_mask": data["completion_mask"].copy(),
            "advantages": data["advantages"].copy(),
            "routing_indices": routing,
            "cu_seqlens": cu_seqlens,
            "metadata": meta,
            "num_samples": len(cu_seqlens) - 1,
            "total_tokens": int(cu_seqlens[-1]),
        }
    return result


def iter_samples(batch: dict) -> Iterator[dict]:
    """Yield per-sample dicts from a loaded batch.

    Each dict contains:
        token_ids:       np.ndarray[T_i] int32
        logprobs:        np.ndarray[T_i] float16
        completion_mask:  np.ndarray[T_i] uint8
        advantages:      np.ndarray[T_i] float16
        routing_indices: np.ndarray[T_i, L, K] uint8 or None
        steps:           list[dict]  (step boundaries for per-round slicing)
        + all other metadata fields (rollout_id, group_id, reward, advantage, ...)
    """
    cu = batch["cu_seqlens"]
    routing = batch["routing_indices"]

    for i in range(batch["num_samples"]):
        s, e = int(cu[i]), int(cu[i + 1])
        sample = {
            "token_ids": batch["token_ids"][s:e],
            "logprobs": batch["logprobs"][s:e],
            "completion_mask": batch["completion_mask"][s:e],
            "advantages": batch["advantages"][s:e],
            "routing_indices": routing[s:e] if routing is not None else None,
        }
        sample.update(batch["metadata"][i])
        yield sample


def load_all_batches(
    rollout_dir: str | Path,
) -> Iterator[dict]:
    """Iterate over all batch_*.npz files in a directory, sorted by index."""
    rollout_dir = Path(rollout_dir)
    paths = sorted(rollout_dir.glob("batch_*.npz"))
    if not paths:
        logger.warning(f"No batch_*.npz files found in {rollout_dir}")
    for path in paths:
        yield load_batch(path)
