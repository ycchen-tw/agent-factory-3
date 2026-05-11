"""Core types for the training pipeline."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator


class TrainingSample(BaseModel):
    """One training sample — a complete rollout trajectory ready for training.

    All sequence fields (input_ids, completion_mask, gen_logprobs, advantages)
    have the same length = full token sequence.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    sample_id: str
    input_ids: list[int]
    completion_mask: list[int]  # 1 for assistant-generated tokens, 0 otherwise
    gen_logprobs: list[float]  # logprobs from generation; 0.0 for non-generated positions
    gen_entropy: list[float] | None = None  # entropy from generation; 0.0 for non-generated positions
    advantages: list[float]  # per-token: advantage * completion_mask
    routing_indices: np.ndarray | None = None  # shape: [T, L, K], dtype=uint8

    # Weight version tracking (for staleness filtering)
    oldest_weight_version: int = 0  # min weight version across all assistant steps
    weight_version_token_counts: dict[int, int] = {}  # {version: completion_token_count}

    # Group-level stats (for loss aggregation modes)
    seq_token_count: int  # completion tokens in this rollout
    prompt_token_count: int  # total completion tokens across all rollouts of this group
    prompt_sequence_count: int  # number of rollouts in this group
    prompt_id: str  # group identifier

    # Filled by flow layer before train_step
    divisor: float | None = None

    @model_validator(mode="after")
    def _validate_lengths(self) -> "TrainingSample":
        n = len(self.input_ids)
        for name in ("completion_mask", "gen_logprobs", "advantages"):
            val = getattr(self, name)
            if len(val) != n:
                raise ValueError(
                    f"{name} has length {len(val)}, expected {n} (same as input_ids)"
                )
        if self.gen_entropy is not None and len(self.gen_entropy) != n:
            raise ValueError(
                f"gen_entropy has length {len(self.gen_entropy)}, expected {n} (same as input_ids)"
            )
        return self

    def num_loss_tokens(self) -> int:
        return sum(self.completion_mask)


@dataclass
class ProcessStats:
    """Statistics from sample processing."""

    total_groups: int = 0
    retained_groups: int = 0
    filtered_all_failed: int = 0
    filtered_all_solved: int = 0
    filtered_zero_loss: int = 0
    filtered_error: int = 0  # rollouts skipped due to EndReason.ERROR/INTERRUPTED
    total_results: int = 0
    total_samples: int = 0

    # Reward stats (computed over all rollouts, before filtering)
    reward_mean: float = 0.0
    reward_max: float = 0.0
    reward_min: float = 0.0
    solve_rate: float = 0.0  # fraction of rollouts solved (uses reward_components["solved"] when available)

    # Aggregated reward component means (empty when reward_components not used)
    reward_component_means: dict[str, float] = field(default_factory=dict)

    # End reason distribution (counts per reason, rates computed in to_metrics_dict)
    end_reason_counts: dict[str, int] = field(default_factory=dict)

    # Generation throughput (system-level)
    gen_completion_tokens_per_sec: float = 0.0
    gen_rounds_mean: float = 0.0
    gen_rounds_max: int = 0

    def to_metrics_dict(self) -> dict[str, float | int]:
        d = {
            "process/total_groups": self.total_groups,
            "process/retained_groups": self.retained_groups,
            "process/filtered_all_failed": self.filtered_all_failed,
            "process/filtered_all_solved": self.filtered_all_solved,
            "process/filtered_zero_loss": self.filtered_zero_loss,
            "process/filtered_error": self.filtered_error,
            "process/total_results": self.total_results,
            "process/total_samples": self.total_samples,
            "reward/mean": self.reward_mean,
            "reward/max": self.reward_max,
            "reward/min": self.reward_min,
            "reward/solve_rate": self.solve_rate,
            "gen/completion_tokens_per_sec": self.gen_completion_tokens_per_sec,
            "gen/rounds_mean": self.gen_rounds_mean,
            "gen/rounds_max": self.gen_rounds_max,
        }
        for key, val in self.reward_component_means.items():
            d[f"reward_components/{key}"] = val
        # End reason rates
        total = self.total_results or 1
        for reason, count in self.end_reason_counts.items():
            d[f"gen/end_reason_rate/{reason}"] = count / total
        return d
