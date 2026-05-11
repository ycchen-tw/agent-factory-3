from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.distributed as dist

MetricWeighting = Literal["sample", "token"]


@dataclass
class MeanMetricState:
    """
    Mergeable scalar metric state: (sum, count).

    - weighting="token": mean is token-weighted (sum over tokens / token_count)
    - weighting="sample": mean is sample-weighted (sum over per-sample stats / num_samples)
    """

    sum: torch.Tensor
    count: torch.Tensor
    weighting: MetricWeighting

    def merge_(self, other: "MeanMetricState") -> None:
        if self.weighting != other.weighting:
            raise ValueError(f"Cannot merge metric states with different weighting: {self.weighting} vs {other.weighting}")
        self.sum.add_(other.sum)
        self.count.add_(other.count)

    def all_reduce_(self) -> None:
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(self.sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.count, op=dist.ReduceOp.SUM)

    def mean(self) -> torch.Tensor:
        return self.sum / self.count.clamp_min(1.0)

