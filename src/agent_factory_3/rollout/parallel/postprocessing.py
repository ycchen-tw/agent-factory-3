"""Postprocessing for rollout results."""

from typing import List

from .config import RolloutResult
from ..types import ReactResult, StepType


def compute_stats(result: RolloutResult) -> RolloutResult:
    """Compute common statistics for a rollout result.

    Populates: routing_valid, completion_tokens_count, num_rounds.
    """
    if not result.success or result.result is None:
        return result

    react_result = result.result

    assistant_steps = [
        s for s in react_result.steps if s.type == StepType.ASSISTANT
    ]

    return result.model_copy(
        update={
            "routing_valid": _validate_routing_indices(react_result),
            "completion_tokens_count": react_result.num_generated_tokens,
            "num_rounds": len(assistant_steps),
        }
    )


def compute_stats_batch(results: List[RolloutResult]) -> List[RolloutResult]:
    """Apply compute_stats to a list of results."""
    return [compute_stats(r) for r in results]


def _validate_routing_indices(react_result: ReactResult) -> bool:
    """Check if routing indices are present and in valid range [0, 127]."""
    if react_result.routing_indices is None:
        return True
    for ri in react_result.routing_indices:
        if ri is None:
            continue
        for layer in ri:
            for idx in layer:
                if idx < 0 or idx > 127:
                    return False
    return True
