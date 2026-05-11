"""Types for orchestrator."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from ..rollout.parallel.config import RolloutConfig, RolloutResult


@dataclass
class RolloutTask:
    """What goes into the worker input queue — one rollout."""

    config: RolloutConfig
    server_url: str
    group_id: str
    index_in_group: int


@dataclass
class GroupConfig:
    """A group of rollouts for the same problem."""

    group_id: str
    rollout_configs: List[RolloutConfig]
    metadata: Dict[str, Any] = field(default_factory=dict)
    server_affinity: bool = False  # True = 整個 group 丟同一台 server（共用 prefix cache）


@dataclass
class GroupResult:
    """Completed group with rewards.

    Orchestrator fills: group_id, results, rewards, metadata.
    SampleProcessor annotates: advantages, trainable_mask, skip_reasons,
    filter_reason, reward_baseline.
    """

    # ---- Orchestrator 填的 ----
    group_id: str
    results: List[RolloutResult]
    rewards: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ---- SampleProcessor 標註的 ----
    advantages: List[float] = field(default_factory=list)
    trainable_mask: List[bool] = field(default_factory=list)
    skip_reasons: List[Optional[str]] = field(default_factory=list)
    filter_reason: Optional[str] = None
    reward_baseline: float = 0.0


# Type alias for reward function: (all_results, group_metadata) -> rewards
# Receives the full group so reward can depend on group-level statistics
# (e.g. length penalty relative to group mean).
RewardFn = Callable[[List[RolloutResult], Dict[str, Any]], List[float]]


RewardReturn = Union[float, Tuple[float, Dict[str, float]]]


def per_rollout(fn: Callable[[RolloutResult, Dict[str, Any]], RewardReturn]) -> RewardFn:
    """Lift a per-rollout reward function to group-level RewardFn.

    The per-rollout function may return:
      - float: scalar reward only
      - (float, dict): scalar reward + named components

    When components are returned, they are stored on
    RolloutResult.reward_components for downstream metrics.
    """

    def wrapped(results: List[RolloutResult], metadata: Dict[str, Any]) -> List[float]:
        rewards = []
        for r in results:
            out = fn(r, metadata)
            if isinstance(out, tuple):
                reward, components = out
                r.reward_components = components
            else:
                reward = out
            rewards.append(reward)
        return rewards

    return wrapped
