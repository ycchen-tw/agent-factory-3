"""SampleProcessor — converts Orchestrator output to training samples.

GroupResult (from Orchestrator) → list[TrainingSample]

Responsibilities:
1. Annotate each GroupResult (trainable_mask, advantages, skip_reasons, etc.)
2. Filter groups (all_failed, all_solved, zero_loss)
3. Build TrainingSamples from trainable rollouts only

Advantage computation lives here (not in Orchestrator) so that the
"trainable" definition and the advantage baseline are always consistent.
"""

import logging
from typing import Optional

import numpy as np

from ..orchestrator.types import GroupResult
from ..rollout.parallel.config import RolloutResult
from ..rollout.types import AssistantStep, EndReason, ReactResult, StepType
from .types import ProcessStats, TrainingSample

logger = logging.getLogger(__name__)


# Default deny-list for ERROR rollouts. Lists infra / tool-server failures that
# aren't model behavior. Model-behavior errors (parse_error, no_final_channel,
# invalid_tool_call, tool_call_in_final_channel) are trainable by default —
# they become negative samples teaching the model to avoid those failures.
DEFAULT_EXCLUDED_ERROR_KINDS: tuple[str, ...] = (
    "llm_error",            # sglang/LLM API failure — no model output
    "max_aborts_exceeded",  # weight-sync abort cascade — infra interruption
    "tool_timeout",         # MCP tool slow — server-side, not model
    "tool_exception",       # MCP tool crash — server-side, not model
)


def _extract_error_kind(detail: str | None) -> str | None:
    """Extract normalized error kind from end_reason_detail.

    Detail format: "error:<kind>[:<extra>]". Returns the <kind> segment, or
    None if detail isn't an error-shaped string. The trailing ":<extra>" is
    dropped (e.g. "max_aborts_exceeded:42" → "max_aborts_exceeded").
    """
    if not detail or not detail.startswith("error:"):
        return None
    return detail.removeprefix("error:").split(":", 1)[0]


class SampleProcessor:
    """Converts list[GroupResult] → list[TrainingSample]."""

    def __init__(
        self,
        *,
        normalize_advantages: bool = False,
        filter_all_failed: bool = True,
        filter_all_solved: bool = True,
        excluded_error_kinds: list[str] | None = None,
    ):
        self.normalize_advantages = normalize_advantages
        self.filter_all_failed = filter_all_failed
        self.filter_all_solved = filter_all_solved
        self.excluded_error_kinds: frozenset[str] = frozenset(
            DEFAULT_EXCLUDED_ERROR_KINDS if excluded_error_kinds is None
            else excluded_error_kinds
        )
        self._prev_batch_end: float | None = None

    def process(
        self,
        group_results: list[GroupResult],
    ) -> tuple[list[TrainingSample], ProcessStats]:
        """Process a batch of completed groups into training samples.

        Side effect: annotates each GroupResult in-place with advantages,
        trainable_mask, skip_reasons, filter_reason, reward_baseline.
        """
        # Compute reward stats over all rollouts (before filtering)
        all_rewards: list[float] = []
        all_results_flat: list[RolloutResult] = []
        for gr in group_results:
            all_rewards.extend(gr.rewards)
            all_results_flat.extend(gr.results)

        # Compute solve_rate: prefer reward_components["solved"] when available
        solve_count = 0
        for r, reward in zip(all_results_flat, all_rewards):
            if r.reward_components is not None and "solved" in r.reward_components:
                if r.reward_components["solved"] > 0:
                    solve_count += 1
            elif reward > 0:
                solve_count += 1
        solve_rate = solve_count / len(all_rewards) if all_rewards else 0.0

        # Aggregate reward_components means across batch
        component_sums: dict[str, float] = {}
        component_counts: dict[str, int] = {}
        for r in all_results_flat:
            if r.reward_components is not None:
                for key, val in r.reward_components.items():
                    component_sums[key] = component_sums.get(key, 0.0) + val
                    component_counts[key] = component_counts.get(key, 0) + 1
        component_means = {
            key: component_sums[key] / component_counts[key]
            for key in component_sums
        }

        # Compute generation throughput (system-level)
        all_results = [r for r in all_results_flat if r.success]
        gen_total_tokens = sum(r.num_generated_tokens for r in all_results)
        if all_results:
            batch_end = max(r.end_time for r in all_results)
            if self._prev_batch_end is not None:
                interval = batch_end - self._prev_batch_end
            else:
                interval = batch_end - min(r.start_time for r in all_results)
            gen_tps = gen_total_tokens / interval if interval > 0 else 0.0
            self._prev_batch_end = batch_end
        else:
            gen_tps = 0.0

        # Compute round stats (number of assistant steps per rollout)
        round_counts = [
            sum(1 for s in r.result.steps if s.type == StepType.ASSISTANT)
            for r in all_results if r.result is not None
        ]

        # Count end reasons (forced_final as separate category for clean stacked area)
        end_reason_counts: dict[str, int] = {}
        for r in all_results_flat:
            if r.result is not None:
                detail = r.result.end_reason_detail or ""
                if detail.endswith(":forced_final"):
                    key = "forced_final"
                else:
                    key = r.result.end_reason.value
            else:
                key = "infra_failure"
            end_reason_counts[key] = end_reason_counts.get(key, 0) + 1

        stats = ProcessStats(
            total_groups=len(group_results),
            total_results=sum(len(gr.results) for gr in group_results),
            reward_mean=sum(all_rewards) / len(all_rewards) if all_rewards else 0.0,
            reward_max=max(all_rewards) if all_rewards else 0.0,
            reward_min=min(all_rewards) if all_rewards else 0.0,
            solve_rate=solve_rate,
            reward_component_means=component_means,
            end_reason_counts=end_reason_counts,
            gen_completion_tokens_per_sec=gen_tps,
            gen_rounds_mean=sum(round_counts) / len(round_counts) if round_counts else 0.0,
            gen_rounds_max=max(round_counts) if round_counts else 0,
        )

        # 1. Annotate all groups (trainable_mask, advantages, skip_reasons, etc.)
        for gr in group_results:
            self._annotate_group(gr)


        # 2. Filter groups
        retained: list[GroupResult] = []
        for gr in group_results:
            reason = self._should_filter(gr)
            if reason is None:
                retained.append(gr)
            else:
                gr.filter_reason = reason
                if reason == "all_failed":
                    stats.filtered_all_failed += 1
                elif reason == "all_solved":
                    stats.filtered_all_solved += 1
                elif reason == "zero_loss":
                    stats.filtered_zero_loss += 1
                logger.debug(
                    f"Filtered group {gr.group_id}: {reason} "
                    f"(rewards={gr.rewards})"
                )

        # Count error-filtered rollouts across all groups
        for gr in group_results:
            stats.filtered_error += sum(
                1 for t, sr in zip(gr.trainable_mask, gr.skip_reasons)
                if not t and sr is not None and sr not in ("infra_failure", "no_logprobs")
            )

        stats.retained_groups = len(retained)

        # 3. Build samples from retained groups
        all_samples: list[TrainingSample] = []
        for gr in retained:
            group_samples = self._build_group_samples(gr)
            all_samples.extend(group_samples)

        # 4. Optionally normalize advantages across the whole batch
        if self.normalize_advantages and len(all_samples) > 1:
            self._normalize_advantages(all_samples)

        stats.total_samples = len(all_samples)
        return all_samples, stats

    # =========================================================================
    # Trainable check
    # =========================================================================

    def _is_trainable(self, result: RolloutResult) -> tuple[bool, str | None]:
        """Determine if a rollout can produce a training sample.

        INTERRUPTED is always excluded (external weight-sync abort, not model
        behavior). ERROR is excluded only when its kind matches
        self.excluded_error_kinds — model-behavior errors (e.g. parse_error)
        flow through as trainable negative samples by default.

        Returns:
            (trainable, skip_reason) — skip_reason is None when trainable.
        """
        if not result.success or result.result is None:
            return False, "infra_failure"
        react = result.result
        if react.logprobs is None:
            return False, "no_logprobs"
        if react.end_reason == EndReason.INTERRUPTED:
            return False, react.end_reason_detail or react.end_reason.value
        if react.end_reason == EndReason.ERROR:
            kind = _extract_error_kind(react.end_reason_detail)
            # Unrecognized detail shape → exclude (don't train on something we
            # can't classify; runner is expected to use "error:<kind>" format).
            if kind is None or kind in self.excluded_error_kinds:
                return False, react.end_reason_detail or react.end_reason.value
        return True, None

    # =========================================================================
    # Group annotation
    # =========================================================================

    def _annotate_group(self, gr: GroupResult) -> None:
        """Annotate a GroupResult in-place with trainable info and advantages.

        Fills: trainable_mask, skip_reasons, reward_baseline, advantages.
        """
        trainable_mask: list[bool] = []
        skip_reasons: list[str | None] = []
        for result in gr.results:
            trainable, reason = self._is_trainable(result)
            trainable_mask.append(trainable)
            skip_reasons.append(reason)

        gr.trainable_mask = trainable_mask
        gr.skip_reasons = skip_reasons

        # Compute advantage baseline from trainable rollouts only
        trainable_rewards = [
            r for r, t in zip(gr.rewards, trainable_mask) if t
        ]
        if trainable_rewards:
            baseline = sum(trainable_rewards) / len(trainable_rewards)
        else:
            baseline = 0.0
        gr.reward_baseline = baseline

        # Compute advantages: trainable → reward - baseline, non-trainable → 0.0
        gr.advantages = [
            reward - baseline if trainable else 0.0
            for reward, trainable in zip(gr.rewards, trainable_mask)
        ]

    # =========================================================================
    # Group filtering
    # =========================================================================

    def _should_filter(self, gr: GroupResult) -> str | None:
        """Check if a group should be filtered out. Returns reason or None.

        Uses gr.trainable_mask (must be annotated first).
        """
        if not any(gr.trainable_mask):
            return "all_failed"

        trainable_rewards = [
            r for r, t in zip(gr.rewards, gr.trainable_mask) if t
        ]

        if self.filter_all_solved:
            if all(r != 0.0 for r in trainable_rewards) and len(set(trainable_rewards)) == 1:
                return "all_solved"

        if self.filter_all_failed:
            if all(r == 0.0 for r in trainable_rewards):
                return "all_failed"

        # Zero loss tokens: no completion tokens across trainable rollouts
        total_completion = sum(
            r.result.num_generated_tokens
            for r, t in zip(gr.results, gr.trainable_mask)
            if t
        )
        if total_completion == 0:
            return "zero_loss"

        return None

    # =========================================================================
    # Sample building
    # =========================================================================

    def _build_group_samples(self, gr: GroupResult) -> list[TrainingSample]:
        """Convert a single GroupResult into TrainingSamples.

        Only trainable rollouts produce samples. Uses pre-computed advantages
        from _annotate_group.
        """
        samples: list[TrainingSample] = []

        # Group-level stats from trainable rollouts only
        group_completion_tokens = sum(
            r.result.num_generated_tokens
            for r, t in zip(gr.results, gr.trainable_mask)
            if t
        )
        group_size = sum(gr.trainable_mask)

        for result, reward, advantage, trainable in zip(
            gr.results, gr.rewards, gr.advantages, gr.trainable_mask
        ):
            if not trainable:
                continue

            react: ReactResult = result.result
            assert react is not None
            assert react.logprobs is not None

            sample = self._react_result_to_sample(
                react_result=react,
                rollout_id=result.rollout_id,
                group_id=gr.group_id,
                advantage=advantage,
                group_completion_tokens=group_completion_tokens,
                group_size=group_size,
            )
            if sample is not None:
                samples.append(sample)

        return samples

    def _react_result_to_sample(
        self,
        *,
        react_result: ReactResult,
        rollout_id: str,
        group_id: str,
        advantage: float,
        group_completion_tokens: int,
        group_size: int,
    ) -> TrainingSample | None:
        """Convert a single ReactResult into a TrainingSample."""
        tokens = react_result.tokens
        loss_mask = react_result.get_loss_mask()  # list[bool]

        # Build completion_mask as int
        completion_mask = [int(m) for m in loss_mask]

        # Build logprobs (full sequence, 0.0 for non-generated positions)
        logprobs = react_result.logprobs
        assert logprobs is not None
        gen_logprobs = [
            lp if lp is not None else 0.0
            for lp in logprobs
        ]

        # Build entropy (full sequence, 0.0 for non-generated positions)
        gen_entropy = None
        if react_result.entropy is not None:
            gen_entropy = [e if e is not None else 0.0 for e in react_result.entropy]

        # Per-token advantages: advantage * completion_mask
        advantages = [advantage * m for m in completion_mask]

        seq_token_count = sum(completion_mask)
        if seq_token_count == 0:
            return None

        # Collect routing indices from assistant steps
        routing_indices = self._collect_routing_indices(react_result)

        oldest_weight_version = self._extract_min_weight_version(react_result)
        wv_token_counts = self._compute_weight_version_token_counts(react_result)

        return TrainingSample(
            sample_id=rollout_id,
            input_ids=tokens,
            completion_mask=completion_mask,
            gen_logprobs=gen_logprobs,
            gen_entropy=gen_entropy,
            advantages=advantages,
            routing_indices=routing_indices,
            oldest_weight_version=oldest_weight_version,
            weight_version_token_counts=wv_token_counts,
            seq_token_count=seq_token_count,
            prompt_token_count=group_completion_tokens,
            prompt_sequence_count=group_size,
            prompt_id=group_id,
        )

    # =========================================================================
    # Helpers
    # =========================================================================

    @staticmethod
    def _collect_routing_indices(
        react_result: ReactResult,
    ) -> np.ndarray | None:
        """Collect routing indices aligned with full token sequence.

        Returns np.ndarray of shape [T, L, K] dtype=uint8, or None.
        Zero-padded positions (prompt, tool, cache-miss) don't affect
        training since they are masked out by completion_mask.
        """
        if react_result.routing_indices is None:
            return None

        # Determine L and K from first non-None entry
        num_layers = num_experts = None
        for ri in react_result.routing_indices:
            if ri is not None:
                num_layers = len(ri)
                num_experts = len(ri[0])
                break
        if num_layers is None:
            return None

        # Verify: Nones are only trailing (no non-None after a None).
        routing = react_result.routing_indices
        seen_none = False
        for ri in routing:
            if ri is None:
                seen_none = True
            elif seen_none:
                raise ValueError(
                    "routing_indices has None followed by non-None — "
                    "mid-sequence gap would corrupt hidden states for loss tokens"
                )

        # Single allocation; None positions stay zero (masked by completion_mask).
        result = np.zeros((len(routing), num_layers, num_experts), dtype=np.uint8)
        for i, ri in enumerate(routing):
            if ri is not None:
                result[i] = ri
        return result

    @staticmethod
    def _extract_min_weight_version(react_result: ReactResult) -> int:
        """Extract minimum weight version across all assistant steps.

        Each AssistantStep may have multiple WeightSegments (from abort/resume).
        Returns 0 if no numeric weight versions are found.
        """
        versions: list[int] = []
        for step in react_result.steps:
            if step.type == StepType.ASSISTANT:
                assert isinstance(step, AssistantStep)
                for ws in step.weight_segments:
                    try:
                        versions.append(int(ws.weight_version))
                    except (ValueError, TypeError):
                        pass  # "default" or non-numeric versions
        return min(versions) if versions else 0

    @staticmethod
    def _compute_weight_version_token_counts(react_result: ReactResult) -> dict[int, int]:
        """Count completion tokens per weight version from WeightSegments.

        All tokens in an AssistantStep are completion tokens, so
        segment length = completion token count for that segment.
        Non-numeric versions (e.g., "default") are skipped, consistent
        with _extract_min_weight_version.
        """
        counts: dict[int, int] = {}
        for step in react_result.steps:
            if step.type == StepType.ASSISTANT:
                assert isinstance(step, AssistantStep)
                for ws in step.weight_segments:
                    try:
                        v = int(ws.weight_version)
                    except (ValueError, TypeError):
                        continue
                    counts[v] = counts.get(v, 0) + (ws.end - ws.start)
        return counts

    @staticmethod
    def _normalize_advantages(samples: list[TrainingSample]) -> None:
        """Normalize advantages across all samples by std."""
        # Collect all non-zero advantages (on completion tokens)
        raw_advantages = []
        for s in samples:
            for a, m in zip(s.advantages, s.completion_mask):
                if m:
                    raw_advantages.append(a)

        if len(raw_advantages) < 2:
            return

        mean = sum(raw_advantages) / len(raw_advantages)
        variance = sum((a - mean) ** 2 for a in raw_advantages) / len(raw_advantages)
        std = variance ** 0.5
        if std < 1e-8:
            return

        for s in samples:
            s.advantages = [a / std for a in s.advantages]
