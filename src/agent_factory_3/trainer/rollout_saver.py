"""RolloutSaver — saves rollout results to disk for analysis and debugging.

Runs in the orchestrator process. Saves per-batch:
- JSON: group metadata, per-rollout summary, full conversation messages
- HTML: interactive rollout viewer (self-contained, openable in browser)
- NPZ: packed training data (token_ids, logprobs, routing_indices, etc.)
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

from ..orchestrator.types import GroupResult
from ..rollout.parallel.config import RolloutResult
from ..rollout.types import ReactResult, StepType

logger = logging.getLogger(__name__)


class RolloutSaver:
    """Saves rollout results to disk (JSON + HTML)."""

    def __init__(self, output_dir: str | Path, save_routing_indices: bool = True):
        self.output_dir = Path(output_dir)
        self.save_routing_indices = save_routing_indices
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_batch(
        self,
        group_results: list[GroupResult],
        batch_index: int,
    ) -> tuple[Path, Path | None]:
        """Save a batch of group results to JSON + HTML.

        Returns:
            (json_path, html_path) — html_path is None if HTML generation failed.
        """
        # JSON (structured data for programmatic analysis)
        json_path = self.output_dir / f"batch_{batch_index:04d}.json"
        data = {
            "batch_index": batch_index,
            "timestamp": time.time(),
            "num_groups": len(group_results),
            "num_rollouts": sum(len(gr.results) for gr in group_results),
            "groups": [_serialize_group(gr) for gr in group_results],
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # HTML (interactive viewer)
        html_path = self.output_dir / f"batch_{batch_index:04d}.html"
        try:
            from ..visualizer import create_groups_viewer
            html = create_groups_viewer(
                group_results,
                title=f"Batch {batch_index}",
            )
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html)
        except Exception:
            logger.exception(f"Failed to generate HTML for batch {batch_index}")
            html_path = None

        logger.info(
            f"Saved batch {batch_index} ({len(group_results)} groups) "
            f"→ {json_path}, {html_path}"
        )
        return json_path, html_path

    def save_training_data(
        self,
        group_results: list[GroupResult],
        batch_index: int,
    ) -> Path | None:
        """Save packed training data (token_ids, logprobs, routing, etc.) to npz.

        Saves all rollouts that have a ReactResult with logprobs (not just
        trainable ones), so filtered groups are still available for analysis.

        Returns:
            npz_path, or None if no valid rollouts.
        """
        all_token_ids: list[np.ndarray] = []
        all_logprobs: list[np.ndarray] = []
        all_masks: list[np.ndarray] = []
        all_advantages: list[np.ndarray] = []
        all_routing: list[np.ndarray | int] = []  # int = deferred zero-fill length
        has_routing = False
        offsets: list[int] = [0]
        meta_list: list[dict] = []

        for gr in group_results:
            for i, r in enumerate(gr.results):
                if r.result is None or r.result.logprobs is None:
                    continue

                react = r.result
                T = len(react.tokens)

                # Validate sequence alignment
                if len(react.logprobs) != T:
                    logger.warning(
                        f"Skipping rollout {r.rollout_id}: "
                        f"logprobs length {len(react.logprobs)} != tokens length {T}"
                    )
                    continue

                # Token IDs
                tids = np.array(react.tokens, dtype=np.int32)

                # Logprobs (None → NaN)
                lps = np.array(
                    [x if x is not None else np.nan for x in react.logprobs],
                    dtype=np.float16,
                )

                # Completion mask (assistant tokens = 1)
                mask = np.array(react.get_loss_mask(), dtype=np.uint8)

                # Per-token advantages
                advantage = gr.advantages[i] if i < len(gr.advantages) else 0.0
                advs = np.array(
                    [advantage * m for m in mask],
                    dtype=np.float16,
                )

                # Routing indices [T, L, K] uint8, sentinel 255 for no-data positions
                if self.save_routing_indices:
                    routing = _collect_routing_array(react)
                    if routing is not None:
                        has_routing = True
                        all_routing.append(routing)
                    else:
                        # Placeholder; will be replaced by sentinel-filled if other rollouts have routing
                        all_routing.append(T)  # store length for deferred fill

                all_token_ids.append(tids)
                all_logprobs.append(lps)
                all_masks.append(mask)
                all_advantages.append(advs)
                offsets.append(offsets[-1] + T)

                # Per-rollout metadata (including step boundaries for per-round reconstruction)
                trainable = gr.trainable_mask[i] if i < len(gr.trainable_mask) else False
                weight_version = _extract_weight_version(react)
                meta_list.append({
                    "rollout_id": r.rollout_id,
                    "group_id": gr.group_id,
                    "reward": gr.rewards[i] if i < len(gr.rewards) else 0.0,
                    "advantage": advantage,
                    "trainable": trainable,
                    "weight_version": weight_version,
                    "end_reason": react.end_reason.value,
                    "filter_reason": gr.filter_reason,
                    "seq_len": T,
                    "steps": _serialize_step_boundaries(react),
                })

        if not all_token_ids:
            return None

        # Pack arrays
        arrays: dict[str, np.ndarray] = {
            "token_ids": np.concatenate(all_token_ids),
            "logprobs": np.concatenate(all_logprobs),
            "completion_mask": np.concatenate(all_masks),
            "advantages": np.concatenate(all_advantages),
            "cu_seqlens": np.array(offsets, dtype=np.int32),
        }

        # Pack routing (resolve deferred sentinel-fills)
        if has_routing:
            # Determine L, K from the first real routing array
            L = K = 0
            for rt in all_routing:
                if isinstance(rt, np.ndarray):
                    L, K = rt.shape[1], rt.shape[2]
                    break
            resolved: list[np.ndarray] = []
            for rt in all_routing:
                if isinstance(rt, np.ndarray):
                    resolved.append(rt)
                else:
                    # rt is an int (T) — fill with sentinel 255 (no routing data)
                    resolved.append(np.full((rt, L, K), _ROUTING_SENTINEL, dtype=np.uint8))
            arrays["routing_indices"] = np.concatenate(resolved)

        # Metadata as JSON bytes
        meta_bytes = json.dumps(meta_list, ensure_ascii=False).encode("utf-8")
        arrays["metadata"] = np.frombuffer(meta_bytes, dtype=np.uint8).copy()

        npz_path = self.output_dir / f"batch_{batch_index:04d}.npz"
        np.savez_compressed(npz_path, **arrays)

        logger.info(
            f"Saved training data batch {batch_index} "
            f"({len(meta_list)} rollouts, {offsets[-1]} tokens) → {npz_path}"
        )
        return npz_path


_ROUTING_SENTINEL = 255  # Marks positions with no routing data (matches ReactResult convention)


def _collect_routing_array(react: ReactResult) -> np.ndarray | None:
    """Collect routing indices from ReactResult into [T, L, K] uint8 array.

    Returns None if routing_indices is absent or all-None.
    Positions where routing is None (prefix cache, tool turns) are filled with
    sentinel value 255 to distinguish from valid expert index 0.
    """
    if react.routing_indices is None:
        return None

    # Find L, K from first non-None entry
    sample = next((x for x in react.routing_indices if x is not None), None)
    if sample is None:
        return None

    num_layers = len(sample)
    num_experts = len(sample[0])
    T = len(react.routing_indices)

    result = np.full((T, num_layers, num_experts), _ROUTING_SENTINEL, dtype=np.uint8)
    for i, ri in enumerate(react.routing_indices):
        if ri is not None:
            result[i] = ri
    return result


def _serialize_step_boundaries(react: ReactResult) -> list[dict]:
    """Extract step boundaries for per-round reconstruction.

    Each entry: {"type": "init"|"assistant"|"tool", "start": int, "end": int}
    start/end are token indices into the rollout's token sequence.
    """
    return [
        {"type": step.type.value, "start": step.start, "end": step.end}
        for step in react.steps
    ]


def _extract_weight_version(react: ReactResult) -> int:
    """Extract minimum weight version across assistant steps. Returns 0 if unknown."""
    versions: list[int] = []
    for step in react.steps:
        if step.type == StepType.ASSISTANT:
            for ws in step.weight_segments:
                try:
                    versions.append(int(ws.weight_version))
                except (ValueError, TypeError):
                    pass
    return min(versions) if versions else 0


def _serialize_group(gr: GroupResult) -> dict:
    assert len(gr.advantages) == len(gr.results), (
        f"GroupResult not annotated by SampleProcessor: "
        f"advantages={len(gr.advantages)}, results={len(gr.results)}"
    )
    return {
        "group_id": gr.group_id,
        "metadata": gr.metadata,
        "rewards": gr.rewards,
        "advantages": gr.advantages,
        "trainable_mask": gr.trainable_mask,
        "skip_reasons": gr.skip_reasons,
        "reward_baseline": gr.reward_baseline,
        "filter_reason": gr.filter_reason,
        "results": [
            _serialize_rollout(r, reward, adv, trainable, skip_reason)
            for r, reward, adv, trainable, skip_reason in zip(
                gr.results, gr.rewards, gr.advantages,
                gr.trainable_mask, gr.skip_reasons,
            )
        ],
    }


def _serialize_rollout(
    r: RolloutResult,
    reward: float,
    advantage: float,
    trainable: bool,
    skip_reason: str | None,
) -> dict:
    data = {
        "rollout_id": r.rollout_id,
        "success": r.success,
        "error": r.error,
        "elapsed_time": r.elapsed_time,
        "server_url": r.server_url,
        "reward": reward,
        "reward_components": r.reward_components,
        "advantage": advantage,
        "trainable": trainable,
        "skip_reason": skip_reason,
    }

    if r.result is not None:
        data["result"] = _serialize_react_result(r.result)
    else:
        data["result"] = None

    return data


def _serialize_react_result(result: ReactResult) -> dict:
    num_rounds = sum(1 for s in result.steps if s.type == StepType.TOOL)
    return {
        "end_reason": result.end_reason.value,
        "end_reason_detail": result.end_reason_detail,
        "num_generated_tokens": result.num_generated_tokens,
        "num_rounds": num_rounds,
        "total_tool_time": result.total_tool_time,
        "num_tokens": len(result.tokens),
        "errors": result.errors,
        "abort_records": [
            {
                "round_index": ar.round_index,
                "weight_version": ar.weight_version,
                "partial_token_count": ar.partial_token_count,
            }
            for ar in result.abort_records
        ],
        "steps": [_serialize_step(s) for s in result.steps],
        "conversation": result.conversation.to_dict(),
    }


def _serialize_step(step) -> dict:
    data = {
        "type": step.type.value,
        "start": step.start,
        "end": step.end,
        "length": step.length,
        "round_index": step.round_index,
    }

    if step.type == StepType.ASSISTANT:
        data["stop_reason"] = step.stop_reason
        data["recipient"] = step.recipient
        if step.weight_segments:
            data["weight_segments"] = [
                {"start": ws.start, "end": ws.end, "weight_version": ws.weight_version}
                for ws in step.weight_segments
            ]
        if step.usage:
            data["usage"] = step.usage
        if step.parse_error:
            data["parse_error"] = step.parse_error

    elif step.type == StepType.TOOL:
        data["tool_name"] = step.tool_name
        data["tool_input"] = step.tool_input
        data["tool_output"] = step.tool_output
        data["early_exit"] = step.early_exit
        if step.error:
            data["error"] = step.error.value

    return data
