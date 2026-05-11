"""Adapter for converting v3 GroupResult to visualizer types."""

from typing import List, Optional

from ...orchestrator.types import GroupResult
from ...rollout.parallel.config import RolloutResult
from ...rollout.types import AssistantStep, EndReason, ReactResult, StepType, ToolStep
from ..core.types import Author, Conversation, GroupData, Message, Role, RolloutData, TextPart
from .harmony import HarmonyAdapter


def _inject_tool_stats(conversation: Conversation, result: ReactResult) -> None:
    """Inject tool execution stats into tool message metadata.

    Matches ToolSteps from ReactResult to tool messages in the conversation.
    """
    tool_steps = result.get_tool_steps()
    tool_idx = 0

    for message in conversation.messages:
        if message.author.role == Role.TOOL:
            if tool_idx < len(tool_steps):
                ts = tool_steps[tool_idx]
                message.metadata["tool_stats"] = {
                    "elapsed": ts.elapsed,
                    "error": ts.error.value if ts.error else None,
                    "early_exit": ts.early_exit,
                }
            tool_idx += 1


def _inject_failure_visibility(conversation: Conversation, result: ReactResult) -> None:
    """Make failed assistant generations visible in the HTML.

    Two flavours of failure:
      1. parse_error — runner caught a harmony parse exception on the step's
         tokens; no Message exists for it. We decode the raw tokens and append
         a synthetic Message tagged with metadata.synthetic=True.
      2. shape error (no_final_channel / tool_call_in_final_channel /
         invalid_tool_call) — the message DID parse but landed in an invalid
         shape. The parsed Message is in the conversation; we tag its metadata.

    Both flavours expose the same metadata schema:
      metadata = {
          "error_kind": str,       # parse_error | no_final_channel | ...
          "error_message": str,    # exception text
          "synthetic": bool,       # true only when the message has no harmony source
          "raw_token_count": int,  # only when synthetic
      }
    """
    # Flavour 1: any AssistantStep with parse_error → synthesize a Message.
    # Independent of end_reason because parse_error is a per-step fact: rollouts
    # interrupted externally (end_reason=INTERRUPTED) can still have a failed
    # parse worth surfacing.
    parse_failed_step: Optional[AssistantStep] = next(
        (s for s in result.steps
         if s.type == StepType.ASSISTANT and getattr(s, "parse_error", None)),
        None,
    )
    if parse_failed_step is not None:
        try:
            raw_text = result.get_step_text(parse_failed_step)
        except Exception as exc:
            raw_text = f"<decode failed: {exc}>"
        conversation.messages.append(Message(
            author=Author(role=Role.ASSISTANT),
            content=[TextPart(text=raw_text)],
            channel=None,
            metadata={
                "synthetic": True,
                "error_kind": "parse_error",
                "error_message": parse_failed_step.parse_error,
                "raw_token_count": parse_failed_step.end - parse_failed_step.start,
            },
        ))
        return

    # Flavour 2: shape error — tag the last assistant message in place. Only
    # applies when the rollout itself ended in ERROR with a shape-related detail.
    if result.end_reason != EndReason.ERROR:
        return
    detail = result.end_reason_detail or ""
    error_kind = detail.removeprefix("error:") if detail.startswith("error:") else detail
    error_msg = result.errors[0] if result.errors else ""
    for msg in reversed(conversation.messages):
        if msg.author.role == Role.ASSISTANT:
            msg.metadata["error_kind"] = error_kind
            msg.metadata["error_message"] = error_msg
            msg.metadata["synthetic"] = False
            break


class GroupResultsAdapter:
    """Converts list[GroupResult] to list[GroupData] for the rollout viewer."""

    def __init__(self):
        self._harmony = HarmonyAdapter()

    def adapt(self, group_results: List[GroupResult]) -> List[GroupData]:
        return [self._adapt_group(gr) for gr in group_results]

    def _adapt_group(self, gr: GroupResult) -> GroupData:
        assert len(gr.advantages) == len(gr.results), (
            f"GroupResult not annotated by SampleProcessor: "
            f"advantages={len(gr.advantages)}, results={len(gr.results)}"
        )
        rollouts = []
        for r, reward, advantage, trainable, skip_reason in zip(
            gr.results, gr.rewards, gr.advantages,
            gr.trainable_mask, gr.skip_reasons,
        ):
            rollouts.append(self._adapt_rollout(r, reward, advantage, trainable, skip_reason))

        return GroupData(
            group_id=gr.group_id,
            rollouts=rollouts,
            filter_reason=gr.filter_reason,
            reward_baseline=gr.reward_baseline,
        )

    def _adapt_rollout(
        self,
        r: RolloutResult,
        reward: float,
        advantage: float,
        trainable: bool = True,
        skip_reason: str | None = None,
    ) -> RolloutData:
        conversation = None
        end_reason = None
        num_rounds = None
        completion_tokens = None

        weight_versions = None

        if r.result is not None:
            end_reason = r.result.end_reason.value
            num_rounds = sum(1 for s in r.result.steps if s.type == StepType.TOOL)
            completion_tokens = r.result.num_generated_tokens

            conversation = self._harmony.adapt_conversation(r.result.conversation)
            _inject_tool_stats(conversation, r.result)
            _inject_failure_visibility(conversation, r.result)

            # Collect unique weight versions in order of appearance
            seen = set()
            versions = []
            for step in r.result.steps:
                if isinstance(step, AssistantStep):
                    for ws in step.weight_segments:
                        if ws.weight_version not in seen:
                            seen.add(ws.weight_version)
                            versions.append(ws.weight_version)
            if versions:
                weight_versions = versions

        return RolloutData(
            rollout_id=r.rollout_id,
            success=r.success,
            conversation=conversation,
            weighted_reward=reward,
            reward_components=r.reward_components,
            raw_advantage=advantage,
            advantage=advantage,
            num_rounds=num_rounds,
            completion_tokens=completion_tokens,
            elapsed_time=r.elapsed_time,
            end_reason=end_reason,
            weight_versions=weight_versions,
            error=r.error,
            traceback=r.traceback,
            trainable=trainable,
            skip_reason=skip_reason,
            config_snapshot=None,
        )
