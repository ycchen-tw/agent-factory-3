"""Unified ReAct Runner - 訓練與推理統一架構"""

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from openai_harmony import (
    Author,
    Conversation,
    HarmonyEncodingName,
    Message,
    RenderConversationConfig,
    Role,
    load_harmony_encoding,
)

from .config import ExtraRoundConfig, LoopConfig, RecordConfig, SamplingParams
from .llm_backend import LLMBackend
from .mcp_executor import McpExecutor, ToolCallReport, ToolErrorType
from .types import (
    AbortRecord,
    AssistantStep,
    EndReason,
    InitStep,
    ReactResult,
    SegmentMeta,
    Step,
    StepType,
    TokenBudget,
    ToolStep,
    TruncationReason,
    WeightSegment,
)

logger = logging.getLogger(__name__)

# TruncationReason -> end_reason_detail 映射
_LIMIT_DETAIL = {
    TruncationReason.ROUND_LIMIT: "token_limit:round",
    TruncationReason.CONTEXT_SPACE: "token_limit:context",
    TruncationReason.GENERATION_QUOTA: "token_limit:total",
}

# =============================================================================
# Segment-aware temperature constants
# =============================================================================

_TOKEN_MESSAGE = 200008   # <|message|>
_TOKEN_END = 200007       # <|end|>
_TOKEN_CALL = 200012      # <|call|>
_TOKEN_RETURN = 200002    # <|return|>

_CHANNEL_RE = re.compile(r'<\|channel\|>(\w+)')
_RECIPIENT_RE = re.compile(r'<\|channel\|>\w+\s+to=([^\s<]+)')


# =============================================================================
# CompletionResult — typed return from _generate()
# =============================================================================

@dataclass
class CompletionResult:
    """_generate() 的 typed return，取代 Dict[str, Any]"""
    token_ids: List[int]
    finish_reason: Optional[str]
    logprobs: Optional[List[float]] = None
    top_logprobs: Optional[List[Dict[int, float]]] = None
    routing_indices: Optional[Any] = None  # np.ndarray[uint8] shape [T, L, K] or None
    cached_tokens: Optional[int] = None
    usage: Optional[Dict[str, Any]] = None
    entropy: Optional[List[float]] = None
    segments: List[SegmentMeta] = field(default_factory=list)
    weight_version: Optional[str] = None


class UnifiedReactRunner:
    """Stateless ReAct runner - 訓練/推理統一

    每次 run() 接收 Conversation，行為可預測、易測試。
    """

    def __init__(
        self,
        config: LoopConfig,
        llm_backend: LLMBackend,
        mcp_executor: McpExecutor,
    ):
        self.config = config
        self.llm_backend = llm_backend
        self.mcp_executor = mcp_executor
        self.encoding = self._load_encoding(config.harmony_custom_config_path)

    @staticmethod
    def _load_encoding(custom_config_path: Optional[str]):
        if custom_config_path is None:
            return load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

        from openai_harmony import load_custom_harmony_encoding
        return load_custom_harmony_encoding(custom_config_path)

    async def run(
        self,
        conversation: Conversation,
        *,
        initial_tokens: Optional[List[int]] = None,
        stop_event: Optional[asyncio.Event] = None,
        preserve_on_stop: bool = False,
        record_config: Optional[RecordConfig] = None,
        rollout_id: Optional[str] = None,
        force_final: bool = False,
    ) -> ReactResult:
        """執行 ReAct loop"""
        rid = f"[{(rollout_id or 'default'):14s}]"
        max_rounds = self.config.max_rounds

        # =====================================================================
        # 初始化狀態
        # =====================================================================
        messages: List[Message] = list(conversation.messages)

        if initial_tokens is not None:
            accumulated_tokens = list(initial_tokens)
        else:
            accumulated_tokens = self.encoding.render_conversation_for_completion(
                conversation, Role.ASSISTANT,
                config=RenderConversationConfig(auto_drop_analysis=self.config.auto_drop_analysis),
            )
        force_final_tokens: List[int] = []
        if force_final:
            force_final_tokens = self.encoding.encode('<|channel|>final', allowed_special='all')
            accumulated_tokens += force_final_tokens

        # Logprobs tracking
        accumulated_logprobs: Optional[List[Optional[float]]] = None
        if record_config is not None and record_config.logprobs:
            accumulated_logprobs = [None] * len(accumulated_tokens)

        accumulated_top_logprobs: Optional[List[Optional[Dict[int, float]]]] = None
        if record_config is not None and record_config.top_logprobs > 0:
            accumulated_top_logprobs = [None] * len(accumulated_tokens)

        accumulated_entropy: Optional[List[Optional[float]]] = None
        if record_config is not None and record_config.entropy:
            accumulated_entropy = [None] * len(accumulated_tokens)

        # Routing tracking
        accumulated_routing: Optional[List[Optional[List[List[int]]]]] = None
        if record_config is not None and record_config.routing_indices:
            accumulated_routing = [None] * len(accumulated_tokens)

        # Steps
        steps: List[Step] = []
        init_step = InitStep(
            start=0,
            end=len(accumulated_tokens),
            message_start=0,
            message_end=len(messages),
            created_at=time.time(),
        )
        steps.append(init_step)

        # Loop 狀態
        round_index = 0
        total_tool_time = 0.0
        total_generated_tokens = 0
        errors: List[str] = []
        end_reason: Optional[EndReason] = None
        end_reason_detail: Optional[str] = None
        parse_retry_count = 0

        # Round-level 累積狀態（跨 abort 保持，round 完成後 reset）
        round_token_ids: List[int] = []
        round_weight_segments: List[WeightSegment] = []
        round_start_offset: Optional[int] = None  # accumulated_tokens 中此 round 的起始位置

        # Abort tracking
        abort_records: List[AbortRecord] = []
        total_abort_count = 0

        logger.info(f"{rid} START  | max_rounds={max_rounds}")

        # =====================================================================
        # Main Loop
        # =====================================================================
        while round_index < max_rounds and end_reason is None:
            # Check external stop
            if stop_event is not None and stop_event.is_set():
                end_reason = EndReason.INTERRUPTED
                end_reason_detail = "interrupted:external_stop"
                logger.info(f"{rid} STOP   | external stop signal received")
                break

            # Check token budget
            budget = self._calculate_token_budget(
                len(accumulated_tokens), total_generated_tokens
            )
            if not budget.can_generate:
                end_reason = EndReason.TOKEN_LIMIT
                end_reason_detail = _LIMIT_DETAIL[budget.limiting_factor]
                break

            # Generate
            step_start_time = time.time()
            try:
                completion = await self._generate(
                    accumulated_tokens,
                    budget.max_tokens,
                    stop_event,
                    record_config,
                )
            except Exception as exc:
                logger.exception(f"{rid} LLM error: {exc}")
                end_reason = EndReason.ERROR
                end_reason_detail = "error:llm_error"
                errors.append(str(exc))
                break

            # Abort handling — keep partial tokens, retry on resume
            if completion.finish_reason == "abort":
                if round_start_offset is None:
                    round_start_offset = len(accumulated_tokens)
                self._on_abort(
                    completion, accumulated_tokens, round_token_ids,
                    round_weight_segments, accumulated_logprobs,
                    accumulated_top_logprobs, accumulated_entropy,
                    accumulated_routing, abort_records, round_index,
                )
                total_generated_tokens += len(completion.token_ids)
                total_abort_count += 1
                if total_abort_count >= self.config.max_aborts:
                    end_reason = EndReason.ERROR
                    end_reason_detail = f"error:max_aborts_exceeded:{total_abort_count}"
                    errors.append(
                        f"Abort count {total_abort_count} exceeded max_aborts "
                        f"{self.config.max_aborts}"
                    )
                    break
                logger.info(
                    f"{rid} ABORT  | R:{round_index:2d} | "
                    f"partial={len(completion.token_ids):5d} | "
                    f"ver={completion.weight_version} | "
                    f"aborts={total_abort_count}"
                )
                # Don't increment round_index, don't parse.
                # Next generate() will block at paused server until resume.
                continue

            # Check if stopped during generation
            stopped = stop_event is not None and stop_event.is_set()
            if stopped and not preserve_on_stop:
                end_reason = EndReason.INTERRUPTED
                end_reason_detail = "interrupted:external_stop"
                break

            # Update accumulated tokens + round tracking
            if round_start_offset is not None:
                assistant_start = round_start_offset
            else:
                assistant_start = len(accumulated_tokens)
            seg_start = len(round_token_ids)

            accumulated_tokens += completion.token_ids
            total_generated_tokens += len(completion.token_ids)
            round_token_ids += completion.token_ids
            round_weight_segments.append(WeightSegment(
                start=seg_start,
                end=seg_start + len(completion.token_ids),
                weight_version=completion.weight_version or "unknown",
            ))

            # Update logprobs
            if accumulated_logprobs is not None:
                if completion.logprobs is not None:
                    accumulated_logprobs.extend(completion.logprobs)
                else:
                    accumulated_logprobs.extend([None] * len(completion.token_ids))

            if accumulated_top_logprobs is not None:
                if completion.top_logprobs is not None:
                    accumulated_top_logprobs.extend(completion.top_logprobs)
                else:
                    accumulated_top_logprobs.extend([None] * len(completion.token_ids))

            if accumulated_entropy is not None:
                if completion.entropy is not None:
                    accumulated_entropy.extend(completion.entropy)
                else:
                    accumulated_entropy.extend([None] * len(completion.token_ids))

            # Update routing — sglang returns routing for positions [0, N-1)
            # where N = prompt + completion (including prefix-cached tokens).
            # Last generated token has no routing.
            # Prefill positions overwrite previous values, so in multi-round
            # rollouts the final routing reflects the last generate call's
            # forward pass for all prompt positions.
            if accumulated_routing is not None and completion.routing_indices is not None:
                prompt_len = len(accumulated_tokens) - len(completion.token_ids)
                routing = completion.routing_indices
                # Fill prefill positions
                for i in range(min(len(routing), prompt_len)):
                    accumulated_routing[i] = routing[i]
                # Append completion positions + pad None for uncovered tail
                if len(routing) > prompt_len:
                    accumulated_routing.extend(routing[prompt_len:])
                    remaining = len(completion.token_ids) - (len(routing) - prompt_len)
                    accumulated_routing.extend([None] * remaining)
                else:
                    accumulated_routing.extend([None] * len(completion.token_ids))
            elif accumulated_routing is not None:
                accumulated_routing.extend([None] * len(completion.token_ids))

            # Parse response — use round_token_ids (covers abort partial + final)
            # force_final: prepend <|channel|>final for first round so parser sees channel
            parse_tokens = round_token_ids
            if force_final_tokens and round_index == 0:
                parse_tokens = force_final_tokens + round_token_ids
            try:
                parsed_messages = self.encoding.parse_messages_from_completion_tokens(
                    parse_tokens,
                    role=Role.ASSISTANT,
                )
            except Exception as exc:
                if not stopped and parse_retry_count < self.config.max_parse_retries:
                    accumulated_tokens = accumulated_tokens[:assistant_start]
                    if accumulated_logprobs is not None:
                        accumulated_logprobs = accumulated_logprobs[:assistant_start]
                    if accumulated_top_logprobs is not None:
                        accumulated_top_logprobs = accumulated_top_logprobs[:assistant_start]
                    if accumulated_entropy is not None:
                        accumulated_entropy = accumulated_entropy[:assistant_start]
                    if accumulated_routing is not None:
                        accumulated_routing = accumulated_routing[:assistant_start]
                    total_generated_tokens -= len(round_token_ids)
                    round_token_ids = []
                    round_weight_segments = []
                    round_start_offset = None
                    parse_retry_count += 1
                    logger.warning(
                        f"{rid} R:{round_index} Parse error, retrying "
                        f"({parse_retry_count}/{self.config.max_parse_retries}): {exc}"
                    )
                    continue

                steps.append(self._make_assistant_step(
                    start=assistant_start,
                    end=len(accumulated_tokens),
                    message_start=len(messages),
                    message_end=len(messages),
                    round_index=round_index,
                    step_start_time=step_start_time,
                    completion=completion,
                    budget=budget,
                    weight_segments=list(round_weight_segments),
                    parse_error=str(exc),
                ))

                if stopped:
                    end_reason = EndReason.INTERRUPTED
                    end_reason_detail = "interrupted:external_stop"
                    errors.append(f"Parse failed during stop: {exc}")
                else:
                    logger.info(f"{rid} Parse error: {exc}")
                    end_reason = EndReason.ERROR
                    end_reason_detail = "error:parse_error"
                    errors.append(str(exc))
                break

            # Track message indices
            assistant_message_start = len(messages)
            messages.extend(parsed_messages)
            assistant_message_end = len(messages)
            last_message = parsed_messages[-1]

            if stopped:
                end_reason = EndReason.INTERRUPTED
                end_reason_detail = "interrupted:external_stop"
                break

            # Check output truncation
            if completion.finish_reason == "length":
                end_reason = EndReason.TOKEN_LIMIT
                end_reason_detail = _LIMIT_DETAIL[budget.limiting_factor]

                steps.append(self._make_assistant_step(
                    start=assistant_start,
                    end=len(accumulated_tokens),
                    message_start=assistant_message_start,
                    message_end=assistant_message_end,
                    round_index=round_index,
                    step_start_time=step_start_time,
                    completion=completion,
                    budget=budget,
                    recipient=last_message.recipient,
                    weight_segments=list(round_weight_segments),
                ))
                break

            # Extract text content
            text_content = ""
            for block in last_message.content:
                if hasattr(block, "text"):
                    text_content += block.text
            text_content = text_content.strip()

            recipient = last_message.recipient
            channel = last_message.channel

            # Case 1: No recipient (end of generation)
            if recipient is None:
                if channel == "final":
                    end_reason = EndReason.COMPLETED
                    end_reason_detail = "completed:final"
                else:
                    end_reason = EndReason.ERROR
                    end_reason_detail = "error:no_final_channel"
                    errors.append(f"recipient=None but channel={channel}, expected channel=final")

                steps.append(self._make_assistant_step(
                    start=assistant_start,
                    end=len(accumulated_tokens),
                    message_start=assistant_message_start,
                    message_end=assistant_message_end,
                    round_index=round_index,
                    step_start_time=step_start_time,
                    completion=completion,
                    budget=budget,
                    weight_segments=list(round_weight_segments),
                ))
                break

            # Case 2: Has recipient but empty content
            if not text_content:
                end_reason = EndReason.ERROR
                end_reason_detail = "error:invalid_tool_call"
                errors.append(f"recipient={recipient} but content is empty")

                steps.append(self._make_assistant_step(
                    start=assistant_start,
                    end=len(accumulated_tokens),
                    message_start=assistant_message_start,
                    message_end=assistant_message_end,
                    round_index=round_index,
                    step_start_time=step_start_time,
                    completion=completion,
                    budget=budget,
                    recipient=recipient,
                    weight_segments=list(round_weight_segments),
                ))
                break

            # Case 3: Tool call in final channel (error)
            if channel == "final":
                end_reason = EndReason.ERROR
                end_reason_detail = "error:tool_call_in_final_channel"
                errors.append(f"recipient={recipient} but channel=final")

                steps.append(self._make_assistant_step(
                    start=assistant_start,
                    end=len(accumulated_tokens),
                    message_start=assistant_message_start,
                    message_end=assistant_message_end,
                    round_index=round_index,
                    step_start_time=step_start_time,
                    completion=completion,
                    budget=budget,
                    recipient=recipient,
                    tool_input=text_content,
                    weight_segments=list(round_weight_segments),
                ))
                break

            # Normal tool call
            steps.append(self._make_assistant_step(
                start=assistant_start,
                end=len(accumulated_tokens),
                message_start=assistant_message_start,
                message_end=assistant_message_end,
                round_index=round_index,
                step_start_time=step_start_time,
                completion=completion,
                budget=budget,
                recipient=recipient,
                tool_input=text_content,
                weight_segments=list(round_weight_segments),
            ))

            # Check stop before tool call
            if stop_event is not None and stop_event.is_set():
                end_reason = EndReason.INTERRUPTED
                end_reason_detail = "interrupted:external_stop"
                break

            # Call tool (check total time budget first)
            tool_start_time = time.time()
            budget_limit = self.config.max_total_tool_time
            if budget_limit > 0 and total_tool_time >= budget_limit:
                report = ToolCallReport(
                    tool_name=recipient,
                    tool_input=text_content,
                    tool_output=(
                        f"Error: tool time budget exceeded "
                        f"(used {total_tool_time:.1f}s / {budget_limit:.1f}s limit)"
                    ),
                    elapsed=0.0,
                    error=ToolErrorType.TIMEOUT,
                )
            else:
                report = await self.mcp_executor.call_tool(recipient, text_content)
                total_tool_time += report.elapsed

            tool_message = Message.from_author_and_content(
                Author.new(Role.TOOL, recipient),
                report.tool_output,
            ).with_recipient("assistant")
            if channel:
                tool_message = tool_message.with_channel(channel)

            tool_message_start = len(messages)
            messages.append(tool_message)
            tool_message_end = len(messages)

            tool_tokens = self.encoding.render_conversation_for_completion(
                Conversation.from_messages([tool_message]),
                Role.ASSISTANT,
            )
            tool_start = len(accumulated_tokens)
            accumulated_tokens += tool_tokens

            if accumulated_logprobs is not None:
                accumulated_logprobs.extend([None] * len(tool_tokens))
            if accumulated_top_logprobs is not None:
                accumulated_top_logprobs.extend([None] * len(tool_tokens))
            if accumulated_entropy is not None:
                accumulated_entropy.extend([None] * len(tool_tokens))
            if accumulated_routing is not None:
                accumulated_routing.extend([None] * len(tool_tokens))

            tool_step = ToolStep(
                start=tool_start,
                end=len(accumulated_tokens),
                message_start=tool_message_start,
                message_end=tool_message_end,
                round_index=round_index,
                created_at=tool_start_time,
                elapsed=report.elapsed,
                tool_name=report.tool_name,
                tool_input=text_content,
                tool_output=report.tool_output,
                error=report.error,
                early_exit=report.early_exit,
                structured_output=report.structured_output,
            )
            steps.append(tool_step)

            tool_name = recipient[:15]
            logger.info(
                f"{rid} R: {round_index:2d} | tool={tool_name:15s} | "
                f"gen={len(completion.token_ids):5d} | tool_t={report.elapsed:5.1f}s"
            )

            if report.early_exit:
                end_reason = EndReason.TOOL_EARLY_EXIT
                end_reason_detail = f"tool_early_exit:{report.tool_name}"
                break

            if report.error is not None and self.config.tool_error_strategy == "stop":
                end_reason = EndReason.ERROR
                end_reason_detail = f"error:tool_{report.error.value}"
                errors.append(f"Tool {report.tool_name} failed: {report.tool_output}")
                break

            # Reset round-level state for next round
            round_token_ids = []
            round_weight_segments = []
            round_start_offset = None
            round_index += 1

        # =====================================================================
        # Finalize
        # =====================================================================
        if end_reason is None:
            end_reason = EndReason.MAX_ROUNDS
            end_reason_detail = f"max_rounds:{max_rounds}"

        # =====================================================================
        # Extra Round（TOKEN_LIMIT / MAX_ROUNDS 時強制收尾）
        # =====================================================================
        extra_round_cfg = self.config.extra_round
        stopped = stop_event is not None and stop_event.is_set()
        should_extra_round = (
            extra_round_cfg is not None
            and not stopped
            and (
                (end_reason == EndReason.TOKEN_LIMIT and extra_round_cfg.on_token_limit)
                or (end_reason == EndReason.MAX_ROUNDS and extra_round_cfg.on_max_rounds)
            )
        )
        if should_extra_round:
            extra_result = await self._run_extra_round(
                rid=rid,
                accumulated_tokens=accumulated_tokens,
                messages=messages,
                steps=steps,
                accumulated_logprobs=accumulated_logprobs,
                accumulated_top_logprobs=accumulated_top_logprobs,
                accumulated_entropy=accumulated_entropy,
                accumulated_routing=accumulated_routing,
                stop_event=stop_event,
                record_config=record_config,
                errors=errors,
            )
            if extra_result is not None:
                total_generated_tokens += extra_result.generated_tokens
                if extra_result.end_reason is not None:
                    end_reason = extra_result.end_reason
                    end_reason_detail = extra_result.detail

        result = ReactResult(
            end_reason=end_reason,
            end_reason_detail=end_reason_detail,
            errors=errors,
            tokens=accumulated_tokens,
            logprobs=accumulated_logprobs,
            entropy=accumulated_entropy,
            top_logprobs=accumulated_top_logprobs,
            routing_indices=accumulated_routing,
            steps=steps,
            conversation=Conversation.from_messages(messages),
            total_tool_time=total_tool_time,
            num_generated_tokens=total_generated_tokens,
            abort_records=abort_records,
        )

        logger.info(
            f"{rid} END    | reason={result.end_reason.value:13s} | "
            f"rounds={round_index:3d} | tokens={total_generated_tokens:6d}"
        )
        return result

    # =========================================================================
    # Helper Methods
    # =========================================================================

    @staticmethod
    def _make_assistant_step(
        *,
        start: int,
        end: int,
        message_start: int,
        message_end: int,
        round_index: int,
        step_start_time: float,
        completion: "CompletionResult",
        budget: TokenBudget,
        recipient: Optional[str] = None,
        tool_input: Optional[str] = None,
        weight_segments: Optional[List["WeightSegment"]] = None,
        parse_error: Optional[str] = None,
    ) -> AssistantStep:
        """統一建構 AssistantStep（消除 5 處 copy-paste）"""
        return AssistantStep(
            start=start,
            end=end,
            message_start=message_start,
            message_end=message_end,
            round_index=round_index,
            created_at=step_start_time,
            elapsed=time.time() - step_start_time,
            stop_reason=completion.finish_reason,
            truncation_reason=budget.limiting_factor if completion.finish_reason == "length" else None,
            recipient=recipient,
            tool_input=tool_input,
            usage=completion.usage,
            segments=completion.segments,
            weight_segments=weight_segments or [],
            parse_error=parse_error,
        )

    @staticmethod
    def _on_abort(
        completion: "CompletionResult",
        accumulated_tokens: List[int],
        round_token_ids: List[int],
        round_weight_segments: List[WeightSegment],
        accumulated_logprobs: Optional[List[Optional[float]]],
        accumulated_top_logprobs: Optional[List[Optional[Dict[int, float]]]],
        accumulated_entropy: Optional[List[Optional[float]]],
        accumulated_routing: Optional[List[Optional[List[List[int]]]]],
        abort_records: List[AbortRecord],
        round_index: int,
    ) -> None:
        """處理 abort：保留 partial tokens，記錄 weight segment 和 abort event。"""
        seg_start = len(round_token_ids)

        # 1. 保留 partial tokens
        prompt_len = len(accumulated_tokens)
        accumulated_tokens.extend(completion.token_ids)
        round_token_ids.extend(completion.token_ids)

        # 2. 記錄 weight segment
        round_weight_segments.append(WeightSegment(
            start=seg_start,
            end=seg_start + len(completion.token_ids),
            weight_version=completion.weight_version or "unknown",
        ))

        # 3. 更新 logprobs（保留 partial 的 logprobs）
        if accumulated_logprobs is not None:
            if completion.logprobs is not None:
                accumulated_logprobs.extend(completion.logprobs)
            else:
                accumulated_logprobs.extend([None] * len(completion.token_ids))

        if accumulated_top_logprobs is not None:
            if completion.top_logprobs is not None:
                accumulated_top_logprobs.extend(completion.top_logprobs)
            else:
                accumulated_top_logprobs.extend([None] * len(completion.token_ids))

        if accumulated_entropy is not None:
            if completion.entropy is not None:
                accumulated_entropy.extend(completion.entropy)
            else:
                accumulated_entropy.extend([None] * len(completion.token_ids))

        # 4. 更新 routing — same logic as main loop
        # sglang returns routing for positions [0, N-1) where N = prompt + completion
        # (including prefix-cached tokens). Prefill positions overwrite previous values.
        if accumulated_routing is not None and completion.routing_indices is not None:
            routing = completion.routing_indices
            # Fill prefill positions
            for i in range(min(len(routing), prompt_len)):
                accumulated_routing[i] = routing[i]
            # Append completion positions + pad None for uncovered tail
            if len(routing) > prompt_len:
                accumulated_routing.extend(routing[prompt_len:])
                remaining = len(completion.token_ids) - (len(routing) - prompt_len)
                accumulated_routing.extend([None] * remaining)
            else:
                accumulated_routing.extend([None] * len(completion.token_ids))
        elif accumulated_routing is not None:
            accumulated_routing.extend([None] * len(completion.token_ids))

        # 4. 記錄 abort event
        abort_records.append(AbortRecord(
            round_index=round_index,
            weight_version=completion.weight_version,
            partial_token_count=len(completion.token_ids),
            timestamp=time.time(),
        ))

    # =========================================================================
    # Extra Round
    # =========================================================================

    @dataclass
    class _ExtraRoundResult:
        end_reason: Optional[EndReason]  # None = 不改變原本的 end_reason
        detail: Optional[str]
        generated_tokens: int

    async def _run_extra_round(
        self,
        *,
        rid: str,
        accumulated_tokens: List[int],
        messages: List[Message],
        steps: List[Step],
        accumulated_logprobs: Optional[List[Optional[float]]],
        accumulated_top_logprobs: Optional[List[Optional[Dict[int, float]]]],
        accumulated_entropy: Optional[List[Optional[float]]],
        accumulated_routing: Optional[List[Optional[List[List[int]]]]],
        stop_event: Optional[asyncio.Event],
        record_config: Optional[RecordConfig],
        errors: List[str],
    ) -> Optional["UnifiedReactRunner._ExtraRoundResult"]:
        """TOKEN_LIMIT / MAX_ROUNDS 後的強制收尾輪。

        注入「時間到」user message + force final channel，讓模型立刻輸出答案。
        不支援 abort retry 和 segment temperature。
        """
        cfg = self.config.extra_round
        assert cfg is not None
        tracking = (accumulated_logprobs, accumulated_top_logprobs, accumulated_entropy)

        # 1. Render injection tokens（先算，不急著 append）
        user_msg = Message.from_role_and_content(Role.USER, cfg.message)
        user_tokens = self.encoding.render_conversation_for_completion(
            Conversation.from_messages([user_msg]), Role.ASSISTANT,
        )
        final_tokens = self.encoding.encode('<|channel|>final', allowed_special='all')

        # 2. 先算 budget 再注入（避免 budget 不足時留下孤兒 tokens）
        injection_len = len(user_tokens) + len(final_tokens)
        context_remaining = self.config.max_context_tokens - len(accumulated_tokens) - injection_len
        extra_budget = min(cfg.budget, context_remaining)
        if extra_budget < cfg.min_budget:
            logger.info(f"{rid} EXTRA  | skipped: context_remaining={context_remaining}")
            return None

        # 3. 注入 user message + force final
        logger.info(f"{rid} EXTRA  | forced final round, budget={extra_budget}")
        messages.append(user_msg)
        for toks in (user_tokens, final_tokens):
            accumulated_tokens.extend(toks)
            for arr in tracking:
                if arr is not None:
                    arr.extend([None] * len(toks))
            if accumulated_routing is not None:
                accumulated_routing.extend([None] * len(toks))

        # 4. Generate（直接呼叫 backend，繞過 segment temperature）
        step_start_time = time.time()
        assistant_start = len(accumulated_tokens)
        stop_token_ids = self.encoding.stop_tokens_for_assistant_actions()
        try:
            gen_result = await self.llm_backend.generate(
                prompt_tokens=accumulated_tokens,
                max_tokens=extra_budget,
                stop_token_ids=stop_token_ids,
                sampling=self.config.sampling,
                stream=self.config.use_streaming,
                stop_event=stop_event,
                record=record_config,
            )
            completion = CompletionResult(
                token_ids=gen_result.token_ids,
                finish_reason=gen_result.finish_reason,
                logprobs=gen_result.logprobs,
                top_logprobs=gen_result.top_logprobs,
                entropy=gen_result.entropy,
                routing_indices=gen_result.routing_indices,
                cached_tokens=gen_result.cached_tokens,
                usage=gen_result.usage,
                weight_version=gen_result.weight_version,
            )
        except Exception as exc:
            logger.exception(f"{rid} EXTRA  | generate error: {exc}")
            errors.append(f"Extra round generate error: {exc}")
            return self._ExtraRoundResult(end_reason=None, detail=None, generated_tokens=0)

        if not completion.token_ids:
            return self._ExtraRoundResult(end_reason=None, detail=None, generated_tokens=0)

        generated_count = len(completion.token_ids)

        # 5. 更新 accumulated arrays
        accumulated_tokens.extend(completion.token_ids)
        for arr, data in zip(tracking, (completion.logprobs, completion.top_logprobs, completion.entropy)):
            if arr is not None:
                arr.extend(data if data is not None else [None] * generated_count)
        if accumulated_routing is not None:
            if completion.routing_indices is not None:
                prompt_len = len(accumulated_tokens) - generated_count
                routing = completion.routing_indices
                for i in range(min(len(routing), prompt_len)):
                    accumulated_routing[i] = routing[i]
                if len(routing) > prompt_len:
                    accumulated_routing.extend(routing[prompt_len:])
                    accumulated_routing.extend([None] * (generated_count - (len(routing) - prompt_len)))
                else:
                    accumulated_routing.extend([None] * generated_count)
            else:
                accumulated_routing.extend([None] * generated_count)

        # 6. Parse + 記錄 step
        weight_segments = [WeightSegment(
            start=0, end=generated_count,
            weight_version=completion.weight_version or "unknown",
        )]
        budget = TokenBudget(max_tokens=extra_budget, limiting_factor=TruncationReason.ROUND_LIMIT)

        # Prepend <|channel|>final so parser sees channel header
        parse_tokens = list(final_tokens) + list(completion.token_ids)
        try:
            parsed_messages = self.encoding.parse_messages_from_completion_tokens(
                parse_tokens, role=Role.ASSISTANT,
            )
        except Exception as exc:
            logger.warning(f"{rid} EXTRA  | parse error: {exc}")
            errors.append(f"Extra round parse error: {exc}")
            steps.append(self._make_assistant_step(
                start=assistant_start, end=len(accumulated_tokens),
                message_start=len(messages), message_end=len(messages),
                round_index=-1, step_start_time=step_start_time,
                completion=completion, budget=budget, weight_segments=weight_segments,
                parse_error=str(exc),
            ))
            return self._ExtraRoundResult(end_reason=None, detail=None, generated_tokens=generated_count)

        msg_start = len(messages)
        messages.extend(parsed_messages)
        last_message = parsed_messages[-1]

        steps.append(self._make_assistant_step(
            start=assistant_start, end=len(accumulated_tokens),
            message_start=msg_start, message_end=len(messages),
            round_index=-1, step_start_time=step_start_time,
            completion=completion, budget=budget,
            recipient=last_message.recipient, weight_segments=weight_segments,
        ))

        # 7. 判定結果
        if completion.finish_reason == "length":
            logger.info(f"{rid} EXTRA  | truncated, tokens={generated_count}")
            return self._ExtraRoundResult(
                end_reason=EndReason.TOKEN_LIMIT, detail="token_limit:forced_final",
                generated_tokens=generated_count,
            )

        if last_message.channel != "final" or last_message.recipient is not None:
            errors.append(
                f"Extra round: expected final channel, "
                f"got channel={last_message.channel} recipient={last_message.recipient}"
            )
        logger.info(f"{rid} EXTRA  | completed, tokens={generated_count}")
        return self._ExtraRoundResult(
            end_reason=EndReason.COMPLETED, detail="completed:forced_final",
            generated_tokens=generated_count,
        )

    def _calculate_token_budget(
        self,
        input_length: int,
        generated_so_far: int,
    ) -> TokenBudget:
        remaining_generation = self.config.max_total_tokens - generated_so_far
        remaining_context = self.config.max_context_tokens - input_length

        candidates = [
            (remaining_generation, TruncationReason.GENERATION_QUOTA),
            (remaining_context, TruncationReason.CONTEXT_SPACE),
            (self.config.max_round_tokens, TruncationReason.ROUND_LIMIT),
        ]
        max_tokens, limiting_factor = min(candidates, key=lambda x: x[0])

        return TokenBudget(
            max_tokens=max_tokens,
            limiting_factor=limiting_factor,
        )

    async def _generate(
        self,
        prompt_tokens: List[int],
        max_tokens: int,
        stop_event: Optional[asyncio.Event],
        record_config: Optional[RecordConfig],
    ) -> CompletionResult:
        """生成 completion

        路徑選擇：
        1. segment_temperature 啟用 → segment-aware 兩階段生成
        2. use_streaming=True → streaming（可中斷，支援 logprobs + top_logprobs）
        3. use_streaming=False → non-streaming（支援 routing_indices）
        """
        if self.config.segment_temperature is not None:
            return await self._generate_with_segment_temperature(
                prompt_tokens=prompt_tokens,
                max_tokens=max_tokens,
                stop_event=stop_event,
            )

        stop_token_ids = self.encoding.stop_tokens_for_assistant_actions()

        result = await self.llm_backend.generate(
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            stop_token_ids=stop_token_ids,
            sampling=self.config.sampling,
            stream=self.config.use_streaming,
            stop_event=stop_event,
            record=record_config,
        )

        return CompletionResult(
            token_ids=result.token_ids,
            finish_reason=result.finish_reason,
            logprobs=result.logprobs,
            top_logprobs=result.top_logprobs,
            entropy=result.entropy,
            routing_indices=result.routing_indices,
            cached_tokens=result.cached_tokens,
            usage=result.usage,
            weight_version=result.weight_version,
        )

    def _parse_header_tokens(
        self,
        header_tokens: List[int],
    ) -> Tuple[Optional[str], Optional[str]]:
        text = self.encoding.decode(header_tokens)
        channel = m.group(1) if (m := _CHANNEL_RE.search(text)) else None
        recipient = m.group(1) if (m := _RECIPIENT_RE.search(text)) else None
        return channel, recipient

    async def _generate_with_segment_temperature(
        self,
        prompt_tokens: List[int],
        max_tokens: int,
        stop_event: Optional[asyncio.Event],
    ) -> CompletionResult:
        """使用 segment-aware temperature 的兩階段生成"""
        seg_config = self.config.segment_temperature
        assert seg_config is not None

        all_token_ids: List[int] = []
        segments: List[SegmentMeta] = []
        current_prompt = list(prompt_tokens)
        tokens_remaining = max_tokens
        finish_reason: Optional[str] = None

        header_stop_tokens = [_TOKEN_MESSAGE]
        content_stop_tokens = [_TOKEN_END, _TOKEN_CALL, _TOKEN_RETURN]

        # Segment temperature 固定用 streaming，不支援 logprobs/routing
        header_sampling = self.config.sampling.model_copy(
            update={"temperature": seg_config.header_temperature}
        )

        header_result = None
        content_result = None

        while tokens_remaining > 0:
            if stop_event is not None and stop_event.is_set():
                finish_reason = "stop"
                break

            # Phase A: Generate header (stop at <|message|>)
            header_result = await self.llm_backend.generate(
                prompt_tokens=current_prompt,
                max_tokens=min(256, tokens_remaining),
                stop_token_ids=header_stop_tokens,
                sampling=header_sampling,
                stream=True,
                stop_event=stop_event,
            )

            if not header_result.token_ids:
                finish_reason = header_result.finish_reason or "stop"
                break

            # Abort during header → return partial immediately
            if header_result.finish_reason == "abort":
                all_token_ids.extend(header_result.token_ids)
                return CompletionResult(
                    token_ids=all_token_ids,
                    finish_reason="abort",
                    segments=segments,
                    weight_version=header_result.weight_version,
                )

            header_start = len(all_token_ids)
            all_token_ids.extend(header_result.token_ids)
            header_end = len(all_token_ids)
            tokens_remaining -= len(header_result.token_ids)
            current_prompt.extend(header_result.token_ids)

            if header_result.finish_reason == "length":
                segments.append(SegmentMeta(
                    phase="header",
                    start=header_start,
                    end=header_end,
                    temperature=seg_config.header_temperature,
                ))
                finish_reason = "length"
                break

            channel, recipient = self._parse_header_tokens(header_result.token_ids)

            segments.append(SegmentMeta(
                phase="header",
                start=header_start,
                end=header_end,
                temperature=seg_config.header_temperature,
                channel=channel,
                recipient=recipient,
            ))

            if stop_event is not None and stop_event.is_set():
                finish_reason = "stop"
                break

            # Phase B: Generate content
            content_temperature = seg_config.get_content_temperature(channel, recipient)

            logger.debug(
                f"Segment generation: channel={channel}, recipient={recipient}, "
                f"content_temp={content_temperature}"
            )

            if tokens_remaining <= 0:
                finish_reason = "length"
                break

            content_sampling = self.config.sampling.model_copy(
                update={"temperature": content_temperature}
            )

            content_result = await self.llm_backend.generate(
                prompt_tokens=current_prompt,
                max_tokens=tokens_remaining,
                stop_token_ids=content_stop_tokens,
                sampling=content_sampling,
                stream=True,
                stop_event=stop_event,
            )

            # Abort during content → return partial immediately
            if content_result.finish_reason == "abort":
                all_token_ids.extend(content_result.token_ids)
                return CompletionResult(
                    token_ids=all_token_ids,
                    finish_reason="abort",
                    segments=segments,
                    weight_version=content_result.weight_version,
                )

            content_start = len(all_token_ids)
            all_token_ids.extend(content_result.token_ids)
            content_end = len(all_token_ids)
            tokens_remaining -= len(content_result.token_ids)
            current_prompt.extend(content_result.token_ids)

            segments.append(SegmentMeta(
                phase="content",
                start=content_start,
                end=content_end,
                temperature=content_temperature,
                channel=channel,
                recipient=recipient,
            ))

            if content_result.finish_reason == "length":
                finish_reason = "length"
                break

            if content_result.token_ids:
                last_token = content_result.token_ids[-1]
                if last_token in (_TOKEN_CALL, _TOKEN_RETURN):
                    finish_reason = "stop"
                    break
                elif last_token == _TOKEN_END:
                    continue

            finish_reason = content_result.finish_reason or "stop"
            break

        if finish_reason is None:
            finish_reason = "length"

        # weight_version: use the last sub-generation's version
        last_version = None
        if content_result is not None:
            last_version = content_result.weight_version
        elif header_result is not None:
            last_version = header_result.weight_version

        return CompletionResult(
            token_ids=all_token_ids,
            finish_reason=finish_reason,
            segments=segments,
            weight_version=last_version,
        )
