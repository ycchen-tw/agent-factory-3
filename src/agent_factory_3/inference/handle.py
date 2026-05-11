"""AgentHandle — self-contained inference ReAct agent with real-time observability.

Self-contained: no imports from agent_factory_3.rollout / orchestrator / trainer.
External deps: openai_harmony, fastmcp, aiohttp.

Created: 2026-03-28

Usage::

    handle = AgentHandle(AgentConfig(server_url="http://localhost:30100"))
    await handle.start("Solve this problem: ...")

    # Real-time observation (any time, including during streaming)
    print(handle.phase, handle.streaming_text)

    # Commands (sync, immediate even mid-stream)
    handle.force_extra_round()
    handle.stop()

    # Event stream
    async for event in handle.events():
        print(event.type)

    # Wait for completion
    result = await handle.join()
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, AsyncGenerator

import aiohttp
from fastmcp import Client as McpClient
from openai_harmony import (
    Author,
    Conversation,
    DeveloperContent,
    HarmonyEncodingName,
    Message,
    RenderConversationConfig,
    ReasoningEffort,
    Role,
    SystemContent,
    ToolDescription,
    ToolNamespaceConfig,
    load_custom_harmony_encoding,
    load_harmony_encoding,
)

from .types import (
    DEFAULT_ASSISTANT_PREFIX,
    AgentConfig,
    AgentResult,
    EndReason,
    Event,
    EventType,
    ForceExtraRoundCommand,
    InjectMessageCommand,
    Phase,
    StepInfo,
    ToolReport,
)

logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

_SYSTEM_PREFIXES = ("builtin_", "system_")


def _is_system_namespace(ns: str) -> bool:
    return ns.startswith(_SYSTEM_PREFIXES)


def _to_harmony_namespace(ns: str) -> str:
    for prefix in _SYSTEM_PREFIXES:
        if ns.startswith(prefix):
            return ns[len(prefix):]
    return ns


_REASONING_EFFORT_MAP = {
    "low": ReasoningEffort.LOW,
    "medium": ReasoningEffort.MEDIUM,
    "high": ReasoningEffort.HIGH,
}


def _normalize_finish_reason(raw: Any) -> str | None:
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw.get("type")
    return str(raw)


# =============================================================================
# Internal types
# =============================================================================


@dataclass
class _ToolMeta:
    namespace: str
    name: str
    raw_string_param: str | None = None
    input_schema: dict | None = None


@dataclass
class _TokenBudget:
    max_tokens: int
    limiting_factor: str  # "round" | "context" | "generation"

    @property
    def can_generate(self) -> bool:
        return self.max_tokens > 0


# =============================================================================
# AgentHandle
# =============================================================================


class AgentHandle:
    """Self-contained inference ReAct agent with real-time observability and control.

    Single sglang backend, streaming-only. All state is readable mid-stream via
    properties (safe in asyncio single-thread). Commands take effect between SSE
    chunks (stop, force_extra_round) or at round boundaries (pause, inject).
    """

    def __init__(
        self,
        config: AgentConfig,
        *,
        mcp_config: dict[str, Any] | None = None,
    ):
        self._config = config
        self._mcp_config = mcp_config

        # Harmony encoding
        if config.harmony_custom_config_path:
            self._encoding = load_custom_harmony_encoding(config.harmony_custom_config_path)
        else:
            self._encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

        # --- Public state (readable externally at any time) ---
        self._phase: Phase = Phase.IDLE
        self._round_index: int = 0
        self._streaming_tokens: list[int] = []
        self._accumulated_tokens: list[int] = []
        self._messages: list[Message] = []
        self._steps: list[StepInfo] = []
        self._total_generated_tokens: int = 0
        self._total_tool_time: float = 0.0
        self._errors: list[str] = []
        self._result: AgentResult | None = None

        # --- Command flags (sync-safe) ---
        self._stop_requested: bool = False
        self._pause_requested: bool = False
        self._force_extra_cmd: ForceExtraRoundCommand | None = None
        self._pending_injections: list[InjectMessageCommand] = []
        self._pause_event: asyncio.Event = asyncio.Event()
        self._pause_event.set()  # not paused

        # --- Internal ---
        self._event_queue: asyncio.Queue[Event | None] = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self._mcp_clients: dict[str, McpClient] = {}
        self._tool_metadata: dict[str, _ToolMeta] = {}
        self._session: aiohttp.ClientSession | None = None
        self._stop_token_ids: list[int] = []
        self._interrupt_event: asyncio.Event = asyncio.Event()

    # =========================================================================
    # Public Properties
    # =========================================================================

    @property
    def phase(self) -> Phase:
        return self._phase

    @property
    def round_index(self) -> int:
        return self._round_index

    @property
    def streaming_tokens(self) -> list[int]:
        """Tokens being generated in the current streaming call (copy)."""
        return list(self._streaming_tokens)

    @property
    def streaming_text(self) -> str:
        """Decoded text of current streaming tokens."""
        if not self._streaming_tokens:
            return ""
        return self._encoding.decode(self._streaming_tokens)

    @property
    def accumulated_tokens(self) -> list[int]:
        """All tokens so far including initial prompt (copy)."""
        return list(self._accumulated_tokens)

    @property
    def steps(self) -> list[StepInfo]:
        return list(self._steps)

    @property
    def total_generated_tokens(self) -> int:
        """Includes both committed tokens and in-flight streaming tokens."""
        return self._total_generated_tokens + len(self._streaming_tokens)

    @property
    def result(self) -> AgentResult | None:
        return self._result

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def is_alive(self) -> bool:
        """True if the handle's session is still open (running or dormant)."""
        return self._session is not None and not self._session.closed

    # =========================================================================
    # Commands (sync — immediate flag set, no await)
    # =========================================================================

    def stop(self) -> None:
        """Gracefully stop the agent. Takes effect between SSE chunks
        or immediately cancels a running tool call."""
        self._stop_requested = True
        self._interrupt_event.set()
        self._pause_event.set()  # unblock if paused

    def pause(self) -> None:
        """Pause the agent. Takes effect between SSE chunks during streaming,
        or at the next safe point between rounds."""
        if self._phase not in (Phase.COMPLETED, Phase.FAILED):
            self._pause_requested = True
            self._pause_event.clear()

    def resume(self) -> None:
        """Resume a paused agent."""
        self._pause_requested = False
        self._pause_event.set()

    def force_extra_round(
        self,
        message: str | None = None,
        budget: int | None = None,
        assistant_prefix: str = DEFAULT_ASSISTANT_PREFIX,
    ) -> None:
        """Force immediate extra round. Interrupts current streaming
        or immediately cancels a running tool call.

        ``assistant_prefix`` is encoded with ``allowed_special="all"`` and
        prepended to the sampled tokens, prefilling the assistant turn.
        Defaults to forcing the final channel; pass e.g.
        ``"<|channel|>final<|message|>\\boxed{"`` to also prefill body content.
        """
        self._force_extra_cmd = ForceExtraRoundCommand(
            message=message, budget=budget, assistant_prefix=assistant_prefix,
        )
        self._interrupt_event.set()
        self._pause_event.set()  # unblock if paused

    def inject_message(self, content: str, role: str = "user") -> None:
        """Queue a message injection. Applied at next round boundary."""
        self._pending_injections.append(InjectMessageCommand(content=content, role=role))

    # =========================================================================
    # Probe (non-destructive read-only fork)
    # =========================================================================

    async def probe(
        self,
        message: str = "Provide your current best answer.",
        budget: int = 512,
        include_partial: bool = True,
        temperature: float | None = None,
        assistant_prefix: str = DEFAULT_ASSISTANT_PREFIX,
    ) -> str | None:
        """Non-destructive probe: snapshot context, force final, extract answer.

        Takes a snapshot of current tokens (including partial streaming if
        ``include_partial``), injects a user message + assistant prefill,
        and sends a **separate** sglang request.  The main generation loop
        is completely unaffected.  Uses sglang prefix cache for cheap prefill.

        Args:
            message: User message injected before the assistant prefill.
            budget: Max tokens for the probe generation.
            include_partial: Include in-flight streaming tokens in snapshot.
            temperature: Sampling temperature for the probe (None = use config default).
            assistant_prefix: Text encoded with ``allowed_special="all"`` and
                appended after the user message to prefill the assistant turn.
                Defaults to forcing the final channel; pass e.g.
                ``"<|channel|>final<|message|>\\boxed{"`` to also prefill body
                content. The returned text will include the prefilled body.

        Returns final-channel text, or None on failure.
        """
        if self._session is None or self._session.closed:
            return None

        # 1. Snapshot (read-only, safe in asyncio single-thread)
        snapshot = list(self._accumulated_tokens)
        if include_partial:
            snapshot.extend(self._streaming_tokens)
        if not snapshot:
            return None

        # 2. Build injection tokens
        user_msg = Message.from_role_and_content(Role.USER, message)
        user_tokens = self._encoding.render_conversation_for_completion(
            Conversation.from_messages([user_msg]),
            Role.ASSISTANT,
        )
        final_tokens = self._encoding.encode(assistant_prefix, allowed_special="all")

        prompt = snapshot + user_tokens + final_tokens
        remaining = self._config.max_context_tokens - len(prompt)
        actual_budget = min(budget, remaining)
        if actual_budget < 32:
            return None

        # 3. One-shot generate (no side effects)
        try:
            output_ids, _finish = await self._sglang_generate_oneshot(
                prompt, actual_budget, temperature=temperature,
            )
        except Exception as e:
            logger.warning(f"Probe generate failed: {e}")
            return None

        if not output_ids:
            return None

        # 4. Parse final-channel text
        try:
            parse_tokens = list(final_tokens) + output_ids
            parsed = self._encoding.parse_messages_from_completion_tokens(
                parse_tokens, role=Role.ASSISTANT,
            )
            return _extract_text(parsed[-1]) if parsed else None
        except Exception as e:
            logger.warning(f"Probe parse failed: {e}")
            return None

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def start(
        self,
        user_prompt: str,
        *,
        system_prompt: str | None = None,
        dev_instructions: str | None = None,
        conversation_start_date: str | None = None,
    ) -> None:
        """Start the agent (non-blocking, creates asyncio.Task)."""
        assert self._task is None, "Agent already started"
        self._task = asyncio.create_task(
            self._run(user_prompt, system_prompt, dev_instructions, conversation_start_date),
        )

    async def events(self) -> AsyncGenerator[Event, None]:
        """Consume the event stream. Ends after COMPLETED or ERROR."""
        while True:
            event = await self._event_queue.get()
            if event is None:
                break
            yield event
            if event.type in (EventType.COMPLETED, EventType.ERROR):
                break

    async def join(self) -> AgentResult | None:
        """Wait for the agent to finish and return the result."""
        if self._task is not None:
            with suppress(Exception):
                await self._task
        return self._result

    async def continue_from(self, message: str, role: str = "user") -> None:
        """Inject message and restart the react loop from dormant state."""
        assert self._phase == Phase.DORMANT, f"continue_from requires DORMANT, got {self._phase}"
        self.inject_message(message, role)
        self._stop_requested = False
        self._force_extra_cmd = None
        self._interrupt_event.clear()
        self._event_queue = asyncio.Queue()
        self._task = asyncio.create_task(self._continue_run())

    async def _continue_run(self) -> None:
        """Apply pending injections and re-enter the react loop."""
        try:
            self._apply_injections()
            end_reason, end_reason_detail = await self._react_loop()
            self._result = AgentResult(
                end_reason=end_reason,
                end_reason_detail=end_reason_detail,
                tokens=list(self._accumulated_tokens),
                conversation=Conversation.from_messages(self._messages),
                steps=list(self._steps),
                num_generated_tokens=self._total_generated_tokens,
                total_tool_time=self._total_tool_time,
                errors=list(self._errors),
            )
            self._phase = Phase.DORMANT
            self._emit(EventType.COMPLETED, end_reason=end_reason.value, result=self._result)
        except Exception as e:
            logger.exception(f"Continue run failed: {e}")
            self._phase = Phase.FAILED
            self._errors.append(str(e))
            self._emit(EventType.ERROR, error=str(e))
            await self._cleanup()
        finally:
            self._event_queue.put_nowait(None)

    async def close(self) -> None:
        """Explicitly clean up resources. Must be called when done with the handle."""
        if self.is_running:
            self.stop()
            with suppress(Exception):
                await self._task
        await self._cleanup()
        self._phase = Phase.COMPLETED

    # =========================================================================
    # Internal: Main orchestration
    # =========================================================================

    async def _run(
        self,
        user_prompt: str,
        system_prompt: str | None,
        dev_instructions: str | None,
        conversation_start_date: str | None,
    ) -> None:
        try:
            # 1. Setup
            await self._setup_session()
            await self._setup_mcp()
            tool_configs = await self._discover_tools()

            # 2. Build conversation
            conversation = self._build_conversation(
                user_prompt, system_prompt, dev_instructions,
                conversation_start_date, tool_configs,
            )
            self._messages = list(conversation.messages)

            # 3. Initialize tokens
            self._accumulated_tokens = self._encoding.render_conversation_for_completion(
                conversation,
                Role.ASSISTANT,
                config=RenderConversationConfig(
                    auto_drop_analysis=self._config.auto_drop_analysis,
                ),
            )
            self._stop_token_ids = self._encoding.stop_tokens_for_assistant_actions()

            self._steps.append(StepInfo(
                type="init",
                round_index=0,
                token_start=0,
                token_end=len(self._accumulated_tokens),
                created_at=time.time(),
            ))

            self._emit(EventType.STARTED)

            # 4. ReAct loop
            end_reason, end_reason_detail = await self._react_loop()

            # 5. Build result
            self._result = AgentResult(
                end_reason=end_reason,
                end_reason_detail=end_reason_detail,
                tokens=list(self._accumulated_tokens),
                conversation=Conversation.from_messages(self._messages),
                steps=list(self._steps),
                num_generated_tokens=self._total_generated_tokens,
                total_tool_time=self._total_tool_time,
                errors=list(self._errors),
            )
            self._phase = Phase.DORMANT
            self._emit(EventType.COMPLETED, end_reason=end_reason.value, result=self._result)

        except Exception as e:
            logger.exception(f"Agent run failed: {e}")
            self._phase = Phase.FAILED
            self._errors.append(str(e))
            self._emit(EventType.ERROR, error=str(e))
            await self._cleanup()
        finally:
            self._event_queue.put_nowait(None)  # sentinel for events()

    # =========================================================================
    # Internal: ReAct Loop
    # =========================================================================

    async def _react_loop(self) -> tuple[EndReason, str]:
        cfg = self._config
        end_reason: EndReason | None = None
        end_reason_detail: str = ""

        while self._round_index < cfg.max_rounds and end_reason is None:
            # --- Safe point: handle commands ---
            if self._stop_requested:
                end_reason = EndReason.INTERRUPTED
                end_reason_detail = "interrupted:stop_command"
                break

            # Pause
            if self._pause_requested:
                self._phase = Phase.PAUSED
                self._emit(EventType.PAUSED)
                await self._pause_event.wait()
                if self._stop_requested:
                    end_reason = EndReason.INTERRUPTED
                    end_reason_detail = "interrupted:stop_command"
                    break
                self._emit(EventType.RESUMED)

            # Force extra round (set externally)
            if self._force_extra_cmd is not None:
                break

            # Process injected messages
            self._apply_injections()

            # Check token budget
            budget = self._calculate_budget()
            if not budget.can_generate:
                end_reason = EndReason.TOKEN_LIMIT
                end_reason_detail = f"token_limit:{budget.limiting_factor}"
                break

            # --- Generate ---
            self._phase = Phase.GENERATING
            self._streaming_tokens = []
            self._emit(EventType.ROUND_START)

            step_start = time.time()
            round_tokens, finish_reason = await self._stream_generate_with_resume(
                budget.max_tokens,
            )

            if finish_reason == "error" or (not round_tokens and finish_reason != "stop"):
                end_reason = EndReason.ERROR
                detail = f"finish_reason={finish_reason},tokens={len(round_tokens)}"
                end_reason_detail = f"error:generation_failed({detail})"
                if not round_tokens:
                    self._errors.append(f"Empty generation: {detail}")
                self._steps.append(StepInfo(
                    type="assistant",
                    round_index=self._round_index,
                    token_start=len(self._accumulated_tokens) - len(round_tokens),
                    token_end=len(self._accumulated_tokens),
                    created_at=step_start,
                    elapsed=time.time() - step_start,
                    finish_reason=finish_reason,
                ))
                break

            # Interrupted during generation?
            if self._stop_requested:
                self._steps.append(StepInfo(
                    type="assistant",
                    round_index=self._round_index,
                    token_start=len(self._accumulated_tokens) - len(round_tokens),
                    token_end=len(self._accumulated_tokens),
                    created_at=step_start,
                    elapsed=time.time() - step_start,
                    finish_reason="interrupted",
                ))
                end_reason = EndReason.INTERRUPTED
                end_reason_detail = "interrupted:stop_during_generation"
                break

            if self._force_extra_cmd is not None:
                # Keep partial tokens (already in accumulated), go to extra round
                self._steps.append(StepInfo(
                    type="assistant",
                    round_index=self._round_index,
                    token_start=len(self._accumulated_tokens) - len(round_tokens),
                    token_end=len(self._accumulated_tokens),
                    created_at=step_start,
                    elapsed=time.time() - step_start,
                    finish_reason="force_extra",
                ))
                end_reason = EndReason.INTERRUPTED
                end_reason_detail = "interrupted:force_extra_round"
                break

            self._emit(EventType.GENERATION_DONE)

            # --- Parse response ---
            try:
                parsed = self._encoding.parse_messages_from_completion_tokens(
                    round_tokens, role=Role.ASSISTANT,
                )
            except Exception as exc:
                # If output was truncated, report token_limit not parse_error
                if finish_reason == "length":
                    end_reason = EndReason.TOKEN_LIMIT
                    end_reason_detail = f"token_limit:{budget.limiting_factor}"
                else:
                    end_reason = EndReason.ERROR
                    end_reason_detail = "error:parse_error"
                    self._errors.append(str(exc))
                self._steps.append(StepInfo(
                    type="assistant",
                    round_index=self._round_index,
                    token_start=len(self._accumulated_tokens) - len(round_tokens),
                    token_end=len(self._accumulated_tokens),
                    created_at=step_start,
                    elapsed=time.time() - step_start,
                ))
                break

            assistant_start = len(self._accumulated_tokens) - len(round_tokens)
            msg_start = len(self._messages)
            self._messages.extend(parsed)
            last_msg = parsed[-1]

            text_content = _extract_text(last_msg)
            recipient = last_msg.recipient
            channel = getattr(last_msg, "channel", None)

            # Truncated by length
            if finish_reason == "length":
                end_reason = EndReason.TOKEN_LIMIT
                end_reason_detail = f"token_limit:{budget.limiting_factor}"
                self._steps.append(StepInfo(
                    type="assistant",
                    round_index=self._round_index,
                    token_start=assistant_start,
                    token_end=len(self._accumulated_tokens),
                    created_at=step_start,
                    elapsed=time.time() - step_start,
                    finish_reason="length",
                    recipient=recipient,
                ))
                break

            # No recipient → final answer
            if recipient is None:
                if channel == "final":
                    end_reason = EndReason.COMPLETED
                    end_reason_detail = "completed:final"
                else:
                    end_reason = EndReason.ERROR
                    end_reason_detail = f"error:no_final_channel(channel={channel})"
                    self._errors.append(f"No recipient and channel={channel}, expected final")
                self._steps.append(StepInfo(
                    type="assistant",
                    round_index=self._round_index,
                    token_start=assistant_start,
                    token_end=len(self._accumulated_tokens),
                    created_at=step_start,
                    elapsed=time.time() - step_start,
                    finish_reason="stop",
                ))
                break

            # Empty content
            if not text_content:
                end_reason = EndReason.ERROR
                end_reason_detail = f"error:empty_tool_call(recipient={recipient})"
                self._errors.append(f"Empty content for recipient={recipient}")
                self._steps.append(StepInfo(
                    type="assistant",
                    round_index=self._round_index,
                    token_start=assistant_start,
                    token_end=len(self._accumulated_tokens),
                    created_at=step_start,
                    elapsed=time.time() - step_start,
                    recipient=recipient,
                ))
                break

            # Tool call in final channel
            if channel == "final":
                end_reason = EndReason.ERROR
                end_reason_detail = f"error:tool_call_in_final(recipient={recipient})"
                self._errors.append(f"Tool call {recipient} in final channel")
                self._steps.append(StepInfo(
                    type="assistant",
                    round_index=self._round_index,
                    token_start=assistant_start,
                    token_end=len(self._accumulated_tokens),
                    created_at=step_start,
                    elapsed=time.time() - step_start,
                    recipient=recipient,
                ))
                break

            # --- Record assistant step ---
            self._steps.append(StepInfo(
                type="assistant",
                round_index=self._round_index,
                token_start=assistant_start,
                token_end=len(self._accumulated_tokens),
                created_at=step_start,
                elapsed=time.time() - step_start,
                finish_reason="stop",
                recipient=recipient,
            ))

            # --- Tool call ---
            self._phase = Phase.CALLING_TOOL
            self._emit(EventType.TOOL_CALL_START, tool_name=recipient)

            # Check tool time budget
            tool_budget = cfg.max_total_tool_time
            if tool_budget > 0 and self._total_tool_time >= tool_budget:
                report = ToolReport(
                    tool_name=recipient,
                    tool_input=text_content,
                    tool_output=(
                        f"Error: tool time budget exceeded "
                        f"(used {self._total_tool_time:.1f}s / {tool_budget:.1f}s limit)"
                    ),
                    elapsed=0.0,
                    error="timeout",
                )
            else:
                report = await self._call_tool_interruptible(recipient, text_content)
                self._total_tool_time += report.elapsed

            self._emit(
                EventType.TOOL_CALL_DONE,
                tool_name=report.tool_name,
                tool_output=report.tool_output[:500],
                tool_error=report.error,
            )

            # Encode tool response (including synthetic "[Tool call cancelled]")
            tool_message = Message.from_author_and_content(
                Author.new(Role.TOOL, recipient),
                report.tool_output,
            ).with_recipient("assistant")
            if channel:
                tool_message = tool_message.with_channel(channel)

            self._messages.append(tool_message)
            tool_tokens = self._encoding.render_conversation_for_completion(
                Conversation.from_messages([tool_message]),
                Role.ASSISTANT,
            )
            tool_start = len(self._accumulated_tokens)
            self._accumulated_tokens.extend(tool_tokens)

            self._steps.append(StepInfo(
                type="tool",
                round_index=self._round_index,
                token_start=tool_start,
                token_end=len(self._accumulated_tokens),
                created_at=time.time(),
                elapsed=report.elapsed,
                tool_name=report.tool_name,
                tool_input=text_content,
                tool_output=report.tool_output,
                tool_error=report.error,
                early_exit=report.early_exit,
            ))

            logger.info(
                f"R:{self._round_index:2d} | tool={recipient:15s} | "
                f"gen={len(round_tokens):5d} | tool_t={report.elapsed:5.1f}s"
            )

            # Tool cancelled by interrupt → break for stop or extra round
            if report.error == "cancelled":
                if self._stop_requested:
                    end_reason = EndReason.INTERRUPTED
                    end_reason_detail = "interrupted:stop_during_tool"
                    break
                if self._force_extra_cmd is not None:
                    end_reason = EndReason.INTERRUPTED
                    end_reason_detail = "interrupted:force_extra_during_tool"
                    break

            if report.early_exit:
                end_reason = EndReason.TOOL_EARLY_EXIT
                end_reason_detail = f"tool_early_exit:{report.tool_name}"
                break

            self._emit(EventType.ROUND_DONE)
            self._round_index += 1

        # === Post-loop ===
        if end_reason is None:
            end_reason = EndReason.MAX_ROUNDS
            end_reason_detail = f"max_rounds:{cfg.max_rounds}"

        # Extra round?
        should_extra = (
            self._force_extra_cmd is not None
            or (
                end_reason in (EndReason.TOKEN_LIMIT, EndReason.MAX_ROUNDS)
                and cfg.extra_round_budget > 0
            )
        )
        if should_extra and not self._stop_requested:
            extra_end, extra_detail = await self._run_extra_round()
            if extra_end is not None:
                end_reason = extra_end
                end_reason_detail = extra_detail

        return end_reason, end_reason_detail

    # =========================================================================
    # Internal: Streaming Generation
    # =========================================================================

    async def _stream_generate_with_resume(
        self,
        max_tokens: int,
    ) -> tuple[list[int], str | None]:
        """Generate with pause/resume support during streaming.

        On pause: breaks SSE, saves partial tokens, awaits resume, then
        re-calls sglang (prefix cache hit for existing tokens).

        Returns (round_tokens, finish_reason).
        """
        round_tokens: list[int] = []
        remaining = max_tokens

        while True:
            tokens, finish_reason = await self._stream_generate(remaining)
            round_tokens.extend(tokens)
            remaining -= len(tokens)

            if finish_reason == "paused":
                self._phase = Phase.PAUSED
                self._emit(EventType.PAUSED)
                await self._pause_event.wait()
                if self._stop_requested or self._force_extra_cmd is not None:
                    return round_tokens, "interrupted"
                self._emit(EventType.RESUMED)
                self._phase = Phase.GENERATING
                if remaining <= 0:
                    return round_tokens, "length"
                continue

            return round_tokens, finish_reason

    async def _stream_generate(
        self,
        max_tokens: int,
    ) -> tuple[list[int], str | None]:
        """Single streaming generate call to sglang /generate.

        Updates self._streaming_tokens in real-time as SSE chunks arrive.
        Breaks on stop/force_extra/pause commands.

        Returns (new_tokens, finish_reason).
        finish_reason: "stop" | "length" | "interrupted" | "paused" | None
        """
        self._streaming_tokens = []

        payload: dict[str, Any] = {
            "input_ids": self._accumulated_tokens,
            "sampling_params": {
                "max_new_tokens": max_tokens,
                "temperature": self._config.temperature,
                "top_p": self._config.top_p,
                "top_k": self._config.top_k if self._config.top_k > 0 else -1,
                "stop_token_ids": self._stop_token_ids,
                **({"sampling_seed": self._config.seed} if self._config.seed is not None else {}),
            },
            "stream": True,
            "return_logprob": False,
            "return_routed_experts": False,
        }
        if self._config.min_p is not None:
            payload["sampling_params"]["min_p"] = self._config.min_p

        assert self._session is not None
        url = f"{self._config.server_url.rstrip('/').removesuffix('/v1')}/generate"
        incremental = self._config.stream_output
        finish_reason: str | None = None
        interrupted = False

        try:
            resp = await self._session.post(url, json=payload)
            try:
                resp.raise_for_status()
                buffer = b""
                done = False

                async for chunk in resp.content.iter_any():
                    if done:
                        break

                    # Check interrupt between chunks
                    if self._stop_requested or self._force_extra_cmd is not None:
                        finish_reason = "interrupted"
                        interrupted = True
                        break
                    if self._pause_requested:
                        finish_reason = "paused"
                        interrupted = True
                        break

                    buffer += chunk
                    while b"\n" in buffer:
                        line, buffer = buffer.split(b"\n", 1)
                        line_str = line.decode("utf-8", errors="replace").strip()
                        if not line_str:
                            continue
                        if line_str == "data: [DONE]":
                            done = True
                            break
                        if not line_str.startswith("data: "):
                            continue
                        data = json.loads(line_str[6:])

                        if "output_ids" in data:
                            new_ids = data["output_ids"]
                            if incremental:
                                self._streaming_tokens.extend(new_ids)
                            else:
                                self._streaming_tokens = new_ids

                        if "meta_info" in data:
                            meta = data["meta_info"]
                            fr = meta.get("finish_reason")
                            if fr:
                                finish_reason = _normalize_finish_reason(fr)
            finally:
                # On early break (interrupt/pause), close connection immediately
                # to avoid TransferEncodingError from partial response read.
                if interrupted:
                    resp.close()
                else:
                    resp.release()

        except aiohttp.ClientError as e:
            n_partial = len(self._streaming_tokens)
            logger.error(f"sglang streaming error: {e} (partial_tokens={n_partial})")
            self._errors.append(f"sglang error: {e} (partial_tokens={n_partial})")
            finish_reason = "error"
        except Exception as e:
            n_partial = len(self._streaming_tokens)
            logger.error(f"sglang unexpected error: {type(e).__name__}: {e} (partial_tokens={n_partial})")
            self._errors.append(f"sglang unexpected: {type(e).__name__}: {e} (partial_tokens={n_partial})")
            finish_reason = "error"

        # Move streaming tokens into accumulated
        tokens = list(self._streaming_tokens)
        self._accumulated_tokens.extend(tokens)
        self._total_generated_tokens += len(tokens)
        self._streaming_tokens = []

        return tokens, finish_reason

    # =========================================================================
    # Internal: Extra Round
    # =========================================================================

    async def _run_extra_round(self) -> tuple[EndReason | None, str]:
        """Forced final round: inject 'time's up' message + force final channel."""
        cmd = self._force_extra_cmd
        cfg = self._config
        message = (cmd.message if cmd and cmd.message else cfg.extra_round_message)
        budget = (cmd.budget if cmd and cmd.budget else cfg.extra_round_budget)
        assistant_prefix = cmd.assistant_prefix if cmd else DEFAULT_ASSISTANT_PREFIX
        self._force_extra_cmd = None

        self._phase = Phase.EXTRA_ROUND
        self._emit(EventType.EXTRA_ROUND_START)

        # Build injection tokens (without appending yet)
        user_msg = Message.from_role_and_content(Role.USER, message)
        user_tokens = self._encoding.render_conversation_for_completion(
            Conversation.from_messages([user_msg]),
            Role.ASSISTANT,
        )
        final_tokens = self._encoding.encode(assistant_prefix, allowed_special="all")

        # Check space
        injection_len = len(user_tokens) + len(final_tokens)
        context_remaining = cfg.max_context_tokens - len(self._accumulated_tokens) - injection_len
        actual_budget = min(budget, context_remaining)
        if actual_budget < 64:
            logger.info(f"Extra round skipped: context_remaining={context_remaining}")
            return None, ""

        # Inject
        self._messages.append(user_msg)
        self._accumulated_tokens.extend(user_tokens)
        self._accumulated_tokens.extend(final_tokens)

        logger.info(f"Extra round: budget={actual_budget}")

        # Generate
        step_start = time.time()
        self._streaming_tokens = []
        tokens, finish_reason = await self._stream_generate(actual_budget)

        if not tokens:
            return None, ""

        # Parse with forced final prefix
        parse_tokens = list(final_tokens) + tokens
        assistant_start = len(self._accumulated_tokens) - len(tokens)
        try:
            parsed = self._encoding.parse_messages_from_completion_tokens(
                parse_tokens, role=Role.ASSISTANT,
            )
        except Exception as exc:
            logger.warning(f"Extra round parse error: {exc}")
            self._steps.append(StepInfo(
                type="assistant", round_index=-1,
                token_start=assistant_start,
                token_end=len(self._accumulated_tokens),
                created_at=step_start, elapsed=time.time() - step_start,
            ))
            # If truncated, report token_limit even if parse fails
            if finish_reason == "length":
                return EndReason.TOKEN_LIMIT, "token_limit:forced_final"
            self._errors.append(f"Extra round parse error: {exc}")
            return None, ""

        self._messages.extend(parsed)
        last_msg = parsed[-1]
        self._steps.append(StepInfo(
            type="assistant", round_index=-1,
            token_start=assistant_start,
            token_end=len(self._accumulated_tokens),
            created_at=step_start, elapsed=time.time() - step_start,
            finish_reason=finish_reason,
            recipient=last_msg.recipient,
        ))

        if finish_reason == "length":
            logger.info(f"Extra round truncated, tokens={len(tokens)}")
            return EndReason.TOKEN_LIMIT, "token_limit:forced_final"

        logger.info(f"Extra round completed, tokens={len(tokens)}")
        return EndReason.COMPLETED, "completed:forced_final"

    # =========================================================================
    # Internal: Non-streaming one-shot generate (for probe)
    # =========================================================================

    async def _sglang_generate_oneshot(
        self,
        prompt_tokens: list[int],
        max_tokens: int,
        *,
        temperature: float | None = None,
    ) -> tuple[list[int], str | None]:
        """Non-streaming one-shot generate. Zero side effects on handle state."""
        payload: dict[str, Any] = {
            "input_ids": prompt_tokens,
            "sampling_params": {
                "max_new_tokens": max_tokens,
                "temperature": temperature if temperature is not None else self._config.temperature,
                "top_p": self._config.top_p,
                "top_k": self._config.top_k if self._config.top_k > 0 else -1,
                "stop_token_ids": self._stop_token_ids,
                **({"sampling_seed": self._config.seed} if self._config.seed is not None else {}),
            },
            "stream": False,
            "return_logprob": False,
            "return_routed_experts": False,
        }
        if self._config.min_p is not None:
            payload["sampling_params"]["min_p"] = self._config.min_p

        assert self._session is not None
        url = f"{self._config.server_url.rstrip('/').removesuffix('/v1')}/generate"

        async with self._session.post(url, json=payload) as resp:
            resp.raise_for_status()
            data = await resp.json()

        output_ids = data.get("output_ids", [])
        meta = data.get("meta_info", {})
        finish_reason = _normalize_finish_reason(meta.get("finish_reason"))
        return output_ids, finish_reason

    # =========================================================================
    # Internal: Interruptible tool call
    # =========================================================================

    async def _call_tool_interruptible(
        self, recipient: str, content: str,
    ) -> ToolReport:
        """Call tool with interrupt support (stop / force_extra_round).

        Races the tool task against ``_interrupt_event``.  On interrupt,
        cancels the tool task and returns a synthetic cancelled report.
        """
        self._interrupt_event.clear()
        tool_task = asyncio.create_task(self._call_tool(recipient, content))
        interrupt_task = asyncio.create_task(self._interrupt_event.wait())

        done, pending = await asyncio.wait(
            {tool_task, interrupt_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        for t in pending:
            t.cancel()
            with suppress(asyncio.CancelledError):
                await t

        if tool_task in done:
            return tool_task.result()

        return ToolReport(
            tool_name=recipient,
            tool_input=content,
            tool_output="[Tool call cancelled]",
            elapsed=0.0,
            error="cancelled",
        )

    # =========================================================================
    # Internal: Helpers
    # =========================================================================

    def _calculate_budget(self) -> _TokenBudget:
        cfg = self._config
        remaining_gen = cfg.max_total_tokens - self._total_generated_tokens
        remaining_ctx = cfg.max_context_tokens - len(self._accumulated_tokens)
        candidates = [
            (remaining_gen, "generation"),
            (remaining_ctx, "context"),
            (cfg.max_round_tokens, "round"),
        ]
        max_tokens, factor = min(candidates, key=lambda x: x[0])
        return _TokenBudget(max_tokens=max_tokens, limiting_factor=factor)

    def _apply_injections(self) -> None:
        """Process pending message injections."""
        while self._pending_injections:
            cmd = self._pending_injections.pop(0)
            role = Role.USER if cmd.role == "user" else Role.DEVELOPER
            msg = Message.from_role_and_content(role, cmd.content)
            self._messages.append(msg)
            tokens = self._encoding.render_conversation_for_completion(
                Conversation.from_messages([msg]),
                Role.ASSISTANT,
            )
            self._accumulated_tokens.extend(tokens)
            self._emit(EventType.INJECTION_APPLIED)
            logger.info(f"Injected {cmd.role} message ({len(tokens)} tokens)")

    def _emit(self, event_type: EventType, **kwargs: Any) -> None:
        event = Event(
            type=event_type,
            timestamp=time.time(),
            round_index=self._round_index,
            phase=self._phase,
            total_generated_tokens=self._total_generated_tokens,
            **kwargs,
        )
        self._event_queue.put_nowait(event)

    # =========================================================================
    # Internal: Conversation Building
    # =========================================================================

    def _build_conversation(
        self,
        user_prompt: str,
        system_prompt: str | None,
        dev_instructions: str | None,
        conversation_start_date: str | None,
        tool_configs: list[ToolNamespaceConfig],
    ) -> Conversation:
        cfg = self._config
        system_tools = [tc for tc in tool_configs if _is_system_namespace(tc.name)]
        dev_tools = [tc for tc in tool_configs if not _is_system_namespace(tc.name)]

        # System message
        system_msg = (
            SystemContent.new()
            .with_model_identity(system_prompt or cfg.model_identity)
            .with_conversation_start_date(conversation_start_date)
            .with_reasoning_effort(
                _REASONING_EFFORT_MAP.get(cfg.reasoning_effort.lower(), ReasoningEffort.MEDIUM)
            )
        )
        for tc in system_tools:
            harmony_ns = _to_harmony_namespace(tc.name)
            tools = [] if harmony_ns == "python" else tc.tools
            system_msg = system_msg.with_tools(ToolNamespaceConfig(
                name=harmony_ns, description=tc.description, tools=tools,
            ))

        messages: list[Message] = [
            Message.from_role_and_content(Role.SYSTEM, system_msg),
        ]

        # Developer message
        dev_content = DeveloperContent.new()
        if dev_instructions:
            dev_content = dev_content.with_instructions(dev_instructions)
        for tc in dev_tools:
            dev_content = dev_content.with_tools(tc)
        if dev_content.instructions or dev_content.tools:
            messages.append(Message.from_role_and_content(Role.DEVELOPER, dev_content))

        # User message
        messages.append(Message.from_role_and_content(Role.USER, user_prompt))

        return Conversation.from_messages(messages)

    # =========================================================================
    # Internal: sglang Session
    # =========================================================================

    async def _setup_session(self) -> None:
        connector = aiohttp.TCPConnector(limit=10, ttl_dns_cache=300, keepalive_timeout=30)
        timeout = aiohttp.ClientTimeout(total=self._config.http_timeout)
        self._session = aiohttp.ClientSession(connector=connector, timeout=timeout)

    # =========================================================================
    # Internal: MCP
    # =========================================================================

    async def _setup_mcp(self) -> None:
        if not self._mcp_config or "mcpServers" not in self._mcp_config:
            return
        for key, server_config in self._mcp_config["mcpServers"].items():
            single_config = {"mcpServers": {key: server_config}}
            client = McpClient(single_config)
            await client.__aenter__()
            self._mcp_clients[key] = client

    async def _discover_tools(self) -> list[ToolNamespaceConfig]:
        configs: list[ToolNamespaceConfig] = []
        for namespace, client in self._mcp_clients.items():
            raw_tools = await client.list_tools()
            tools: list[ToolDescription] = []
            for tool in raw_tools:
                schema = _trim_schema(tool.inputSchema)
                raw_string_param = _extract_raw_string_param(schema)
                self._tool_metadata[f"{namespace}.{tool.name}"] = _ToolMeta(
                    namespace=namespace,
                    name=tool.name,
                    raw_string_param=raw_string_param,
                    input_schema=schema,
                )
                tools.append(ToolDescription.new(
                    name=tool.name,
                    description=tool.description,
                    parameters=schema,
                ))
            # instructionsOverride from mcp_config takes precedence over server instructions
            server_cfg = (self._mcp_config or {}).get("mcpServers", {}).get(namespace, {})
            description = server_cfg.get("instructionsOverride") or client.initialize_result.instructions
            configs.append(ToolNamespaceConfig(
                name=namespace,
                description=description,
                tools=tools,
            ))
        return configs

    async def _call_tool(self, recipient: str, content: str) -> ToolReport:
        start = time.perf_counter()
        tool_name = recipient
        try:
            namespace, tool_name = self._resolve_tool_reference(recipient)
            raw_args = _parse_tool_content(content)
            args = self._prepare_tool_args(namespace, tool_name, raw_args)
            result = await self._mcp_clients[namespace].call_tool(
                tool_name,
                arguments=args,
                timeout=self._config.tool_call_timeout,
            )
            elapsed = time.perf_counter() - start
            text_parts = [
                block.text for block in result.content
                if getattr(block, "type", "") == "text"
            ]
            output = "\n".join(text_parts) if text_parts else "<no text output>"
            structured = result.structured_content or {}
            return ToolReport(
                tool_name=tool_name,
                tool_input=content,
                tool_output=output,
                elapsed=elapsed,
                early_exit=structured.get("early_exit", False),
            )
        except TimeoutError:
            return ToolReport(
                tool_name=tool_name, tool_input=content,
                tool_output="Tool execution timed out.",
                elapsed=time.perf_counter() - start, error="timeout",
            )
        except Exception as exc:
            return ToolReport(
                tool_name=tool_name, tool_input=content,
                tool_output=str(exc),
                elapsed=time.perf_counter() - start, error="exception",
            )

    def _resolve_tool_reference(self, reference: str) -> tuple[str, str]:
        if "." in reference:
            namespace, name = reference.split(".", 1)
            if namespace not in self._mcp_clients:
                raise ValueError(f"Unknown namespace: {namespace}")
            return namespace, name
        # Search system namespaces
        for ns in self._mcp_clients:
            if _is_system_namespace(ns) and ns.endswith(f"_{reference}"):
                return ns, reference
        raise ValueError(f"Unknown tool reference: {reference}")

    def _prepare_tool_args(
        self,
        namespace: str,
        tool_name: str,
        raw_args: dict[str, Any] | str | None,
    ) -> dict[str, Any]:
        if isinstance(raw_args, dict):
            return raw_args
        tool_key = f"{namespace}.{tool_name}"
        meta = self._tool_metadata.get(tool_key)
        if meta and meta.raw_string_param and isinstance(raw_args, str):
            return {meta.raw_string_param: raw_args}
        if isinstance(raw_args, str):
            try:
                return json.loads(raw_args)
            except json.JSONDecodeError:
                pass
            # Auto-infer single required string param
            if meta and meta.input_schema:
                props = meta.input_schema.get("properties", {})
                required = set(meta.input_schema.get("required", []))
                if len(required) == 1:
                    param = next(iter(required))
                    if props.get(param, {}).get("type") == "string":
                        return {param: raw_args}
            raise ValueError(
                f"Tool {tool_key}: non-JSON string and no single string param to auto-map"
            )
        return raw_args or {}

    # =========================================================================
    # Internal: Cleanup
    # =========================================================================

    async def _cleanup(self) -> None:
        for key, client in self._mcp_clients.items():
            try:
                await client.__aexit__(None, None, None)
            except Exception as e:
                logger.debug(f"MCP client {key} close error: {e}")
        self._mcp_clients.clear()

        if self._session is not None and not self._session.closed:
            connector = self._session.connector
            await self._session.close()
            if connector is not None:
                await connector.close()
            self._session = None


# =============================================================================
# Module-level helpers (no class dependency)
# =============================================================================


def _extract_text(message: Message) -> str:
    parts: list[str] = []
    for block in message.content:
        if hasattr(block, "text"):
            parts.append(block.text)
    return "".join(parts).strip()


def _trim_schema(schema: dict) -> dict:
    schema = dict(schema)
    schema.pop("title", None)
    if schema.get("default") is None:
        schema.pop("default", None)
    if "anyOf" in schema:
        types = [
            t["type"] for t in schema["anyOf"]
            if "type" in t and t["type"] != "null"
        ]
        if types:
            schema["type"] = types if len(types) > 1 else types[0]
        del schema["anyOf"]
    if "properties" in schema:
        schema["properties"] = {k: _trim_schema(v) for k, v in schema["properties"].items()}
    return schema


def _extract_raw_string_param(schema: dict) -> str | None:
    for name, prop in schema.get("properties", {}).items():
        if prop.get("rawStringParam"):
            return name
    return None


def _parse_tool_content(content: str) -> dict[str, Any] | str | None:
    if not content:
        return None
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return content
