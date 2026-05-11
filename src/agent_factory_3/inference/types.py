"""Type definitions for the inference agent system.

Created: 2026-03-28
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from openai_harmony import Conversation


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class AgentConfig:
    """Agent configuration — sglang-only, streaming-only."""

    server_url: str

    # Generation limits
    max_rounds: int = 10
    max_total_tokens: int = 80_000
    max_round_tokens: int = 32_000
    max_context_tokens: int = 128_000

    # Sampling
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float | None = None
    seed: int | None = None

    # Tool
    tool_call_timeout: float = 60.0
    max_total_tool_time: float = 0.0  # 0 = unlimited

    # Harmony
    auto_drop_analysis: bool = True
    harmony_custom_config_path: str | None = None

    # Extra round defaults (used when no explicit command overrides)
    extra_round_budget: int = 4096
    extra_round_message: str = (
        "[System notice: You have run out of budget (time/rounds). "
        "Provide your final answer immediately based on what you know so far. "
        "Do not use any more tools.]"
    )

    # Conversation building
    model_identity: str = "You are ChatGPT, a large language model trained by OpenAI."
    reasoning_effort: str = "medium"

    # sglang server config
    stream_output: bool = True  # server --stream-output flag (incremental vs cumulative)
    http_timeout: float = 3600.0


# =============================================================================
# Phase & Event Types
# =============================================================================


class Phase(str, Enum):
    IDLE = "idle"
    GENERATING = "generating"
    CALLING_TOOL = "calling_tool"
    PAUSED = "paused"
    EXTRA_ROUND = "extra_round"
    DORMANT = "dormant"
    COMPLETED = "completed"
    FAILED = "failed"


class EventType(str, Enum):
    STARTED = "started"
    ROUND_START = "round_start"
    STREAM_CHUNK = "stream_chunk"
    GENERATION_DONE = "generation_done"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_DONE = "tool_call_done"
    ROUND_DONE = "round_done"
    INJECTION_APPLIED = "injection_applied"
    PAUSED = "paused"
    RESUMED = "resumed"
    EXTRA_ROUND_START = "extra_round_start"
    COMPLETED = "completed"
    ERROR = "error"


class EndReason(str, Enum):
    COMPLETED = "completed"
    TOOL_EARLY_EXIT = "tool_early_exit"
    MAX_ROUNDS = "max_rounds"
    TOKEN_LIMIT = "token_limit"
    INTERRUPTED = "interrupted"
    ERROR = "error"

    @property
    def is_success(self) -> bool:
        return self in (EndReason.COMPLETED, EndReason.TOOL_EARLY_EXIT, EndReason.MAX_ROUNDS)


# =============================================================================
# Commands (frozen — safe to share across tasks)
# =============================================================================


@dataclass(frozen=True)
class StopCommand:
    pass


@dataclass(frozen=True)
class PauseCommand:
    pass


@dataclass(frozen=True)
class ResumeCommand:
    pass


DEFAULT_ASSISTANT_PREFIX = "<|channel|>final"


@dataclass(frozen=True)
class ForceExtraRoundCommand:
    message: str | None = None  # None = use config default
    budget: int | None = None  # None = use config default
    assistant_prefix: str = DEFAULT_ASSISTANT_PREFIX


@dataclass(frozen=True)
class InjectMessageCommand:
    content: str
    role: Literal["user", "developer"] = "user"


Command = StopCommand | PauseCommand | ResumeCommand | ForceExtraRoundCommand | InjectMessageCommand


# =============================================================================
# Tool Report
# =============================================================================


@dataclass
class ToolReport:
    """Result of a single tool call."""

    tool_name: str
    tool_input: str
    tool_output: str
    elapsed: float
    error: str | None = None  # "timeout" | "exception" | None
    early_exit: bool = False


# =============================================================================
# Step Info
# =============================================================================


@dataclass
class StepInfo:
    """Simplified step info for inference (no training fields)."""

    type: Literal["init", "assistant", "tool"]
    round_index: int
    token_start: int
    token_end: int
    created_at: float
    elapsed: float = 0.0
    # Assistant-specific
    finish_reason: str | None = None
    recipient: str | None = None
    # Tool-specific
    tool_name: str | None = None
    tool_input: str | None = None
    tool_output: str | None = None
    tool_error: str | None = None
    early_exit: bool = False


# =============================================================================
# Event
# =============================================================================


@dataclass
class Event:
    """Progress event emitted by the agent."""

    type: EventType
    timestamp: float
    round_index: int
    phase: Phase
    total_generated_tokens: int = 0
    # Stream payload
    chunk_tokens: list[int] | None = None
    chunk_text: str | None = None
    # Tool payload
    tool_name: str | None = None
    tool_output: str | None = None
    tool_error: str | None = None
    # Completion payload
    end_reason: str | None = None
    end_reason_detail: str | None = None
    error: str | None = None
    result: AgentResult | None = None


# =============================================================================
# Result
# =============================================================================


@dataclass
class AgentResult:
    """Final result of an agent run (no training fields)."""

    end_reason: EndReason
    end_reason_detail: str
    tokens: list[int]
    conversation: Conversation
    steps: list[StepInfo]
    num_generated_tokens: int
    total_tool_time: float
    errors: list[str] = field(default_factory=list)
