"""Rollout system — ReAct loop + parallel execution."""

# Config
from .config import (
    ConversationConfig,
    ExtraRoundConfig,
    LoopConfig,
    RecordConfig,
    SamplingParams,
    SegmentTemperatureConfig,
)

# Types
from .types import (
    EndReason,
    StepType,
    TruncationReason,
    TokenBudget,
    SegmentMeta,
    BaseStep,
    InitStep,
    AssistantStep,
    ToolStep,
    Step,
    ReactResult,
)

# MCP
from .mcp_executor import McpExecutor, ToolCallReport, ToolErrorType

# Builder
from .conversation_builder import ConversationBuilder

# Runner
from .runner import UnifiedReactRunner, CompletionResult

# Parallel execution
from .parallel import RolloutConfig, RolloutResult, execute_rollout

__all__ = [
    # Config
    "ConversationConfig",
    "ExtraRoundConfig",
    "LoopConfig",
    "RecordConfig",
    "SamplingParams",
    "SegmentTemperatureConfig",
    # Types
    "EndReason",
    "StepType",
    "TruncationReason",
    "TokenBudget",
    "SegmentMeta",
    "BaseStep",
    "InitStep",
    "AssistantStep",
    "ToolStep",
    "Step",
    "ReactResult",
    # MCP
    "McpExecutor",
    "ToolCallReport",
    "ToolErrorType",
    # Builder
    "ConversationBuilder",
    # Runner
    "UnifiedReactRunner",
    "CompletionResult",
    # Parallel
    "RolloutConfig",
    "RolloutResult",
    "execute_rollout",
]
