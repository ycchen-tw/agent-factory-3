"""Type definitions for the rollout system."""

import base64
import json
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

import numpy as np
from pydantic import BaseModel, Field, field_serializer, field_validator

from openai_harmony import Conversation

from .mcp_executor import ToolErrorType


@lru_cache(maxsize=1)
def _harmony_encoding():
    from openai_harmony import HarmonyEncodingName, load_harmony_encoding
    return load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)


# =============================================================================
# Enums
# =============================================================================


class EndReason(str, Enum):
    """結束原因（6 個粗分類）"""

    COMPLETED = "completed"
    TOOL_EARLY_EXIT = "tool_early_exit"
    MAX_ROUNDS = "max_rounds"
    TOKEN_LIMIT = "token_limit"
    INTERRUPTED = "interrupted"
    ERROR = "error"

    @property
    def is_success(self) -> bool:
        return self in (
            EndReason.COMPLETED,
            EndReason.TOOL_EARLY_EXIT,
            EndReason.MAX_ROUNDS,
        )


class StepType(str, Enum):
    INIT = "init"
    ASSISTANT = "assistant"
    TOOL = "tool"


class TruncationReason(str, Enum):
    """截斷原因 - 記錄 max_tokens 被哪個限制決定"""
    ROUND_LIMIT = "round_limit"
    CONTEXT_SPACE = "context_space"
    GENERATION_QUOTA = "generation_quota"


# =============================================================================
# TokenBudget
# =============================================================================


@dataclass
class TokenBudget:
    """Token 預算計算結果"""
    max_tokens: int
    limiting_factor: TruncationReason

    @property
    def can_generate(self) -> bool:
        return self.max_tokens > 0


# =============================================================================
# Segment Metadata (for segment-aware temperature)
# =============================================================================


class SegmentMeta(BaseModel):
    """單個生成段落的元數據（segment-aware temperature 用）

    token 位置是相對於 AssistantStep.start 的偏移量。
    """

    phase: Literal["header", "content"]
    start: int  # 相對於 AssistantStep.start 的起始偏移
    end: int    # 相對於 AssistantStep.start 的結束偏移（exclusive）
    temperature: float

    channel: Optional[str] = None
    recipient: Optional[str] = None


class WeightSegment(BaseModel):
    """一段由同一 weight version 生成的 tokens。

    用於 RL 訓練中 abort/resume 場景：一個 AssistantStep 可能跨越多次
    weight update，每段 tokens 由不同版本的 weights 生成。

    token 位置是相對於 AssistantStep.start 的偏移量。
    """

    start: int          # 相對於 AssistantStep.start 的起始偏移
    end: int            # 相對於 AssistantStep.start 的結束偏移（exclusive）
    weight_version: str


class AbortRecord(BaseModel):
    """一次 sglang abort 事件的記錄。

    當 sglang server 執行 pause_generation(mode="abort") 時，
    in-flight 的 generate request 會返回 finish_reason="abort" 和部分結果。
    """

    round_index: int
    weight_version: Optional[str]
    partial_token_count: int
    timestamp: float


# =============================================================================
# Step Types (Discriminated Union)
# =============================================================================


class BaseStep(BaseModel):
    """所有 Step 的基類"""
    type: StepType
    start: int                          # token 起始位置（inclusive）
    end: int                            # token 結束位置（exclusive）
    round_index: Optional[int] = None

    # Message 索引（對應 conversation.messages）
    message_start: Optional[int] = None
    message_end: Optional[int] = None

    # 時間資訊
    created_at: Optional[float] = None
    elapsed: Optional[float] = None

    @property
    def length(self) -> int:
        return self.end - self.start

    @property
    def message_count(self) -> int:
        if self.message_start is None or self.message_end is None:
            return 0
        return self.message_end - self.message_start


class InitStep(BaseStep):
    """初始 prompt (system + developer + user)"""
    type: Literal[StepType.INIT] = StepType.INIT

    num_system_tokens: Optional[int] = None
    num_developer_tokens: Optional[int] = None
    num_user_tokens: Optional[int] = None


class AssistantStep(BaseStep):
    """Assistant 生成"""
    type: Literal[StepType.ASSISTANT] = StepType.ASSISTANT

    stop_reason: Optional[str] = None
    truncation_reason: Optional[TruncationReason] = None

    recipient: Optional[str] = None
    tool_input: Optional[str] = None

    usage: Optional[Dict[str, Any]] = None

    # Segment metadata（segment-aware temperature 用）
    segments: List[SegmentMeta] = Field(default_factory=list)

    # Weight version tracking（RL abort/resume 用）
    weight_segments: List[WeightSegment] = Field(default_factory=list)

    # Set by runner when harmony parsing fails on this step's tokens.
    # Carries the exception text. None means parse succeeded (or wasn't attempted).
    # When set, message_start == message_end (no parsed message exists).
    parse_error: Optional[str] = None


class ToolStep(BaseStep):
    """Tool output"""
    type: Literal[StepType.TOOL] = StepType.TOOL

    tool_name: str
    tool_input: str
    tool_output: str

    error: Optional[ToolErrorType] = None
    early_exit: bool = False

    structured_output: Optional[Dict[str, Any]] = None


# Discriminated Union
Step = Annotated[
    Union[InitStep, AssistantStep, ToolStep],
    Field(discriminator="type")
]


# =============================================================================
# ReactResult
# =============================================================================


class ReactResult(BaseModel):
    """ReAct loop 的完整結果"""

    # ===== 結束狀態 =====
    end_reason: EndReason
    end_reason_detail: str
    errors: List[str] = Field(default_factory=list)

    # ===== Token 序列（訓練核心）=====
    tokens: List[int]
    logprobs: Optional[List[Optional[float]]] = None
    entropy: Optional[List[Optional[float]]] = None
    top_logprobs: Optional[List[Optional[Dict[int, float]]]] = None

    # ===== MoE routing indices（全 token）=====
    # List of per-token routing: each element is np.ndarray[L,K] uint8 or None.
    # None 位置表示該 token 被 prefix cache 命中，未經 forward pass.
    # Type is Any to prevent Pydantic v2 from coercing numpy arrays to nested lists.
    routing_indices: Optional[List[Optional[Any]]] = None

    # ----- routing_indices 序列化/反序列化（base64 uint8, sentinel=255 for None）-----
    @field_serializer('routing_indices')
    def _serialize_routing_indices(
        self, v: Optional[List[Optional[List[List[int]]]]]
    ) -> Optional[Dict[str, Any]]:
        if v is None:
            return None
        # 取第一個非 None 元素推斷 shape
        sample = next((x for x in v if x is not None), None)
        if sample is None:
            return None
        num_layers = len(sample)
        num_topk = len(sample[0])
        sentinel = 255
        arr = np.full((len(v), num_layers, num_topk), sentinel, dtype=np.uint8)
        for i, ri in enumerate(v):
            if ri is not None:
                arr[i] = ri
        return {
            "shape": list(arr.shape),
            "dtype": "uint8",
            "sentinel": sentinel,
            "data": base64.b64encode(arr.tobytes()).decode(),
        }

    @field_validator('routing_indices', mode='before')
    @classmethod
    def _deserialize_routing_indices(
        cls, v: Any
    ) -> Optional[List[Optional[List[List[int]]]]]:
        if v is None:
            return None
        if isinstance(v, dict) and "data" in v:
            arr = np.frombuffer(base64.b64decode(v["data"]), dtype=np.uint8)
            arr = arr.reshape(v["shape"])
            sentinel = v.get("sentinel", 255)
            result = []
            for row in arr:
                if row[0, 0] == sentinel:
                    result.append(None)
                else:
                    result.append(row.tolist())
            return result
        return v

    # ----- logprobs 序列化/反序列化（base64 float16 壓縮）-----
    @field_serializer('logprobs')
    def _serialize_logprobs(
        self, v: Optional[List[Optional[float]]]
    ) -> Optional[Dict[str, Any]]:
        if v is None:
            return None
        arr = np.array([x if x is not None else np.nan for x in v], dtype=np.float16)
        return {
            "dtype": "float16",
            "data": base64.b64encode(arr.tobytes()).decode(),
        }

    @field_validator('logprobs', mode='before')
    @classmethod
    def _deserialize_logprobs(
        cls, v: Any
    ) -> Optional[List[Optional[float]]]:
        if v is None:
            return None
        if isinstance(v, dict) and "data" in v:
            arr = np.frombuffer(base64.b64decode(v["data"]), dtype=np.float16)
            return [None if np.isnan(x) else float(x) for x in arr]
        return v

    # ----- entropy 序列化/反序列化（base64 float16，和 logprobs 同 pattern）-----
    @field_serializer('entropy')
    def _serialize_entropy(
        self, v: Optional[List[Optional[float]]]
    ) -> Optional[Dict[str, Any]]:
        if v is None:
            return None
        arr = np.array([x if x is not None else np.nan for x in v], dtype=np.float16)
        return {
            "dtype": "float16",
            "data": base64.b64encode(arr.tobytes()).decode(),
        }

    @field_validator('entropy', mode='before')
    @classmethod
    def _deserialize_entropy(
        cls, v: Any
    ) -> Optional[List[Optional[float]]]:
        if v is None:
            return None
        if isinstance(v, dict) and "data" in v:
            arr = np.frombuffer(base64.b64decode(v["data"]), dtype=np.float16)
            return [None if np.isnan(x) else float(x) for x in arr]
        return v

    # ===== Step 索引 =====
    steps: List[Step]

    # ===== 對話結果（推理用）=====
    conversation: Conversation

    # ----- Conversation 序列化/反序列化 -----
    @field_serializer('conversation')
    def _serialize_conversation(self, conv: Conversation) -> Dict[str, Any]:
        return conv.to_dict()

    @field_validator('conversation', mode='before')
    @classmethod
    def _deserialize_conversation(cls, v: Any) -> Conversation:
        if isinstance(v, dict):
            return Conversation.from_json(json.dumps(v))
        return v

    # ===== 匯總統計 =====
    total_tool_time: float
    num_generated_tokens: int

    # ===== Abort 記錄（RL abort/resume 用）=====
    abort_records: List[AbortRecord] = Field(default_factory=list)

    # ===== 狀態判斷屬性 =====

    @property
    def is_success(self) -> bool:
        return self.end_reason.is_success

    @property
    def has_tool_errors(self) -> bool:
        return any(
            isinstance(s, ToolStep) and s.error is not None
            for s in self.steps
        )

    @property
    def tool_error_count(self) -> int:
        return sum(
            1 for s in self.steps
            if isinstance(s, ToolStep) and s.error is not None
        )

    # ===== 便利方法 =====

    def get_step_tokens(self, step: BaseStep) -> List[int]:
        return self.tokens[step.start:step.end]

    def get_step_text(self, step: BaseStep) -> str:
        """Decode this step's tokens to text using the harmony encoding."""
        return _harmony_encoding().decode_utf8(self.tokens[step.start:step.end])

    def get_step_logprobs(self, step: BaseStep) -> Optional[List[Optional[float]]]:
        if self.logprobs is None:
            return None
        return self.logprobs[step.start:step.end]

    def get_step_routing(self, step: BaseStep) -> Optional[List[Optional[List[List[int]]]]]:
        if self.routing_indices is None:
            return None
        return self.routing_indices[step.start:step.end]

    def get_step_messages(self, step: BaseStep) -> List:
        if step.message_start is None or step.message_end is None:
            return []
        return list(self.conversation.messages[step.message_start:step.message_end])

    def get_assistant_steps(self) -> List[AssistantStep]:
        return [s for s in self.steps if s.type == StepType.ASSISTANT]

    def get_tool_steps(self) -> List[ToolStep]:
        return [s for s in self.steps if s.type == StepType.TOOL]

    def get_loss_mask(self) -> List[bool]:
        """生成 loss mask（只有 assistant 部分為 True）"""
        mask = [False] * len(self.tokens)
        for step in self.steps:
            if step.type == StepType.ASSISTANT:
                for i in range(step.start, step.end):
                    mask[i] = True
        return mask

    def get_trainable_data(
        self,
        logprob_fill_value: float = 0.0,
    ) -> tuple[List[int], List[float], List[bool]]:
        """取得訓練資料

        Returns:
            tokens, logprobs, completion_mask
        """
        if self.logprobs is None:
            raise ValueError("No logprobs recorded - was RecordConfig.logprobs=True?")

        tokens: List[int] = []
        logprobs: List[float] = []
        completion_mask: List[bool] = []

        for step in self.get_assistant_steps():
            step_tokens = self.tokens[step.start:step.end]
            step_logprobs = self.logprobs[step.start:step.end]

            tokens.extend(step_tokens)
            for lp in step_logprobs:
                if lp is not None:
                    logprobs.append(lp)
                    completion_mask.append(True)
                else:
                    logprobs.append(logprob_fill_value)
                    completion_mask.append(False)

        assert len(tokens) == len(logprobs) == len(completion_mask)
        return tokens, logprobs, completion_mask
