"""Configuration and result types for parallel rollout execution."""

from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field

from ..config import ConversationConfig, LoopConfig, RecordConfig
from ..types import ReactResult


class RolloutConfig(BaseModel):
    """單個 rollout 的完整配置"""

    # 必填
    rollout_id: str
    user_prompt: str

    # 配置
    conv_config: ConversationConfig = Field(default_factory=ConversationConfig)
    loop_config: LoopConfig = Field(default_factory=LoopConfig)
    record_config: Optional[RecordConfig] = None  # None = 推理模式

    # MCP 配置（fastmcp 格式）
    mcp_config: Optional[Dict[str, Any]] = None

    # 擴展資料（用於 reward 計算、分析等）
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RolloutResult(BaseModel):
    """Rollout 執行結果"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # 標識
    rollout_id: str

    # 結果
    result: Optional[ReactResult] = None

    # 時間統計
    start_time: float
    end_time: float
    elapsed_time: float

    # 執行資訊
    server_url: str

    # 狀態
    success: bool
    error: Optional[str] = None
    traceback: Optional[str] = None

    # 預計算統計（由 postprocessor 填充）
    routing_valid: Optional[bool] = None
    completion_tokens_count: Optional[int] = None
    num_rounds: Optional[int] = None

    # Reward（由外部計算器填充）
    weighted_reward: Optional[float] = None
    reward_components: Optional[Dict[str, float]] = None

    @property
    def is_task_success(self) -> bool:
        return self.success and self.result is not None and self.result.is_success

    @property
    def end_reason(self) -> Optional[str]:
        if self.result is not None:
            return self.result.end_reason.value
        return None

    @property
    def num_generated_tokens(self) -> int:
        if self.result is not None:
            return self.result.num_generated_tokens
        return 0
