"""Configuration classes for the rollout system."""

from typing import Dict, Literal, Optional

from pydantic import BaseModel, Field, model_validator


class SamplingParams(BaseModel):
    """LLM 取樣參數（集中管理，不再散佈在 LoopConfig 和每個 generate call）"""

    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: Optional[float] = None
    seed: Optional[int] = None


class ConversationConfig(BaseModel):
    """對話構建配置

    Builtin tools (python, browser) 自動從 mcp_config 檢測：
    - 有 builtin_python namespace → 自動啟用 python tool
    - 有 builtin_browser namespace → 自動啟用 browser tool
    """

    model_identity: str = "You are ChatGPT, a large language model trained by OpenAI."
    conversation_start_date: Optional[str] = None
    reasoning_effort: str = "medium"  # low / medium / high
    dev_instructions: Optional[str] = None


class SegmentTemperatureConfig(BaseModel):
    """分段 temperature 配置

    啟用後，生成會拆分成多個 segment，每個 segment 用不同 temperature：
    - Header 階段（到 <|message|>）：使用 header_temperature
    - Content 階段（到 <|end|>/<|call|>/<|return|>）：根據 channel/recipient 選擇

    優先順序：recipient_temperatures > channel_temperatures > default_content_temperature
    """

    header_temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=2.0,
        description="Header 階段的 temperature（低值確保格式穩定）",
    )

    default_content_temperature: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="Content 階段的預設 temperature",
    )

    channel_temperatures: Dict[str, float] = Field(
        default_factory=dict,
        description="按 channel 名稱設定 temperature，如 {'analysis': 1.0, 'final': 0.8}",
    )

    recipient_temperatures: Dict[str, float] = Field(
        default_factory=dict,
        description="按 recipient 設定 temperature（優先於 channel），如 {'python': 0.2}",
    )

    def get_content_temperature(
        self,
        channel: Optional[str],
        recipient: Optional[str],
    ) -> float:
        """根據 channel 和 recipient 決定 content temperature"""
        if recipient and recipient in self.recipient_temperatures:
            return self.recipient_temperatures[recipient]
        if channel and channel in self.channel_temperatures:
            return self.channel_temperatures[channel]
        return self.default_content_temperature


class ExtraRoundConfig(BaseModel):
    """TOKEN_LIMIT / MAX_ROUNDS 觸發時的強制收尾配置

    當 rollout 因 token 預算耗盡或 tool call 輪數用完而結束時，
    注入「時間到」訊息 + 強制 final channel，讓模型立刻輸出最終答案。
    """

    budget: int = Field(
        default=4096,
        gt=0,
        description="Extra round 最大生成 token 數（不受 max_total_tokens 限制，但受 max_context_tokens 約束）",
    )

    message: str = Field(
        default=(
            "[System notice: You have run out of budget (time/rounds). "
            "Provide your final answer immediately based on what you know so far. "
            "Do not use any more tools.]"
        ),
        description="注入的提示訊息（作為 user message）",
    )

    min_budget: int = Field(
        default=64,
        ge=0,
        description="注入訊息後 context 剩餘空間低於此值則放棄 extra round",
    )

    on_token_limit: bool = Field(
        default=True,
        description="TOKEN_LIMIT 時觸發",
    )

    on_max_rounds: bool = Field(
        default=True,
        description="MAX_ROUNDS 時觸發",
    )

    @model_validator(mode="after")
    def _validate(self) -> "ExtraRoundConfig":
        if self.min_budget > self.budget:
            raise ValueError(
                f"min_budget ({self.min_budget}) must be <= budget ({self.budget})"
            )
        if not self.on_token_limit and not self.on_max_rounds:
            raise ValueError(
                "At least one of on_token_limit/on_max_rounds must be True"
            )
        return self


class LoopConfig(BaseModel):
    """執行 loop 相關的配置（訓練/推理共用）"""

    # Backend 選擇
    # vllm path is preserved in code but not currently supported (validator raises).
    backend: Literal["vllm", "sglang"] = "sglang"

    # Model name（vLLM 的 OpenAI API 需要，sglang 不需要）
    model_name: Optional[str] = None

    # 取樣參數
    sampling: SamplingParams = Field(default_factory=SamplingParams)

    # Model config for routing reshape (sglang only)
    num_hidden_layers: Optional[int] = None      # e.g., 28 for Qwen-7B
    num_experts_per_tok: Optional[int] = None    # e.g., 8 for DeepSeek-V3

    # Loop 控制
    max_rounds: int = 10

    # Token 限制
    max_total_tokens: int = 80_000      # 總生成 token 上限
    max_round_tokens: int = 32_000      # 單輪生成上限
    max_context_tokens: int = 128_000   # Context window 大小

    # Tool
    tool_call_timeout: float = 60.0
    max_total_tool_time: float = 0.0    # 總 tool 執行時間上限（秒），0 = 不限制
    tool_error_strategy: Literal["continue", "stop"] = "continue"
    filter_by_include_in_prompt: bool = True

    # Harmony 特定
    auto_drop_analysis: bool = True
    harmony_custom_config_path: Optional[str] = None  # 自訂 tokenizer 配置（None = 內建 HarmonyGptOss）
    max_parse_retries: int = 0

    # 生成模式
    use_streaming: bool = True

    # Abort 限制（RL weight update 用）
    max_aborts: int = 20

    # Prefix cache isolation (sglang only). Set by GroupConfigFactory according to
    # RLFlow's cache_salt_mode — users should not write to this directly.
    cache_salt: Optional[str] = None

    # LoRA adapter name（sglang per-request lora_path）
    lora_adapter_name: Optional[str] = None

    # MCP server spawn rate limiting（跨 process file lock，防止 thundering herd）
    mcp_spawn_interval: float = 0.0  # 秒，全局每次 spawn 之間的最小間隔（0 = 不限制）

    # 分段 temperature（啟用後使用兩階段生成，不支援 logprobs/routing_indices）
    segment_temperature: Optional[SegmentTemperatureConfig] = None

    # Extra round（TOKEN_LIMIT / MAX_ROUNDS 時強制收尾）
    extra_round: Optional[ExtraRoundConfig] = None

    @model_validator(mode="after")
    def _check_parse_retry_abort_mutual_exclusion(self) -> "LoopConfig":
        if self.max_parse_retries > 0 and self.max_aborts > 0:
            raise ValueError(
                f"max_parse_retries ({self.max_parse_retries}) and "
                f"max_aborts ({self.max_aborts}) cannot both be > 0: "
                f"parse retry rollback does not handle abort partial state correctly"
            )
        return self

    @model_validator(mode="after")
    def _check_backend_supported(self) -> "LoopConfig":
        if self.backend == "vllm":
            raise NotImplementedError(
                "backend='vllm' is temporarily not supported. "
                "The VLLMBackend code is preserved but training/rollout has only been "
                "validated against sglang. Use backend='sglang'."
            )
        return self


class RecordConfig(BaseModel):
    """訓練記錄配置 - 推理時不傳（為 None）"""

    logprobs: bool = True
    top_logprobs: int = 0               # 記錄 top-k logprobs，0=不記錄
    routing_indices: bool = False
    usage: bool = True
    entropy: bool = False            # 記錄 per-token entropy，需 sglang return_entropy 支援
