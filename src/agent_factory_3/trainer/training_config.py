"""Training configuration for v3 DDIS trainer."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, computed_field, model_validator

LRSchedulerType = Literal[
    "linear",
    "cosine",
    "cosine_with_restarts",
    "polynomial",
    "constant",
    "constant_with_warmup",
    "inverse_sqrt",
]


class TrainingConfig(BaseModel):
    """Training-only configuration. Orchestrator/rollout config is separate."""

    # ========== Model ==========
    model_id: str
    attn_implementation: str = "flex_attention"
    gradient_checkpointing: bool = True
    use_reentrant: bool = False
    model_use_cache: bool = False
    model_init_kwargs: dict | None = None

    # ========== Training Mode ==========
    training_mode: str = "lora"  # "lora" = LoRA adapter; "full" = full-param / selective module
    trainable_modules: list[str] | None = None  # full mode only: None=all params, list=selective

    # ========== LoRA (training_mode="lora" only) ==========
    lora_r: int = 8
    lora_alpha: int = 32
    lora_target_modules: list[str] = ["q_proj", "k_proj", "v_proj", "o_proj"]
    lora_dropout: float = 0.0
    autocast_adapter_dtype: bool = True
    lora_kwargs: dict | None = None

    # ========== Optimizer ==========
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    optimizer_type: str = "AdamW"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    lr_scheduler_type: LRSchedulerType = "constant_with_warmup"
    lr_warmup_steps: int = 20

    # ========== DDIS ==========
    ddis_eps_low: float = 0.2    # ratio < 1 - eps_low → mask
    ddis_eps_high: float = 0.2   # ratio > 1 + eps_high → mask

    # ========== OPSM (Off-Policy Sequence Masking) ==========
    use_opsm: bool = False       # 開關，預設不啟用
    opsm_delta: float = 1e-4     # KL 閾值，negative seq 的 KL > delta 則整條 mask 掉

    # ========== Entropy Token Masking ==========
    entropy_top_pct: float | None = None  # 只訓練 top x% entropy tokens，None = 不啟用

    # ========== Loss ==========
    loss_temperature: float = 1.0
    loss_backend: str = "triton"
    loss_aggregation_mode: str = "seq-mean-token-sum"

    # ========== Multi-Step ==========
    num_inner_steps: int = 1  # split rollout batch into K mini-batches, each gets its own optimizer step

    # ========== Packing ==========
    max_capacity: int = 20000
    pad_to_multiple_of_training: int = 1024
    dataloader_num_workers: int = 0

    # ========== Routing ==========
    use_routing_replay: bool = True
    save_routing_indices: bool | None = None  # None = follow use_routing_replay

    # ========== Weight Sync ==========
    weight_sync_interval: int = 1       # sync every N batches (not optimizer steps)
    min_weight_sync_secs: float = 0.0   # minimum seconds between syncs (0 = no throttle)
    weight_sync_mode: str = "lora"      # "lora" = adapter sync; "merged" = base weight update
    flush_cache_on_sync: bool = True    # merged mode only; lora mode always flushes
    flush_cache_every_n_steps: int = 0  # 0=no periodic flush; N>0=flush every N batches; merged+no_flush only
    reset_optimizer_on_sync: bool = False  # with num_inner_steps>1, momentum from all inner steps is discarded
    lora_adapter_name: str = "policy"

    # ========== Staleness ==========
    max_staleness: int = 2  # drop samples N+ batches behind (uses batch_idx as weight_version)

    # ========== Paths ==========
    run_dir: str = "./runs/default"
    save_checkpoint_every: int = 5      # checkpoint every N batches (0=disabled)
    save_latest_checkpoint: bool = True  # always overwrite checkpoint_latest every step
    resume_from_checkpoint: str | None = None   # checkpoint path, or "latest"
    max_checkpoints_to_keep: int = 0            # keep last N checkpoints, 0 = keep all

    # ========== Wandb ==========
    use_wandb: bool = True
    wandb_project: str = "agent-factory-3"
    wandb_run_name: str | None = None
    wandb_tags: list[str] = []
    wandb_notes: str = ""
    wandb_resume: bool = True

    # ========== Derived paths (from run_dir) ==========

    @computed_field
    @property
    def checkpoint_dir(self) -> str:
        return str(Path(self.run_dir) / "checkpoints")

    @computed_field
    @property
    def rollout_dir(self) -> str:
        return str(Path(self.run_dir) / "rollouts")

    @computed_field
    @property
    def log_dir(self) -> str:
        return str(Path(self.run_dir) / "logs")

    @computed_field
    @property
    def weight_sync_dir(self) -> str:
        return str(Path(self.run_dir) / "weight_sync")

    @computed_field
    @property
    def weight_init_dir(self) -> str:
        return str(Path(self.run_dir) / "weight_init")

    @computed_field
    @property
    def save_routing_indices_resolved(self) -> bool:
        """Resolve the three-state save_routing_indices: None → follow use_routing_replay."""
        if self.save_routing_indices is None:
            return self.use_routing_replay
        return self.save_routing_indices

    @model_validator(mode="after")
    def _validate(self) -> "TrainingConfig":
        assert self.training_mode in ("lora", "full"), \
            f"training_mode must be 'lora' or 'full', got {self.training_mode}"
        if self.training_mode == "full":
            assert self.weight_sync_mode == "merged", \
                f"training_mode='full' requires weight_sync_mode='merged', got {self.weight_sync_mode}"
        assert self.num_inner_steps >= 1, \
            f"num_inner_steps must be >= 1, got {self.num_inner_steps}"
        assert self.max_capacity > 0
        assert self.learning_rate > 0
        # DDIS trust region: ratio < 1-eps_low or ratio > 1+eps_high gets masked.
        # eps must be in [0, 1); eps >= 1 makes the corresponding side vacuous
        # (since rollout-policy ratio is always positive).
        assert 0 <= self.ddis_eps_low < 1, \
            f"ddis_eps_low must be in [0, 1), got {self.ddis_eps_low}"
        assert 0 <= self.ddis_eps_high < 1, \
            f"ddis_eps_high must be in [0, 1), got {self.ddis_eps_high}"
        if self.entropy_top_pct is not None:
            assert 0 < self.entropy_top_pct <= 100, \
                f"entropy_top_pct must be in (0, 100], got {self.entropy_top_pct}"
        assert self.weight_sync_interval >= 1
        assert self.weight_sync_mode in ("lora", "merged"), \
            f"weight_sync_mode must be 'lora' or 'merged', got {self.weight_sync_mode}"
        assert self.max_staleness >= 0
        assert self.flush_cache_every_n_steps >= 0
        # Periodic flush only makes sense in merged mode without per-sync flush
        # (lora mode always flushes; merged + flush_on_sync already flushes every sync).
        if self.flush_cache_every_n_steps > 0:
            assert self.weight_sync_mode == "merged", (
                f"flush_cache_every_n_steps={self.flush_cache_every_n_steps} requires "
                f"weight_sync_mode='merged', got {self.weight_sync_mode!r}"
            )
            assert not self.flush_cache_on_sync, (
                f"flush_cache_every_n_steps={self.flush_cache_every_n_steps} is "
                "redundant when flush_cache_on_sync=True (which already flushes every sync)"
            )
        # Optimizer reset only matters when there are inner steps to accumulate state
        # across — with num_inner_steps=1 the reset just discards a single-step state.
        if self.reset_optimizer_on_sync:
            assert self.num_inner_steps > 1, (
                "reset_optimizer_on_sync=True requires num_inner_steps>1; "
                f"got num_inner_steps={self.num_inner_steps}."
            )
        valid_modes = ["token-mean", "seq-mean-token-sum", "seq-mean-token-mean", "prompt-mean-token-mean"]
        assert self.loss_aggregation_mode in valid_modes, \
            f"loss_aggregation_mode must be one of {valid_modes}, got {self.loss_aggregation_mode}"
        assert self.loss_backend in ["triton", "torch", "cce", "liger"], \
            f"loss_backend must be 'triton', 'torch', 'cce', or 'liger', got {self.loss_backend}"
        return self
