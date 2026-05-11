"""Wordle demo — RL training of gpt-oss-120b LoRA on English Wordle (4× H100).

Recipe: DDIS loss, symmetric trust region, LoRA r=8 on attention, routing
replay ON, DFlash speculative decoding, merged weight sync.

GPU layout (4× H100 80GB):
    GPU 0,1   sglang TP=2 (+ DFlash if DRAFT_MODEL_PATH is set)   — serve.sh
    GPU 2,3   FSDP2 training (this script via accelerate launch)  — run_train.sh

Quick start:
    bash examples/wordle/serve.sh        # Wait for "/health 200" before starting training
    bash examples/wordle/run_train.sh
"""

import json
import logging
import os
import sys
from pathlib import Path

# gpt-oss-120b model-specific support: import these BEFORE transformers loads
# the model so the FSDP2 BF16 fix and the mxfp4 HfQuantizer are in place.
from agent_factory_3.models.gpt_oss import (
    fsdp2_fix,            # noqa: F401  side-effect: patches torch._foreach_copy_
    quantizer_config,     # noqa: F401  side-effect: registers Mxfp4Bf16DequantQuantizer
    Mxfp4Bf16DequantConfig,
    enable_routing_replay,
)

import torch
from accelerate import Accelerator
from liger_kernel.transformers import apply_liger_kernel_to_gpt_oss
from transformers import AutoConfig, AutoModelForCausalLM

from agent_factory_3.orchestrator.types import per_rollout
from agent_factory_3.rollout import (
    ConversationConfig,
    LoopConfig,
    RecordConfig,
    SamplingParams,
)
from agent_factory_3.rollout.parallel.config import RolloutResult
from agent_factory_3.trainer.model_trainer import ModelTrainer
from agent_factory_3.trainer.rl_flow import RLFlow
from agent_factory_3.trainer.training_config import TrainingConfig

# Liger fused RMSNorm + RoPE (must be applied before model loading).
apply_liger_kernel_to_gpt_oss(
    rms_norm=True, rope=True,
    cross_entropy=False, fused_linear_cross_entropy=False, swiglu=False,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ── Paths ────────────────────────────────────────────────────────────────

EXAMPLE_DIR = Path(__file__).parent
MODEL_PATH = os.environ.get("MODEL_PATH", "/path/to/gpt-oss-120b")
RUN_DIR = os.environ.get("RUN_DIR", str(EXAMPLE_DIR.parent.parent / "runs" / "wordle_demo"))

# sglang server endpoints (one TP=2 server by default).
SGLANG_URLS = os.environ.get("SGLANG_URLS", "http://localhost:30100").split(",")

# Per-rollout MCP server: spawned as a subprocess with a fixed target word.
WORDLE_SERVER = str(EXAMPLE_DIR / "mcp_tool" / "server.py")
WORDLE_ANSWERS = EXAMPLE_DIR / "mcp_tool" / "data" / "answers.txt"


# ── Dataset ──────────────────────────────────────────────────────────────

def make_dataset() -> list[dict]:
    """One item per target word; each item spawns its own Wordle MCP server."""
    words = [w.strip() for w in WORDLE_ANSWERS.read_text().splitlines() if w.strip()]
    logger.info(f"Loaded {len(words)} Wordle target words")
    return [
        {
            "prompt": "Play Wordle. Guess the 5-letter English word.",
            "target_word": word,
            "mcp_config": {
                "mcpServers": {
                    "wordle": {
                        "command": sys.executable,
                        "args": [WORDLE_SERVER, "--target", word, "--no-banner"],
                    }
                }
            },
        }
        for word in words
    ]


# ── Reward ───────────────────────────────────────────────────────────────

def wordle_reward(result: RolloutResult, metadata: dict) -> float:
    """Binary reward: 1.0 if any guess returned won=True, else 0.0."""
    if result.result is None:
        return 0.0
    for step in result.result.get_tool_steps():
        try:
            if json.loads(step.tool_output).get("won"):
                return 1.0
        except (json.JSONDecodeError, TypeError, AttributeError):
            continue
    return 0.0


# ── Model loading ────────────────────────────────────────────────────────

def load_model():
    """gpt-oss-120b with mxfp4→BF16 dequant + Liger RMSNorm/RoPE + routing replay.

    No device_map: FSDP2 handles device placement during accelerator.prepare().
    """
    config = AutoConfig.from_pretrained(MODEL_PATH)
    config.quantization_config = Mxfp4Bf16DequantConfig(dequant_dtype="bf16").to_dict()

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        config=config,
        dtype=torch.bfloat16,
        attn_implementation="kernels-community/vllm-flash-attn3",
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    enable_routing_replay(model)
    return model


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    training_config = TrainingConfig(
        model_id=MODEL_PATH,

        # LoRA on attention only (MoE experts are mxfp4-frozen).
        training_mode="lora",
        lora_r=8,
        lora_alpha=32,
        lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],

        # Optimizer.
        learning_rate=1e-5,
        adam_beta1=0.9,
        adam_beta2=0.95,                # RL default; SFT typically uses 0.999
        max_grad_norm=1.0,
        lr_scheduler_type="constant_with_warmup",
        lr_warmup_steps=20,

        # DDIS — asymmetric trust region [0.8, 1.28] (slightly more room to step up).
        ddis_eps_low=0.2,
        ddis_eps_high=0.28,

        # Loss.
        loss_backend="liger",
        loss_aggregation_mode="seq-mean-token-mean",

        # Token packing (fits 4× H100 80GB with this context length).
        max_capacity=128000,

        # Routing replay: capture expert routing during rollout, replay during
        # training so train-time MoE matches inference-time MoE exactly.
        use_routing_replay=True,

        # Merged weight sync: merge LoRA into base, push full weights to sglang.
        # Simpler than per-adapter sync.
        weight_sync_mode="merged",
        flush_cache_on_sync=False,
        weight_sync_interval=1,
        min_weight_sync_secs=15.0,
        max_staleness=4,

        # Run.
        run_dir=RUN_DIR,
        save_checkpoint_every=10,
        save_latest_checkpoint=True,

        # Wandb (set use_wandb=False to disable).
        use_wandb=True,
        wandb_project="agent-factory-wordle",
        wandb_run_name="wordle-demo",
        wandb_notes="gpt-oss-120b LoRA r=8 on English Wordle, 4× H100, DDIS 0.2/0.28.",
    )

    accelerator = Accelerator()
    logger.info(
        f"Accelerator: {accelerator.state.distributed_type}, "
        f"{accelerator.num_processes} processes"
    )

    logger.info("Loading model (mxfp4→bf16 dequant + Liger + routing replay)...")
    model = load_model()
    model_trainer = ModelTrainer(config=training_config, accelerator=accelerator, model=model)

    flow = RLFlow(
        training_config=training_config,
        model_trainer=model_trainer,

        # Dataset.
        dataset=make_dataset(),
        reward_fn=per_rollout(wordle_reward),
        prompt_key="prompt",

        # Orchestrator: 1 sglang TP=2 server, 12 workers × 8 concurrency = 96 in-flight.
        server_urls=SGLANG_URLS,
        num_workers=12,
        worker_concurrency=8,
        batch_size=128,                 # rollouts per training batch (must be a multiple of group_size)
        group_size=16,                  # rollouts per Wordle word (DDIS computes group-relative advantage)

        # Rollout: non-streaming so we can capture routing_indices.
        loop_config=LoopConfig(
            backend="sglang",
            sampling=SamplingParams(temperature=1.0),
            max_rounds=8,               # at most 8 turns (≥ 6 wordle attempts + a couple of slack rounds)
            max_total_tokens=10000,
            max_round_tokens=10000,
            max_context_tokens=20000,
            use_streaming=False,        # routing_indices requires non-streaming
            max_aborts=10,
            num_hidden_layers=36,       # gpt-oss-120b has 36 layers (TODO: derive from model config)
            num_experts_per_tok=4,      # gpt-oss-120b top-k=4         (TODO: derive from model config)
            tool_call_timeout=10.0,
            max_total_tool_time=120.0,
            mcp_spawn_interval=0.05,
        ),
        conv_config=ConversationConfig(
            reasoning_effort="low",
            dev_instructions=(
                "You are playing Wordle. Use the 'guess' tool to guess a 5-letter English word. "
                "Each guess returns per-letter feedback (correct / present / absent). "
                "You have 6 attempts."
            ),
        ),
        record_config=RecordConfig(logprobs=True, routing_indices=True),

        # Drop unsignaled groups (all-pass / all-fail give zero advantage to DDIS).
        filter_all_failed=True,
        filter_all_solved=True,

        # Each rollout gets a unique prefix-cache salt → fresh sampling per rollout.
        cache_salt_mode="per_rollout",

        max_batches=150,
        base_seed=42,
    )

    def on_step(metrics: dict, step: int):
        loss = metrics.get("loss", float("nan"))
        grad = metrics.get("grad_norm", float("nan"))
        mask = metrics.get("algo/ddis_mask_rate", float("nan"))
        wv = metrics.get("flow/weight_version", 0)
        logger.info(
            f"Step {step}: loss={loss:.4f} grad_norm={grad:.4f} "
            f"mask_rate={mask:.2%} weight_version={wv:.0f}"
        )

    logger.info("Starting RLFlow training (Wordle demo)")
    metrics = flow.run(on_step=on_step)
    logger.info(f"Training complete: {len(metrics)} steps")


if __name__ == "__main__":
    main()
