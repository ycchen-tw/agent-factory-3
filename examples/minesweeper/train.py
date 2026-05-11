"""Minesweeper demo — RL training of gpt-oss-120b LoRA on 8x8 Minesweeper (4× H100).

Recipe: DDIS loss, asymmetric trust region, LoRA r=8 on attention, routing
replay ON, DFlash speculative decoding, merged weight sync.

GPU layout (4× H100 80GB) — 3 sglang TP=1 + 1 GPU FSDP2:
    GPU 0     sglang TP=1, port 30100   — serve.sh 0 30100 1
    GPU 1     sglang TP=1, port 30101   — serve.sh 1 30101 1
    GPU 2     sglang TP=1, port 30102   — serve.sh 2 30102 1
    GPU 3     FSDP2 training (single rank, this script)

Quick start:
    bash examples/minesweeper/serve.sh 0 30100 1 &
    bash examples/minesweeper/serve.sh 1 30101 1 &
    bash examples/minesweeper/serve.sh 2 30102 1 &
    # wait for all three /health to return 200
    bash examples/minesweeper/run_train.sh
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
RUN_DIR = os.environ.get("RUN_DIR", str(EXAMPLE_DIR.parent.parent / "runs" / "minesweeper_demo"))

# sglang server endpoints (3 TP=1 servers by default).
SGLANG_URLS = os.environ.get(
    "SGLANG_URLS",
    "http://localhost:30100,http://localhost:30101,http://localhost:30102",
).split(",")

# Per-rollout MCP server: spawned as a subprocess with a fixed seed.
MINESWEEPER_SERVER = str(EXAMPLE_DIR / "mcp_tool" / "server.py")


# ── Game / dataset ───────────────────────────────────────────────────────

BOARD_WIDTH = 8
BOARD_HEIGHT = 8
NUM_MINES = 8
NUM_BOARDS = 1000        # number of distinct seeds → distinct boards


def make_dataset() -> list[dict]:
    """One item per seed; each item spawns a Minesweeper MCP server with that seed."""
    return [
        {
            "prompt": (
                f"Play Minesweeper on the {BOARD_WIDTH}×{BOARD_HEIGHT} board "
                f"({NUM_MINES} mines). This board is guaranteed solvable without guessing."
            ),
            "seed": seed,
            "mcp_config": {
                "mcpServers": {
                    "minesweeper": {
                        "command": sys.executable,
                        "args": [
                            MINESWEEPER_SERVER,
                            "--width", str(BOARD_WIDTH),
                            "--height", str(BOARD_HEIGHT),
                            "--mines", str(NUM_MINES),
                            "--seed", str(seed),
                            "--mode", "text",
                            "--no-banner",
                        ],
                    }
                }
            },
        }
        for seed in range(NUM_BOARDS)
    ]


# ── Reward ───────────────────────────────────────────────────────────────

def minesweeper_reward(result: RolloutResult, metadata: dict) -> float:
    """Binary reward: 1.0 if any tool response signals a win, else 0.0.

    Supports both server modes:
      - json mode → tool_output is `{"state": "won", ...}`
      - text mode → tool_output ends with `state: won | ...`
    """
    if result.result is None:
        return 0.0
    for step in result.result.get_tool_steps():
        out = step.tool_output or ""
        # text-mode marker is unambiguous (no quotes around "won")
        if "state: won" in out:
            return 1.0
        # json-mode fallback
        try:
            if json.loads(out).get("state") == "won":
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

        # Token packing (fits 4× H100 80GB at this rollout length).
        max_capacity=128000,

        # Routing replay: capture expert routing during rollout, replay during
        # training so train-time MoE matches inference-time MoE exactly.
        use_routing_replay=True,

        # Merged weight sync: merge LoRA into base, push full weights to sglang.
        weight_sync_mode="merged",
        flush_cache_on_sync=False,
        weight_sync_interval=1,
        min_weight_sync_secs=15.0,
        max_staleness=8,

        # Run.
        run_dir=RUN_DIR,
        save_checkpoint_every=10,
        save_latest_checkpoint=True,

        # Wandb (set use_wandb=False to disable).
        use_wandb=True,
        wandb_project="agent-factory-minesweeper",
        wandb_run_name="minesweeper-demo",
        wandb_notes=(
            f"gpt-oss-120b LoRA r=8 on Minesweeper {BOARD_WIDTH}x{BOARD_HEIGHT} "
            f"({NUM_MINES} mines, no-guess). 4× H100, DDIS 0.2/0.28."
        ),
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
        reward_fn=per_rollout(minesweeper_reward),
        prompt_key="prompt",

        # Orchestrator: 3× sglang TP=1, 16 workers × 8 concurrency = 128 in-flight.
        # batch=128 / group=32 → 4 groups per batch (more reward variance).
        server_urls=SGLANG_URLS,
        num_workers=16,
        worker_concurrency=8,
        batch_size=128,
        group_size=32,

        # Rollout: non-streaming so we can capture routing_indices.
        loop_config=LoopConfig(
            backend="sglang",
            sampling=SamplingParams(temperature=1.0),
            max_rounds=50,                # 8x8 board cleared in <20 actions; ample slack
            max_total_tokens=32000,
            max_round_tokens=32000,
            max_context_tokens=48000,
            use_streaming=False,          # routing_indices requires non-streaming
            max_aborts=10,
            num_hidden_layers=36,         # gpt-oss-120b has 36 layers
            num_experts_per_tok=4,        # gpt-oss-120b top-k=4
            tool_call_timeout=10.0,
            max_total_tool_time=180.0,
            mcp_spawn_interval=0.05,
        ),
        conv_config=ConversationConfig(
            reasoning_effort="medium",
            # No dev_instructions: goal, board symbols, coordinates, and tool
            # semantics already come from the MCP server's instructions/tool
            # descriptions. Per-task board dimensions and the no-guess
            # guarantee live in the user prompt above.
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

    logger.info("Starting RLFlow training (Minesweeper demo)")
    metrics = flow.run(on_step=on_step)
    logger.info(f"Training complete: {len(metrics)} steps")


if __name__ == "__main__":
    main()
