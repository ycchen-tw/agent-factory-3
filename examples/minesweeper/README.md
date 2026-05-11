# Minesweeper demo — gpt-oss-120b LoRA on 4× H100

End-to-end RL recipe: train **gpt-oss-120b** (LoRA r=8, attention only) to play
8×8 Minesweeper with **DDIS** loss and routing replay, on a single 4× H100 box.
The boards are generated in *no-guess* mode, so every board is solvable by pure
logic — the model gets a clean signal from logical deduction alone.

> *Created 2026-05-07.*

---

## What's in here

```
minesweeper/
├── train.py                 # training entry: TrainingConfig + RLFlow + reward + dataset
├── serve.sh                 # launches sglang TP=2 server (with optional DFlash spec decoding)
├── run_train.sh             # launches FSDP2 training via `accelerate launch`
├── accelerate_config.yaml   # FSDP2 settings (2 ranks, bf16)
└── mcp_tool/
    ├── server.py            # FastMCP minesweeper game server (reveal / flag / reset)
    └── core.py              # game logic: board, no-guess generator, state machine
```

Each rollout spawns a fresh `mcp_tool/server.py` subprocess seeded by the
dataset item's seed, so every rollout for a given seed sees the same board.

---

## Hardware

- **4× H100 80GB** (or comparable; H200 / A100 80GB also work)

GPU layout used by `serve.sh` and `run_train.sh`:

```
GPU 0,1   sglang inference (TP=2, optionally + DFlash)
GPU 2,3   FSDP2 training (2 ranks, LoRA on attention)
```

---

## Prerequisites

1. **gpt-oss-120b** weights downloaded locally. Set `MODEL_PATH`.
2. **sglang virtualenv** built somewhere — the launch script defaults to
   `<repo>/sglang_venv/.venv/bin/python`. Override with `SGLANG_PYTHON`.
3. **(optional) DFlash draft model** for speculative decoding. Set
   `DRAFT_MODEL_PATH`. Without it, `serve.sh` falls back to plain decoding
   (correct, just slower).

---

## Run it

Open three terminals (or a tmux with three panes).

### 1. sglang server (GPU 0,1)

```bash
export MODEL_PATH=/path/to/gpt-oss-120b
export DRAFT_MODEL_PATH=/path/to/dflash-draft   # optional
bash examples/minesweeper/serve.sh
```

Wait until `curl -s localhost:30100/health` returns `200` (3–5 minutes from
cold cache).

### 2. Training (GPU 2,3)

```bash
export MODEL_PATH=/path/to/gpt-oss-120b
bash examples/minesweeper/run_train.sh
```

### 3. Watch logs

```bash
tail -f runs/minesweeper_demo/logs/train_*.log
tail -f runs/minesweeper_demo/logs/sglang_*.log
```

The training log prints one line per optimizer step:

```
Step 5: loss=0.0612 grad_norm=0.91 mask_rate=5.1% weight_version=5
```

---

## Configuration choices

The recipe in `train.py` is the locked-in demo config. Notable choices:

| Field                       | Value                  | Why                                                |
| --------------------------- | ---------------------- | -------------------------------------------------- |
| Board                       | 8×8, 8 mines, no-guess | Beginner difficulty; logic-only solves              |
| Dataset size                | 1000 distinct seeds    | Each seed → one deterministic board                |
| `training_mode`             | `lora`                 | LoRA is friendly for users without huge clusters    |
| `lora_target_modules`       | `q/k/v/o_proj`         | Attention only — MoE experts are mxfp4-frozen      |
| `loss_backend`              | `liger`                | Highest throughput on gpt-oss                      |
| `ddis_eps_low/high`         | `0.2 / 0.28`           | Asymmetric trust region [0.8, 1.28], slightly more room to step up |
| `loss_aggregation_mode`     | `seq-mean-token-mean`  | Per-rollout normalization, less sensitive to length |
| `use_routing_replay`        | `True`                 | Train-time MoE matches inference-time MoE exactly  |
| `weight_sync_mode`          | `merged`               | Merge LoRA into base, push full weights            |
| `batch_size / group_size`   | `128 / 16`             | 8 groups per batch; DDIS needs ≥ 4 for stability   |
| `reasoning_effort`          | `medium`               | Logical deduction benefits from a thinking budget  |
| `max_total_tokens`          | `32000`                | Larger budget than wordle to leave room for reasoning |
| `max_rounds`                | `30`                   | 8×8 boards usually clear in ≤20 actions, +slack    |

To experiment, edit `train.py` directly — there is no separate config file.

---

## What to expect

| Time         | What you should see                                                  |
| ------------ | -------------------------------------------------------------------- |
| 0–5 min      | sglang weight load, memory pool, CUDA graph warmup                   |
| 5–15 min     | First training batch fills (128 rollouts, 8 groups × 16); first step lands |
| 15–60 min    | Solve rate trends up; loss settles, mask rate stays low (< 15%)      |

The base solve rate on 8×8/8-mine no-guess boards is roughly **40–60%** for
gpt-oss-120b out of the box; training typically lifts this above 70% within
the first hour.

---

## Customization

| Want to...                             | Change                                                     |
| -------------------------------------- | ---------------------------------------------------------- |
| Try a harder board                     | `BOARD_WIDTH/HEIGHT/NUM_MINES` constants in `train.py`     |
| Use more / fewer boards                | `NUM_BOARDS` constant in `train.py`                        |
| Disable wandb                          | `use_wandb=False` in `train.py`                            |
| Change GPU layout                      | Pass GPU lists to `serve.sh` and `run_train.sh`            |
| Disable DFlash                         | Don't set `DRAFT_MODEL_PATH`                               |
| Change run output dir                  | `export RUN_DIR=/your/path`                                |
| Train longer                           | Increase `max_batches` in `train.py`                       |
| Resume from checkpoint                 | `resume_from_checkpoint="latest"` in `TrainingConfig(...)` |

---

## Troubleshooting

| Symptom                                       | Likely cause / fix                                              |
| --------------------------------------------- | --------------------------------------------------------------- |
| `MODEL_PATH does not exist`                   | Set the env var; path must contain `*.safetensors`              |
| sglang OOM at startup                         | Lower `--mem-fraction-static` in `serve.sh` (e.g. 0.78)         |
| Training OOM at first batch                   | Lower `max_capacity` in `train.py` (e.g. 120000 → 80000)        |
| Reward stuck at 0 for many steps              | Verify `serve.sh` is fully ready before training started        |
| Many "abort" log lines                        | Normal during weight sync; `max_aborts=10` rollouts before giving up |
