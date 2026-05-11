# Minesweeper Demo

English | [中文](README_zh.md) | [日本語](README_ja.md)

Train **gpt-oss-120b** (LoRA r=8) to play 8×8 Minesweeper on 4× H100.
Boards are generated in *no-guess* mode so every game is solvable by
pure deduction — clean training signal ([sample wandb run](https://api.wandb.ai/links/asdzxcasdzxctw/h213s1fq)).

> *Created 2026-05-07.*

---

## Hardware

- 4× H100 80GB

GPU layout:

```
GPU 0   sglang inference (TP=1, port 30100, optional DFlash)
GPU 1   sglang inference (TP=1, port 30101, optional DFlash)
GPU 2   sglang inference (TP=1, port 30102, optional DFlash)
GPU 3   FSDP2 training (1 rank + CPU offload, LoRA on attention)
```

---

## Run it

Open four terminals:

```bash
# Terminals 1-3: launch 3 sglang servers (TP=1 per GPU)
export MODEL_PATH=/path/to/gpt-oss-120b
export DRAFT_MODEL_PATH=/path/to/dflash-draft   # optional; unset to disable DFlash

bash examples/minesweeper/serve.sh 0 30100 1   # GPU 0
bash examples/minesweeper/serve.sh 1 30101 1   # GPU 1
bash examples/minesweeper/serve.sh 2 30102 1   # GPU 2

# Wait until all 3 /health endpoints return 200 (3–5 min cold start).

# Terminal 4: launch training
export MODEL_PATH=/path/to/gpt-oss-120b
export SGLANG_URLS=http://localhost:30100,http://localhost:30101,http://localhost:30102
bash examples/minesweeper/run_train.sh
```

Logs land in `runs/minesweeper_demo/logs/`.

---

## Key settings

Edit `train.py` directly; there is no separate config file.

| Setting | Value | Note |
|---|---|---|
| Board | 8×8 / 8 mines / no-guess | Beginner difficulty, pure logic |
| Dataset | 1000 seeds | One deterministic board per seed |
| LoRA | r=8 / alpha=32 / q,k,v,o | MoE experts are mxfp4-frozen |
| `loss_backend` | `liger` | Fastest on gpt-oss |
| `ddis_eps_low/high` | 0.2 / 0.28 | Asymmetric trust region [0.8, 1.28] |
| `use_routing_replay` | True | Train-time MoE matches inference-time MoE |
| `weight_sync_mode` | `merged` | Merge LoRA into base, push full weights |
| `batch_size / group_size` | 128 / 16 | 8 groups per batch (DDIS needs ≥ 4) |
| `reasoning_effort` | `medium` | Logical deduction needs a thinking budget |
| `max_total_tokens` | 32000 | More than wordle to leave room for reasoning |
| `max_rounds` | 30 | 8×8 boards usually clear in ≤20 moves |
| Learning rate | 1e-5 | Typical LoRA value |

---

## Customization

| Want to... | Change |
|---|---|
| Change difficulty (board size, mines) | `BOARD_WIDTH / BOARD_HEIGHT / NUM_MINES` in `train.py` |
| Change number of boards | `NUM_BOARDS` |
| Disable wandb | `use_wandb=False` in `train.py` |
| Change GPU layout | `serve.sh` / `run_train.sh` args |
| Disable DFlash | Don't set `DRAFT_MODEL_PATH` |
| Change output dir | `export RUN_DIR=/your/path` |
| Train longer | Increase `max_batches` |
| Resume from checkpoint | `TrainingConfig(resume_from_checkpoint="latest", ...)` |

---

## File structure

```
minesweeper/
├── train.py                 # training entry (TrainingConfig + RLFlow + reward + dataset)
├── serve.sh                 # launches sglang TP=1 (DFlash optional)
├── run_train.sh             # accelerate launch FSDP2 training (CPU offload)
├── accelerate_config.yaml   # FSDP2 settings (1 rank + offload)
└── mcp_tool/
    ├── server.py            # FastMCP minesweeper game (reveal / flag / reset)
    └── core.py              # game logic: board, no-guess generator, state machine
```
