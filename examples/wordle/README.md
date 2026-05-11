# Wordle Demo

English | [中文](README_zh.md) | [日本語](README_ja.md)

Train **gpt-oss-120b** to play English Wordle on 4× H100.
About four hours of training shows a clear solve-rate lift ([sample wandb run](https://wandb.ai/asdzxcasdzxctw/agent-factory-wordle/reports/Untitled-Report--VmlldzoxNjg0MjE2OQ)).

> *Created 2026-05-07.*

---

## Hardware

- 4× H100 80GB

GPU layout:

```
GPU 0,1   sglang inference (TP=2, optional DFlash)
GPU 2,3   FSDP2 training (2 ranks, LoRA on attention)
```

---

## Run it

Open two terminals:

```bash
# Terminal 1: launch sglang server
export MODEL_PATH=/path/to/gpt-oss-120b
export DRAFT_MODEL_PATH=/path/to/dflash-draft   # optional; unset to disable DFlash
bash examples/wordle/serve.sh

# Wait until curl localhost:30100/health returns 200 (3–5 min cold start).

# Terminal 2: launch training
export MODEL_PATH=/path/to/gpt-oss-120b
bash examples/wordle/run_train.sh
```

Logs land in `runs/wordle_demo/logs/`.

---

## Key settings

Edit `train.py` directly; there is no separate config file.

| Setting | Value | Note |
|---|---|---|
| LoRA | r=8 / alpha=32 / q,k,v,o | MoE experts are mxfp4-frozen; only attention is trainable |
| `loss_backend` | `liger` | Fastest on gpt-oss |
| `ddis_eps_low/high` | 0.2 / 0.28 | Asymmetric trust region [0.8, 1.28], slightly more room to step up |
| `use_routing_replay` | True | Train-time MoE matches inference-time MoE |
| `weight_sync_mode` | `merged` | Merge LoRA into base, push full weights to sglang |
| `batch_size / group_size` | 128 / 16 | 8 groups per batch (DDIS needs ≥ 4) |
| Learning rate | 1e-5 | Typical LoRA value |

---

## Customization

| Want to... | Change |
|---|---|
| Disable wandb | `use_wandb=False` in `train.py` |
| Change GPU layout | First arg of `serve.sh` and `run_train.sh` |
| Disable DFlash | Don't set `DRAFT_MODEL_PATH` |
| Change output dir | `export RUN_DIR=/your/path` |
| Train longer | Increase `max_batches` in `train.py` |
| Resume from checkpoint | `TrainingConfig(resume_from_checkpoint="latest", ...)` |

---

## File structure

```
wordle/
├── train.py                 # training entry (TrainingConfig + RLFlow + reward + dataset)
├── serve.sh                 # launches sglang TP=2 (DFlash optional)
├── run_train.sh             # accelerate launch FSDP2 training
├── accelerate_config.yaml   # FSDP2 settings
└── mcp_tool/
    ├── server.py            # FastMCP wordle game
    └── data/
        ├── answers.txt          # 2,314 target words
        └── allowed_guesses.txt  # 10,656 valid guess words
```
