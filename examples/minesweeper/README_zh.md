# Minesweeper Demo

[English](README.md) | 中文 | [日本語](README_ja.md)

用 RL 訓練 **gpt-oss-120b**（LoRA r=8）在 4× H100 上玩 8×8 Minesweeper。
盤面用 *no-guess* 模式生成，保證可純邏輯解出，訓練訊號乾淨（[實際訓練紀錄](https://api.wandb.ai/links/asdzxcasdzxctw/h213s1fq)）。

> *建立日期：2026-05-07*

---

## 硬體

- 4× H100 80GB

GPU 分配：

```
GPU 0   sglang 推論（TP=1，port 30100，可選 DFlash）
GPU 1   sglang 推論（TP=1，port 30101，可選 DFlash）
GPU 2   sglang 推論（TP=1，port 30102，可選 DFlash）
GPU 3   FSDP2 訓練（1 rank + CPU offload，LoRA on attention）
```

---

## 跑起來

開 4 個終端：

```bash
# Terminal 1-3：起 3 個 sglang server（每張 GPU 一個 TP=1）
export MODEL_PATH=/path/to/gpt-oss-120b
export DRAFT_MODEL_PATH=/path/to/dflash-draft   # 沒有就不設

bash examples/minesweeper/serve.sh 0 30100 1   # GPU 0
bash examples/minesweeper/serve.sh 1 30101 1   # GPU 1
bash examples/minesweeper/serve.sh 2 30102 1   # GPU 2

# 等 3 個 /health 都回 200（冷啟約 3-5 分鐘）

# Terminal 4：起訓練
export MODEL_PATH=/path/to/gpt-oss-120b
export SGLANG_URLS=http://localhost:30100,http://localhost:30101,http://localhost:30102
bash examples/minesweeper/run_train.sh
```

Log 在 `runs/minesweeper_demo/logs/`。

---

## 主要設定

直接改 `train.py`，沒有獨立 config 檔。

| 設定 | 值 | 備註 |
|---|---|---|
| Board | 8×8 / 8 mines / no-guess | Beginner 難度，純邏輯題 |
| Dataset | 1000 個 seeds | 每個 seed 對應確定性盤面 |
| LoRA | r=8 / alpha=32 / q,k,v,o | MoE experts 被 mxfp4 凍結 |
| `loss_backend` | `liger` | gpt-oss 上最快 |
| `ddis_eps_low/high` | 0.2 / 0.28 | 不對稱信賴域 [0.8, 1.28] |
| `use_routing_replay` | True | 訓練時 MoE routing 對齊推論 |
| `weight_sync_mode` | `merged` | LoRA merge 後推給 sglang |
| `batch_size / group_size` | 128 / 16 | 每 batch 8 個 group（DDIS 至少需要 4 個）|
| `reasoning_effort` | `medium` | 邏輯推理需要 thinking budget |
| `max_total_tokens` | 32000 | 比 wordle 大，留空間給 reasoning |
| `max_rounds` | 30 | 8×8 盤通常 ≤20 動作就解完 |
| 學習率 | 1e-5 | LoRA 慣用值 |

---

## 自訂

| 想做的事 | 改哪 |
|---|---|
| 換難度（板大小、地雷數） | `train.py` 裡 `BOARD_WIDTH / BOARD_HEIGHT / NUM_MINES` |
| 換盤面數量 | `NUM_BOARDS` |
| 關掉 wandb | `train.py` 裡 `use_wandb=False` |
| 改 GPU 分配 | `serve.sh` / `run_train.sh` 的參數 |
| 關掉 DFlash | 不設 `DRAFT_MODEL_PATH` |
| 改輸出位置 | `export RUN_DIR=/your/path` |
| 訓練更久 | `max_batches` 改大 |
| 從 checkpoint 接續 | `TrainingConfig(resume_from_checkpoint="latest", ...)` |

---

## 檔案結構

```
minesweeper/
├── train.py                 # 訓練入口（TrainingConfig + RLFlow + reward + dataset）
├── serve.sh                 # 起 sglang TP=1（DFlash 可選）
├── run_train.sh             # accelerate launch FSDP2 訓練（CPU offload）
├── accelerate_config.yaml   # FSDP2 設定（1 rank + offload）
└── mcp_tool/
    ├── server.py            # FastMCP minesweeper game（reveal / flag / reset）
    └── core.py              # game logic：board、no-guess generator、狀態機
```
