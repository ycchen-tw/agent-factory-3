# Minesweeper Demo（繁中）

用 RL 訓練 **gpt-oss-120b**（LoRA r=8）在 4× H100 上玩 8×8 Minesweeper。
盤面用 *no-guess* 模式生成，保證可純邏輯解出，訓練訊號乾淨。

> *建立日期：2026-05-07*

---

## 硬體

- 4× H100 80GB（H200 / A100 80GB 也可）

GPU 分配：

```
GPU 0,1   sglang 推論（TP=2，可選 DFlash）
GPU 2,3   FSDP2 訓練（2 ranks，LoRA on attention）
```

---

## 跑起來

```bash
# Terminal 1
export MODEL_PATH=/path/to/gpt-oss-120b
export DRAFT_MODEL_PATH=/path/to/dflash-draft   # 沒有就不設
bash examples/minesweeper/serve.sh

# Terminal 2 (等 /health 200 後)
export MODEL_PATH=/path/to/gpt-oss-120b
bash examples/minesweeper/run_train.sh
```

Log 在 `runs/minesweeper_demo/logs/`。

---

## 主要設定

直接改 `train.py`：

| 設定 | 值 | 備註 |
|---|---|---|
| Board | 8×8 / 8 mines / no-guess | Beginner 難度，純邏輯題 |
| Dataset | 1000 個 seeds | 每個 seed 對應確定性盤面 |
| LoRA | r=8 / alpha=32 / q,k,v,o | MoE experts 被 mxfp4 凍結 |
| `loss_backend` | `liger` | gpt-oss 上最快 |
| `ddis_eps_low/high` | 0.2 / 0.28 | 不對稱信賴域 [0.8, 1.28] |
| `use_routing_replay` | True | 訓練時 MoE routing 對齊推論 |
| `weight_sync_mode` | `merged` | LoRA merge 後推給 sglang |
| `batch_size / group_size` | 128 / 16 | DDIS 至少需要 4 個 group |
| `reasoning_effort` | `medium` | 邏輯推理需要 thinking budget |
| `max_total_tokens` | 32000 | 比 wordle 大，留空間給 reasoning |
| `max_rounds` | 30 | 8×8 盤通常 ≤20 動作就解完 |

---

## 預期結果

| 時間 | 狀況 |
|---|---|
| 0–5 分鐘 | sglang 載入權重、CUDA graph warmup |
| 5–15 分鐘 | 第一個 batch 收滿（128 rollout，8 個 group × 16），第一個 step |
| 15–60 分鐘 | solve rate 從 ~40-60% 慢慢爬到 70%+ |

---

## 常見狀況

| 狀況 | 對策 |
|---|---|
| sglang OOM | 降 `--mem-fraction-static`（0.82 → 0.78） |
| 訓練 OOM | 降 `max_capacity`（120000 → 80000） |
| 前幾步 reward 都 0 | 正常。`filter_all_failed=True` 把 0% 的 group 過濾掉，等模型開始有信號 |
| 看到很多 `abort` | 正常，weight sync 時會 abort 進行中的 rollout，最多重試 10 次 |

---

## 自訂

| 想做的事 | 改哪 |
|---|---|
| 換難度（板大小、地雷數） | `train.py` 裡 `BOARD_WIDTH / BOARD_HEIGHT / NUM_MINES` |
| 換盤面數量 | `NUM_BOARDS` |
| 關掉 wandb | `use_wandb=False` |
| 改 GPU 分配 | `serve.sh` 跟 `run_train.sh` 第一個參數 |
| 關掉 DFlash | 不設 `DRAFT_MODEL_PATH` |
| 改輸出位置 | `export RUN_DIR=/your/path` |
| 訓練更久 | `max_batches` 改大 |
| 從 checkpoint 接續 | `TrainingConfig(resume_from_checkpoint="latest", ...)` |

---

## 檔案結構

```
minesweeper/
├── train.py                 # 訓練入口
├── serve.sh                 # 起 sglang TP=2（DFlash 可選）
├── run_train.sh             # accelerate launch FSDP2 訓練
├── accelerate_config.yaml   # FSDP2 設定
└── mcp_tool/
    ├── server.py            # FastMCP minesweeper game（reveal / flag / reset）
    └── core.py              # game logic：board、no-guess generator、狀態機
```
