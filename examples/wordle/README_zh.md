# Wordle Demo（繁中）

用 RL 訓練 **gpt-oss-120b**（LoRA r=8）在 4× H100 上玩英文 Wordle。
大約 1 小時可以看到 solve rate 明顯上升。

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

開兩個終端：

```bash
# Terminal 1：起 sglang server
export MODEL_PATH=/path/to/gpt-oss-120b
export DRAFT_MODEL_PATH=/path/to/dflash-draft   # 沒有就不設，自動關閉 DFlash
bash examples/wordle/serve.sh

# 等 curl localhost:30100/health 回 200（冷啟約 3-5 分鐘）

# Terminal 2：起訓練
export MODEL_PATH=/path/to/gpt-oss-120b
bash examples/wordle/run_train.sh
```

Log 在 `runs/wordle_demo/logs/`。

---

## 主要設定

直接改 `train.py`，沒有獨立 config 檔。比較關鍵的：

| 設定 | 值 | 備註 |
|---|---|---|
| LoRA | r=8 / alpha=32 / q,k,v,o | MoE experts 被 mxfp4 凍結，只能訓 attention |
| `loss_backend` | `liger` | gpt-oss 上最快 |
| `ddis_eps_low/high` | 0.2 / 0.28 | 不對稱信賴域 [0.8, 1.28]，往上踩稍微寬 |
| `use_routing_replay` | True | 訓練時 MoE routing 對齊推論 |
| `weight_sync_mode` | `merged` | LoRA merge 後當完整權重推給 sglang |
| `batch_size / group_size` | 128 / 16 | 每 batch 8 個 group（DDIS 至少需要 4 個） |
| 學習率 | 1e-5 | LoRA 慣用值 |

---

## 預期結果

| 時間 | 狀況 |
|---|---|
| 0–5 分鐘 | sglang 載入權重、CUDA graph warmup |
| 5–15 分鐘 | 第一個 batch 收滿（128 rollout，8 個 group × 16），第一個 step |
| 15–60 分鐘 | solve rate 從 ~30-50% 慢慢爬到 55-70% |

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
| 關掉 wandb | `train.py` 裡 `use_wandb=False` |
| 改 GPU 分配 | `serve.sh` 跟 `run_train.sh` 第一個參數 |
| 關掉 DFlash | 不設 `DRAFT_MODEL_PATH` |
| 改輸出位置 | `export RUN_DIR=/your/path` |
| 訓練更久 | `train.py` 裡 `max_batches` 改大 |
| 從 checkpoint 接續 | `TrainingConfig(resume_from_checkpoint="latest", ...)` |

---

## 檔案結構

```
wordle/
├── train.py                 # 訓練入口（TrainingConfig + RLFlow + reward + dataset）
├── serve.sh                 # 起 sglang TP=2（DFlash 可選）
├── run_train.sh             # accelerate launch FSDP2 訓練
├── accelerate_config.yaml   # FSDP2 設定
└── mcp_tool/
    ├── server.py            # FastMCP wordle game
    └── data/
        ├── answers.txt          # 2,314 個目標單字
        └── allowed_guesses.txt  # 10,656 個可猜單字
```
