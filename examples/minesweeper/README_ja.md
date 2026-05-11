# Minesweeper Demo

[English](README.md) | [中文](README_zh.md) | 日本語

**gpt-oss-120b**（LoRA r=8）を RL で訓練し、4× H100 上で 8×8 Minesweeper をプレイさせます。
盤面は *no-guess* モードで生成され、純粋な論理で解けることが保証されているため、訓練信号がクリーンです（[wandb の訓練記録](https://api.wandb.ai/links/asdzxcasdzxctw/h213s1fq)）。

> *作成日：2026-05-07*

---

## ハードウェア

- 4× H100 80GB

GPU 配置:

```
GPU 0   sglang 推論（TP=1、port 30100、DFlash オプション）
GPU 1   sglang 推論（TP=1、port 30101、DFlash オプション）
GPU 2   sglang 推論（TP=1、port 30102、DFlash オプション）
GPU 3   FSDP2 訓練（1 rank + CPU offload、attention に LoRA）
```

---

## 実行

ターミナルを 4 つ開く:

```bash
# Terminal 1-3: sglang server を 3 つ起動（各 GPU に TP=1）
export MODEL_PATH=/path/to/gpt-oss-120b
export DRAFT_MODEL_PATH=/path/to/dflash-draft   # 未設定なら DFlash 無効

bash examples/minesweeper/serve.sh 0 30100 1   # GPU 0
bash examples/minesweeper/serve.sh 1 30101 1   # GPU 1
bash examples/minesweeper/serve.sh 2 30102 1   # GPU 2

# 3 つの /health が全て 200 を返すまで待つ（コールドスタート 3-5 分）

# Terminal 4: 訓練を起動
export MODEL_PATH=/path/to/gpt-oss-120b
export SGLANG_URLS=http://localhost:30100,http://localhost:30101,http://localhost:30102
bash examples/minesweeper/run_train.sh
```

ログは `runs/minesweeper_demo/logs/` に出力されます。

---

## 主な設定

`train.py` を直接編集します（独立の config ファイルなし）。

| 設定 | 値 | 備考 |
|---|---|---|
| Board | 8×8 / 8 mines / no-guess | Beginner 難度、純粋な論理問題 |
| Dataset | 1000 個の seeds | seed 1 つにつき決定論的な盤面 1 つ |
| LoRA | r=8 / alpha=32 / q,k,v,o | MoE expert は mxfp4 で凍結 |
| `loss_backend` | `liger` | gpt-oss で最速 |
| `ddis_eps_low/high` | 0.2 / 0.28 | 非対称信頼領域 [0.8, 1.28] |
| `use_routing_replay` | True | 訓練時の MoE routing を推論時と一致させる |
| `weight_sync_mode` | `merged` | LoRA を base にマージしてフル重みを push |
| `batch_size / group_size` | 128 / 16 | batch あたり 8 group（DDIS は最低 4 group 必要）|
| `reasoning_effort` | `medium` | 論理推論には thinking budget が必要 |
| `max_total_tokens` | 32000 | wordle より大きく、reasoning の余地を確保 |
| `max_rounds` | 30 | 8×8 盤は通常 ≤20 手で完了 |
| 学習率 | 1e-5 | LoRA の標準値 |

---

## カスタマイズ

| やりたいこと | 変更箇所 |
|---|---|
| 難度を変える（盤サイズ、地雷数）| `train.py` 内の `BOARD_WIDTH / BOARD_HEIGHT / NUM_MINES` |
| 盤面数を変える | `NUM_BOARDS` |
| wandb を無効化 | `train.py` 内の `use_wandb=False` |
| GPU 配置の変更 | `serve.sh` / `run_train.sh` の引数 |
| DFlash 無効化 | `DRAFT_MODEL_PATH` を未設定にする |
| 出力先の変更 | `export RUN_DIR=/your/path` |
| もっと長く訓練 | `max_batches` を増やす |
| checkpoint から再開 | `TrainingConfig(resume_from_checkpoint="latest", ...)` |

---

## ファイル構成

```
minesweeper/
├── train.py                 # 訓練エントリ（TrainingConfig + RLFlow + reward + dataset）
├── serve.sh                 # sglang TP=1 起動（DFlash オプション）
├── run_train.sh             # accelerate launch FSDP2 訓練（CPU offload）
├── accelerate_config.yaml   # FSDP2 設定（1 rank + offload）
└── mcp_tool/
    ├── server.py            # FastMCP minesweeper game（reveal / flag / reset）
    └── core.py              # game logic：board、no-guess generator、ステートマシン
```
