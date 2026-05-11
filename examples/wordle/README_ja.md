# Wordle Demo

[English](README.md) | [中文](README_zh.md) | 日本語

**gpt-oss-120b** を RL で訓練し、4× H100 上で英語 Wordle をプレイさせます。
約 4 時間の訓練で solve rate の明確な向上が確認できます（[wandb の訓練記録](https://wandb.ai/asdzxcasdzxctw/agent-factory-wordle/reports/Untitled-Report--VmlldzoxNjg0MjE2OQ)）。

> *作成日：2026-05-07*

---

## ハードウェア

- 4× H100 80GB

GPU 配置:

```
GPU 0,1   sglang 推論（TP=2、DFlash オプション）
GPU 2,3   FSDP2 訓練（2 ranks、attention に LoRA）
```

---

## 実行

ターミナルを 2 つ開く:

```bash
# Terminal 1: sglang server を起動
export MODEL_PATH=/path/to/gpt-oss-120b
export DRAFT_MODEL_PATH=/path/to/dflash-draft   # 未設定なら DFlash 無効
bash examples/wordle/serve.sh

# curl localhost:30100/health が 200 を返すまで待つ（コールドスタート 3-5 分）

# Terminal 2: 訓練を起動
export MODEL_PATH=/path/to/gpt-oss-120b
bash examples/wordle/run_train.sh
```

ログは `runs/wordle_demo/logs/` に出力されます。

---

## 主な設定

`train.py` を直接編集します（独立の config ファイルなし）。

| 設定 | 値 | 備考 |
|---|---|---|
| LoRA | r=8 / alpha=32 / q,k,v,o | MoE expert は mxfp4 で凍結。attention のみ訓練可能 |
| `loss_backend` | `liger` | gpt-oss で最速 |
| `ddis_eps_low/high` | 0.2 / 0.28 | 非対称信頼領域 [0.8, 1.28]、上方向に少し緩め |
| `use_routing_replay` | True | 訓練時の MoE routing を推論時と一致させる |
| `weight_sync_mode` | `merged` | LoRA を base にマージしてフル重みを sglang に push |
| `batch_size / group_size` | 128 / 16 | batch あたり 8 group（DDIS は最低 4 group 必要） |
| 学習率 | 1e-5 | LoRA の標準値 |

---

## カスタマイズ

| やりたいこと | 変更箇所 |
|---|---|
| wandb を無効化 | `train.py` 内の `use_wandb=False` |
| GPU 配置の変更 | `serve.sh` と `run_train.sh` の第一引数 |
| DFlash 無効化 | `DRAFT_MODEL_PATH` を未設定にする |
| 出力先の変更 | `export RUN_DIR=/your/path` |
| もっと長く訓練 | `train.py` 内の `max_batches` を増やす |
| checkpoint から再開 | `TrainingConfig(resume_from_checkpoint="latest", ...)` |

---

## ファイル構成

```
wordle/
├── train.py                 # 訓練エントリ（TrainingConfig + RLFlow + reward + dataset）
├── serve.sh                 # sglang TP=2 起動（DFlash オプション）
├── run_train.sh             # accelerate launch FSDP2 訓練
├── accelerate_config.yaml   # FSDP2 設定
└── mcp_tool/
    ├── server.py            # FastMCP wordle game
    └── data/
        ├── answers.txt          # 2,314 個の正解単語
        └── allowed_guesses.txt  # 10,656 個の入力可能単語
```
