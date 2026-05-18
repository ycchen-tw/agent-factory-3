#!/usr/bin/env bash
# Launch sglang TP=2 server for the multilingual Wordle demo (Japanese mode).
#
# Identical hyperparameters to examples/wordle/serve.sh — only RUN_DIR differs
# so logs/checkpoints land in runs/wordle_ja_demo/ instead of runs/wordle_demo/.
#
# DFlash speculative decoding is enabled when DRAFT_MODEL_PATH is set;
# unset it to fall back to plain decoding (slower but works without a draft model).
#
# GPU layout (4× H100):
#   GPU 0,1   sglang TP=2 (this script)
#   GPU 2,3   FSDP2 training (run_train.sh)
#
# Usage:
#   bash examples/wordle_multilingual/serve.sh                # GPU 0,1, port 30100
#   bash examples/wordle_multilingual/serve.sh 0,1 30100      # explicit GPUs and port

set -euo pipefail

GPUS="${1:-0,1}"
PORT="${2:-30100}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODEL_PATH="${MODEL_PATH:-/path/to/gpt-oss-120b}"
DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-}"   # leave empty to disable DFlash
RUN_DIR="${RUN_DIR:-$REPO_DIR/runs/wordle_ja_demo}"
SGLANG_PYTHON="${SGLANG_PYTHON:-$REPO_DIR/sglang_venv/.venv/bin/python}"

LOG_DIR="$RUN_DIR/logs"
mkdir -p "$LOG_DIR"

if [[ ! -d "$MODEL_PATH" ]]; then
    echo "ERROR: MODEL_PATH does not exist: $MODEL_PATH" >&2
    echo "       Set MODEL_PATH=/your/gpt-oss-120b before running this script." >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPUS"
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export TORCHINDUCTOR_QUIESCE_ASYNC_COMPILE_POOL=1
export TORCHINDUCTOR_COMPILE_THREADS=16
# expandable_segments collides with sglang custom_all_reduce (CUDA graph capture fails).
# A tiny KV-cache leak is harmless during long runs; strict check would kill the server.
export SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=false
# Required pair for DFlash with logprobs/routing capture.
export SGLANG_ENABLE_SPEC_V2=True
export SGLANG_ENABLE_DFLASH_SPEC_V2=True
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=True

echo "[GPU $GPUS] Pre-warming page cache..."
t0=$SECONDS
cat "$MODEL_PATH"/*.safetensors > /dev/null
[[ -n "$DRAFT_MODEL_PATH" && -d "$DRAFT_MODEL_PATH" ]] && cat "$DRAFT_MODEL_PATH"/*.safetensors > /dev/null || true
echo "[GPU $GPUS] Cache warm in $((SECONDS - t0))s"

ARGS=(
    --model-path "$MODEL_PATH"
    --tp-size 2
    --dtype bfloat16
    --attention-backend fa3
    --kv-cache-dtype fp8_e4m3
    --mem-fraction-static 0.82
    --max-running-requests 128
    --swa-full-tokens-ratio 0.1
    --trust-remote-code
    --host 0.0.0.0
    --port "$PORT"
    --scheduler-recv-interval 16
    --log-level info
    --enable-return-routed-experts
    --weight-loader-disable-mmap
    --disable-cuda-graph-padding
)

if [[ -n "$DRAFT_MODEL_PATH" ]]; then
    if [[ ! -d "$DRAFT_MODEL_PATH" ]]; then
        echo "ERROR: DRAFT_MODEL_PATH set but does not exist: $DRAFT_MODEL_PATH" >&2
        exit 1
    fi
    echo "DFlash: ON (draft=$DRAFT_MODEL_PATH)"
    ARGS+=(
        --speculative-algorithm DFLASH
        --speculative-draft-model-path "$DRAFT_MODEL_PATH"
        --speculative-dflash-block-size 10
    )
else
    echo "DFlash: OFF (set DRAFT_MODEL_PATH to enable)"
fi

LOG_FILE="$LOG_DIR/sglang_$(hostname)_$(date +%Y%m%d_%H%M%S).log"
echo "Log: $LOG_FILE"
exec "$SGLANG_PYTHON" -m sglang.launch_server "${ARGS[@]}" 2>&1 | tee "$LOG_FILE"
