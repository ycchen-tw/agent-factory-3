#!/usr/bin/env bash
# Launch FSDP2 training for the Minesweeper demo on GPU 3 (single GPU).
#
# Prerequisite: 3 sglang TP=1 servers running (see serve.sh).
#
# Usage:
#   bash examples/minesweeper/run_train.sh           # GPU 3
#   bash examples/minesweeper/run_train.sh 3         # explicit

set -euo pipefail

GPUS="${1:-3}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODEL_PATH="${MODEL_PATH:-/path/to/gpt-oss-120b}"
RUN_DIR="${RUN_DIR:-$REPO_DIR/runs/minesweeper_demo}"
SGLANG_URLS="${SGLANG_URLS:-http://localhost:30100}"

LOG_DIR="$RUN_DIR/logs"
mkdir -p "$LOG_DIR"

if [[ ! -d "$MODEL_PATH" ]]; then
    echo "ERROR: MODEL_PATH does not exist: $MODEL_PATH" >&2
    exit 1
fi

export RUN_DIR
export SGLANG_URLS
export MODEL_PATH
export CUDA_VISIBLE_DEVICES="$GPUS"
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TRANSFORMERS_NO_TQDM=1
# Reduce CUDA fragmentation; helps when packed-seq activations spike
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Minesweeper demo training ==="
echo "  Training GPUs: $GPUS"
echo "  sglang URLs:   $SGLANG_URLS"
echo "  Run dir:       $RUN_DIR"

cd "$REPO_DIR"

echo "Pre-warming page cache: $MODEL_PATH"
t0=$SECONDS
cat "$MODEL_PATH"/*.safetensors > /dev/null
echo "Cache warm in $((SECONDS - t0))s"

TRAIN_LOG="$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"
exec uv run accelerate launch \
    --config_file "$SCRIPT_DIR/accelerate_config.yaml" \
    "$SCRIPT_DIR/train.py" 2>&1 | tee "$TRAIN_LOG"
