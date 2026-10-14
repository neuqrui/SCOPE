#!/usr/bin/env bash
# Rejection-sample WebShop train goals with MiMo-V2.6-Distill-Qwen-9B.
# Data matches training: use_small catalog, human_goals=false, seed 0,
# train indices [500, n_goals), history_length=2, max_steps=15.
#
# GPUs 0-1 are often busy. Default to the next free pair; override with
# CUDA_VISIBLE_DEVICES=4,5 bash script/run_webshop_reject_sample.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TASK_KIND=WEBSHOP
source "$PROJECT_ROOT/examples/_sdar_env.sh"

# Do not inherit the training shell's CUDA_VISIBLE_DEVICES. That mask is often
# two GPUs, and vLLM then joins the training Ray cluster to look for more.
export CUDA_VISIBLE_DEVICES="${REJECT_CUDA_VISIBLE_DEVICES:-4,5,6,7}"
unset RAY_ADDRESS
export PYTHONUNBUFFERED=1
# _sdar_env.sh sets MODEL_PATH to the 3B training model.
MODEL_PATH="${REJECT_MODEL_PATH:-/data/qirui/models/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B}"
OUT_DIR="${OUT_DIR:-/data/qirui/SDAR/reject_sample/webshop_mimo_9b}"

# One replica per GPU. Tensor parallel 4 spends the cards on communication;
# four independent engines keep each H200's batch on that card.
IFS=',' read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
NUM_SHARDS=${#GPUS[@]}
CONCURRENCY=40
pids=()
for i in "${!GPUS[@]}"; do
    CUDA_VISIBLE_DEVICES="${GPUS[$i]}" \
    RAY_TMPDIR="/data/qirui/SDAR/tmp/rs${i}" \
    python3 -m script.webshop_reject_sample \
        --model-path "$MODEL_PATH" \
        --out-dir "$OUT_DIR" \
        --temperature 0.6 \
        --top-p 1.0 \
        --top-k -1 \
        --max-tokens 512 \
        --max-prompt-length 4096 \
        --max-model-len 4608 \
        --samples-per-round 4 \
        --max-rounds 2 \
        --max-steps 15 \
        --history-length 2 \
        --seed 0 \
        --train-start 500 \
        "$@" \
        --tensor-parallel-size 1 \
        --concurrency "$CONCURRENCY" \
        --num-shards "$NUM_SHARDS" \
        --shard-id "$i" &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
exit "$status"
