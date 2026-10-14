#!/usr/bin/env bash
# Rejection-sample the WebShop tasks actually drawn in 150 GRPO steps.
# Model: Qwen2.5-3B GRPO checkpoint at global_step_150 (HF export).
# Schedule matches script/webshop_reject_sample.py defaults:
# 4 rollouts per task per round, a second round only if the first has no success.
#
# CUDA_VISIBLE_DEVICES=0,1 bash script/run_webshop_grpo_step150_reject.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TASK_KIND=WEBSHOP
source "$PROJECT_ROOT/examples/_sdar_env.sh"

export CUDA_VISIBLE_DEVICES="${REJECT_CUDA_VISIBLE_DEVICES:-0,1}"
unset RAY_ADDRESS
export PYTHONUNBUFFERED=1
export JAVA_HOME="${CONDA_PREFIX:-$ENV_PREFIX}/lib/jvm"
export JVM_PATH="$JAVA_HOME/lib/server/libjvm.so"
export PATH="$JAVA_HOME/bin:$PATH"
export _JAVA_OPTIONS="-XX:+UseSerialGC -Xss512k -Xmx1g"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

MODEL_PATH="${REJECT_MODEL_PATH:-/data/qirui/SDAR/ckpt/hf/grpo_qwen2.5_3b_step150}"
OUT_DIR="${OUT_DIR:-/data/qirui/SDAR/reject_sample/webshop_grpo_3b_step150}"
mkdir -p "$OUT_DIR"

IFS=',' read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
NUM_SHARDS=${#GPUS[@]}
CONCURRENCY=32
pids=()
for i in "${!GPUS[@]}"; do
    CUDA_VISIBLE_DEVICES="${GPUS[$i]}" \
    RAY_TMPDIR="/data/qirui/SDAR/tmp/rs_grpo150_${i}" \
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
        --base-seed 0 \
        --train-start 500 \
        --draw-steps 150 \
        --env-num 16 \
        --tensor-parallel-size 1 \
        --concurrency "$CONCURRENCY" \
        --num-shards "$NUM_SHARDS" \
        --shard-id "$i" \
        > "$OUT_DIR/shard${i}.log" 2>&1 &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
exit "$status"
