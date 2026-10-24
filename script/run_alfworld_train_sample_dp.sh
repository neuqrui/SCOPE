#!/usr/bin/env bash
# 180 ALFWorld train games, 30 from each task type, split across 4 GPUs.
# Each card rolls out 45 games x 8 samples with Qwen2.5-3B.
# Stop the single-card run before launching; GPU 6 is that job.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TASK_KIND=ALFWORLD
source "$PROJECT_ROOT/examples/_sdar_env.sh"

export SAMPLE_GPUS="${SAMPLE_GPUS:-0,1,6,7}"
unset RAY_ADDRESS
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.6}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

IFS=',' read -r -a GPU_LIST <<< "$SAMPLE_GPUS"
NUM_SHARDS=${#GPU_LIST[@]}
OUT="${OUT:-work/ckpt/rollouts_val/alfworld_train180_t1.0}"
LOG_DIR="${LOG_DIR:-work/logs/alfworld_train180_dp}"
ENV_NUM="${ENV_NUM:-15}"
mkdir -p "$LOG_DIR" "$OUT/shards"
cd "$PROJECT_ROOT"

pids=()
for shard in "${!GPU_LIST[@]}"; do
    gpu="${GPU_LIST[$shard]}"
    mkdir -p "work/tmp/ray_alfworld_dp_${shard}"
    CUDA_VISIBLE_DEVICES="$gpu" \
    RAY_TMPDIR="work/tmp/ray_alfworld_dp_${shard}" \
    python3 -m script.eval_alfworld_train_sample \
        --output-dir "$OUT" \
        --model-path work/models/Qwen2.5-3B-Instruct \
        --env-num "$ENV_NUM" \
        --shard-index "$shard" \
        --num-shards "$NUM_SHARDS" \
        > "$LOG_DIR/shard${shard}.log" 2>&1 &
    pids+=("$!")
    echo "shard $shard -> GPU $gpu pid ${pids[-1]}"
done

failed=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        failed=1
    fi
done
if [[ "$failed" -ne 0 ]]; then
    echo "a shard failed; logs are in $LOG_DIR" >&2
    exit 1
fi

python3 -m script.eval_alfworld_train_sample \
    --output-dir "$OUT" \
    --model-path work/models/Qwen2.5-3B-Instruct \
    --num-shards "$NUM_SHARDS" \
    --merge-shards
echo "done -> $OUT/qwen2.5_3b.json"
