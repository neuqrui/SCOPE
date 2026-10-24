#!/usr/bin/env bash
# 180 training items, 8 rollouts, base model qwen2.5_3b.
# Two GPUs sample in parallel, 90 games each, then the shards are merged.
# SAMPLE_GPUS=4,5 overrides the default cards.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TASK_KIND=ALFWORLD
source "$PROJECT_ROOT/scope/env.sh"
unset RAY_ADDRESS
RAY_TMPDIR_BASE="${RAY_TMPDIR:-$TMPDIR/ray_scope_alfworld_3b}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export SCOPE_DISABLE_THINKING=0
OUT="${OUT:-$SCOPE_SAMPLE_ROOT/alfworld_qwen2.5_3b}"
ENV_NUM="${ENV_NUM:-15}"
NUM_SHARDS=2
mkdir -p "$OUT"
cd "$PROJECT_ROOT"

IFS=',' read -r -a GPU_ARR <<< "${SAMPLE_GPUS:-0,1}"
if [[ ${#GPU_ARR[@]} -ne "$NUM_SHARDS" ]]; then
    echo "SAMPLE_GPUS must list ${NUM_SHARDS} devices, got ${SAMPLE_GPUS:-0,1}" >&2
    exit 1
fi

python3 -m script.eval_alfworld_train_sample \
    --output-dir "$OUT" \
    --model-path "$SCOPE_BASE_3B" \
    --env-num "$ENV_NUM" \
    --prepare-only

pids=()
for shard in 0 1; do
    mkdir -p "${RAY_TMPDIR_BASE}_${shard}"
    CUDA_VISIBLE_DEVICES="${GPU_ARR[$shard]}" \
    RAY_TMPDIR="${RAY_TMPDIR_BASE}_${shard}" \
    python3 -m script.eval_alfworld_train_sample \
        --output-dir "$OUT" \
        --model-path "$SCOPE_BASE_3B" \
        --env-num "$ENV_NUM" \
        --shard-index "$shard" \
        --num-shards "$NUM_SHARDS" \
        > "$OUT/shard${shard}.log" 2>&1 &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
if [[ "$status" -ne 0 ]]; then
    echo "A sampling shard failed. Logs: $OUT/shard0.log $OUT/shard1.log" >&2
    exit 1
fi

python3 -m script.eval_alfworld_train_sample \
    --output-dir "$OUT" \
    --model-path "$SCOPE_BASE_3B" \
    --env-num "$ENV_NUM" \
    --num-shards "$NUM_SHARDS" \
    --merge-shards
