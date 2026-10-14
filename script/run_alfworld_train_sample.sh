#!/usr/bin/env bash
# 180 ALFWorld train games, 30 from each task type, 8 rollouts on Qwen2.5-3B.
# Set SAMPLE_GPUS to a free card before launching.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TASK_KIND=ALFWORLD
source "$PROJECT_ROOT/examples/_sdar_env.sh"

if [[ -n "${SAMPLE_GPUS:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="$SAMPLE_GPUS"
fi
unset RAY_ADDRESS
export RAY_TMPDIR="${RAY_TMPDIR:-/data/qirui/SDAR/tmp/ray_alfworld_sample180}"
mkdir -p "$RAY_TMPDIR"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.6}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

OUT="${OUT:-/data/qirui/SDAR/ckpt/rollouts_val/alfworld_train180_t1.0}"
mkdir -p "$OUT"
cd "$PROJECT_ROOT"
python3 -m script.eval_alfworld_train_sample \
    --output-dir "$OUT" \
    --model-path /data/qirui/SDAR/models/Qwen2.5-3B-Instruct \
    --env-num "${ENV_NUM:-15}"
