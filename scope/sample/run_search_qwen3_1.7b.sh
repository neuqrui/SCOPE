#!/usr/bin/env bash
# 180 training items, 8 rollouts, base model qwen3_1.7b.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TASK_KIND=SEARCH
source "$PROJECT_ROOT/scope/env.sh"
if [[ -n "${SAMPLE_GPUS:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="$SAMPLE_GPUS"
fi
unset RAY_ADDRESS
export RAY_TMPDIR="${RAY_TMPDIR:-$TMPDIR/ray_scope_search_qwen3_1.7b}"
mkdir -p "$RAY_TMPDIR"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export SCOPE_DISABLE_THINKING=1
OUT="${OUT:-$SCOPE_SAMPLE_ROOT/search_qwen3_1.7b}"
mkdir -p "$OUT"
cd "$PROJECT_ROOT"
python3 "$SCRIPT_DIR/eval_search_train_sample.py" \
    --output-dir "$OUT" \
    --model-path "$SCOPE_BASE_QWEN3" \
    --model-name "qwen3_1.7b" \
    --train-parquet "$SEARCH_PROCESSED_ROOT/train.parquet"
