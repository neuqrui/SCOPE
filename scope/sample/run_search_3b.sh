#!/usr/bin/env bash
# 180 training items, 8 rollouts, base model qwen2.5_3b.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TASK_KIND=SEARCH
source "$PROJECT_ROOT/scope/env.sh"
if [[ -n "${SAMPLE_GPUS:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="$SAMPLE_GPUS"
fi
unset RAY_ADDRESS
export RAY_TMPDIR="${RAY_TMPDIR:-$TMPDIR/ray_scope_search_3b}"
mkdir -p "$RAY_TMPDIR"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export SCOPE_DISABLE_THINKING=0
OUT="${OUT:-$SCOPE_SAMPLE_ROOT/search_qwen2.5_3b}"
mkdir -p "$OUT"
cd "$PROJECT_ROOT"
python3 "$SCRIPT_DIR/eval_search_train_sample.py" \
    --output-dir "$OUT" \
    --model-path "$SCOPE_BASE_3B" \
    --model-name "qwen2.5_3b" \
    --train-parquet "$SEARCH_PROCESSED_ROOT/train.parquet"
