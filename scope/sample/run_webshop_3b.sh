#!/usr/bin/env bash
# 180 training items, 8 rollouts, base model qwen2.5_3b.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TASK_KIND=WEBSHOP
source "$PROJECT_ROOT/scope/env.sh"
if [[ -n "${SAMPLE_GPUS:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="$SAMPLE_GPUS"
fi
unset RAY_ADDRESS
export RAY_TMPDIR="${RAY_TMPDIR:-$TMPDIR/ray_scope_webshop_3b}"
mkdir -p "$RAY_TMPDIR"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
unset VLLM_GPU_MEMORY_UTILIZATION
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export SCOPE_DISABLE_THINKING=0
OUT="${OUT:-$SCOPE_SAMPLE_ROOT/webshop_qwen2.5_3b}"
mkdir -p "$OUT"
cd "$PROJECT_ROOT"
export JAVA_HOME="${CONDA_PREFIX:-$ENV_PREFIX}/lib/jvm"
export JVM_PATH="$JAVA_HOME/lib/server/libjvm.so"
export PATH="$JAVA_HOME/bin:$PATH"
export _JAVA_OPTIONS="-XX:+UseSerialGC -Xss512k -Xmx1g"
python3 -m script.eval_webshop_train_sample \
    --output-dir "$OUT" \
    --model-path "$SCOPE_BASE_3B" \
    --model-name "qwen2.5_3b"
