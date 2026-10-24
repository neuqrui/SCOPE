#!/usr/bin/env bash
# 180 WebShop training goals x 8 rollouts x 5 models, temperature 0.7.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TASK_KIND=WEBSHOP
source "$PROJECT_ROOT/examples/_sdar_env.sh"

# .env exports a training GPU list. This sampler uses one card and sizes
# its KV cache from whatever that card is holding at startup.
export CUDA_VISIBLE_DEVICES="${SAMPLE_CUDA_VISIBLE_DEVICES:-2}"
export RAY_TMPDIR="${RAY_TMPDIR:-work/tmp/ray_webshop_train180}"
mkdir -p "$RAY_TMPDIR"
unset RAY_ADDRESS
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# Leave unset so the sampler sizes KV cache from the card's current free memory.
unset VLLM_GPU_MEMORY_UTILIZATION
export JAVA_HOME="${CONDA_PREFIX:-$ENV_PREFIX}/lib/jvm"
export JVM_PATH="$JAVA_HOME/lib/server/libjvm.so"
export PATH="$JAVA_HOME/bin:$PATH"
export _JAVA_OPTIONS="-XX:+UseSerialGC -Xss512k -Xmx1g"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

cd "$PROJECT_ROOT"
python3 -m script.eval_webshop_train_sample "$@"
