#!/usr/bin/env bash
# Pass@k on the WebShop validation split used by training.
# 128 goals x 8 rollouts, temperature 0.7, vLLM.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TASK_KIND=WEBSHOP
source "$PROJECT_ROOT/examples/_sdar_env.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export RAY_TMPDIR="${RAY_TMPDIR:-work/tmp/ray_passk}"
mkdir -p "$RAY_TMPDIR"
unset RAY_ADDRESS
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export JAVA_HOME="${CONDA_PREFIX:-$ENV_PREFIX}/lib/jvm"
export JVM_PATH="$JAVA_HOME/lib/server/libjvm.so"
export PATH="$JAVA_HOME/bin:$PATH"
export _JAVA_OPTIONS="-XX:+UseSerialGC -Xss512k -Xmx1g"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

BASE_MODEL="${MODEL_PATH:-work/models/Qwen2.5-3B-Instruct}"
CKPT="${CKPT:-work/ckpt/verl_agent_webshopv1/grpo_qwen2.5_3b/global_step_150}"

cd "$PROJECT_ROOT"
python3 -m script.eval_webshop_passk \
    --models "$CKPT" "$BASE_MODEL" \
    "$@"
