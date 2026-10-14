#!/usr/bin/env bash
# Install a separate env that can serve Qwen3.5, then score MiMo on the
# WebShop validation batch from examples/opsd_kl_trainer/run_webshop_3b.sh.
#
# Uses free GPU 0 by default. The WebShop env stays in sdar-webshop.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MIMO_ENV="${MIMO_ENV:-/data/qirui/envs/sdar-mimo}"
MIMO_MODEL="${MIMO_MODEL:-/data/qirui/models/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B}"
MIMO_CUDA="${MIMO_CUDA:-0}"
PORT="${PORT:-8100}"
export PYTHONNOUSERSITE=1

if [[ ! -x "$MIMO_ENV/bin/python" ]]; then
    conda create -y -p "$MIMO_ENV" python=3.11 pip
fi
if ! "$MIMO_ENV/bin/python" -c "import vllm" >/dev/null 2>&1; then
    "$MIMO_ENV/bin/pip" install -U "vllm==0.31.0"
fi

TASK_KIND=WEBSHOP
source "$PROJECT_ROOT/examples/_sdar_env.sh"
cd "$PROJECT_ROOT"
# .env sets MODEL_PATH and CUDA_VISIBLE_DEVICES for the training job.
export MODEL_PATH="$MIMO_MODEL"
export CUDA_VISIBLE_DEVICES="$MIMO_CUDA"
export RAY_TMPDIR="/data/qirui/SDAR/tmp/mimo_val"

SERVER_LOG="${SERVER_LOG:-/data/qirui/SDAR/reject_sample/webshop_mimo_val/vllm.log}"
mkdir -p "$(dirname "$SERVER_LOG")"
if ! curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null; then
    # sdar-webshop's nvcc is too old for FlashInfer JIT (--compress-mode=size).
    CUDA13_BIN="$MIMO_ENV/lib/python3.11/site-packages/nvidia/cu13/bin"
    PATH="${CUDA13_BIN}:${MIMO_ENV}/bin:${PATH}" \
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    VLLM_USE_FLASHINFER_SAMPLER=0 \
    "$MIMO_ENV/bin/vllm" serve "$MODEL_PATH" \
        --host 127.0.0.1 \
        --port "$PORT" \
        --tensor-parallel-size 1 \
        --max-model-len 4608 \
        --gpu-memory-utilization 0.9 \
        --trust-remote-code \
        --language-model-only \
        --gdn-prefill-backend triton \
        >"$SERVER_LOG" 2>&1 &
    echo $! >"$(dirname "$SERVER_LOG")/vllm.pid"
    for _ in $(seq 1 180); do
        if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null; then
            break
        fi
        sleep 5
    done
    curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null
fi

python3 -m script.eval_webshop_mimo_val \
    --model-path "$MODEL_PATH" \
    --base-url "http://127.0.0.1:${PORT}/v1" \
    "$@"
