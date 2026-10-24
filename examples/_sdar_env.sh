#!/usr/bin/env bash
# Shared bootstrap for SDAR training / retrieval scripts.
# Source after SCRIPT_DIR/PROJECT_ROOT are set.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_FILE="${ENV_FILE:-$PROJECT_ROOT/.env}"

if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

TASK_KIND="${TASK_KIND:-ALFWORLD}"
case "$TASK_KIND" in
    WEBSHOP)
        CONDA_ENV="${CONDA_ENV_WEBSHOP:-scope-webshop}"
        ;;
    RETRIEVER)
        CONDA_ENV="${CONDA_ENV_RETRIEVER:-scope-retriever}"
        ;;
    SEARCH|ALFWORLD|*)
        CONDA_ENV="${CONDA_ENV:-scope}"
        ;;
esac

# Optional prefix of conda env directories. Leave unset to activate by env name.
ENV_PREFIX=""
if [[ -n "${CONDA_ENV_BIN_ROOT:-}" ]]; then
    ENV_PREFIX="${CONDA_ENV_BIN_ROOT}/${CONDA_ENV}"
fi

if [[ -n "$CONDA_ENV" && "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" && "${CONDA_PREFIX:-}" != "$ENV_PREFIX" ]]; then
    if command -v conda >/dev/null 2>&1; then
        set +u
        eval "$(conda shell.bash hook)"
        if [[ -n "$ENV_PREFIX" && -d "$ENV_PREFIX" ]]; then
            conda activate "$ENV_PREFIX"
        else
            conda activate "$CONDA_ENV"
        fi
        set -u
    fi
fi

export PYTHONNOUSERSITE=1
if [[ -n "$ENV_PREFIX" && -d "$ENV_PREFIX/bin" ]]; then
    export PATH="${ENV_PREFIX}/bin:${PATH:-}"
fi
# Ray AF_UNIX sockets must stay under 107 bytes, so the default tmp dir is short.
export SCOPE_WORK_ROOT="${SCOPE_WORK_ROOT:-$PROJECT_ROOT/work}"
export TMPDIR="${TMPDIR:-$PROJECT_ROOT/tmp}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$PROJECT_ROOT/tmp/pip-cache}"
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/scope-ray}"
mkdir -p "$TMPDIR" "$RAY_TMPDIR" "${SWANLAB_LOG_DIR:-$PROJECT_ROOT/logs/swanlab}"
unset RAY_ADDRESS

MODELS_ROOT="${MODELS_ROOT:-$PROJECT_ROOT/models}"
MODEL_PATH="${MODEL_PATH:-$MODELS_ROOT/Qwen2.5-3B-Instruct}"
MODEL_PATH_QWEN3="${MODEL_PATH_QWEN3:-$MODELS_ROOT/Qwen3-1.7B}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data}"
CKPT_ROOT="${CKPT_ROOT:-$SCOPE_WORK_ROOT/ckpt}"
ALFWORLD_DATA="${ALFWORLD_DATA:-$PROJECT_ROOT/data/alfworld}"
SEARCH_DATA_ROOT="${SEARCH_DATA_ROOT:-$PROJECT_ROOT/data/searchR1}"
SEARCH_PROCESSED_ROOT="${SEARCH_PROCESSED_ROOT:-$PROJECT_ROOT/data/searchR1_processed_direct}"
SEARCH_URL="${SEARCH_URL:-http://127.0.0.1:8000/retrieve}"
export ALFWORLD_DATA MODEL_PATH MODEL_PATH_QWEN3 DATA_ROOT CKPT_ROOT SEARCH_DATA_ROOT SEARCH_PROCESSED_ROOT SEARCH_URL

cd "$PROJECT_ROOT"
