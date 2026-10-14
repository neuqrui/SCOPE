#!/usr/bin/env bash
# Analyzer SFT on Qwen2.5-3B-Instruct.
# Each example is system + user + one assistant completion, including empty step_skills.
#
# CUDA_VISIBLE_DEVICES=0,1 bash examples/sft/analyzer/run_qwen25_3b.sh

set -x
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
source "$PROJECT_ROOT/examples/_sdar_env.sh"

export CUDA_VISIBLE_DEVICES="${SFT_CUDA_VISIBLE_DEVICES:-0,1}"
unset RAY_ADDRESS
export PYTHONUNBUFFERED=1

MODEL_PATH="${SFT_MODEL_PATH:-$MODEL_PATH}"
JSONL_PATH="${JSONL_PATH:-$PROJECT_ROOT/all_sft_data.jsonl}"
VAL_JSONL="${VAL_JSONL:-}"
DATA_DIR="${DATA_DIR:-/data/qirui/SDAR/sft/analyzer_qwen2.5_3b}"
SAVE_DIR="${SAVE_DIR:-$CKPT_ROOT/sft/analyzer_qwen2.5_3b}"
MAX_RESPONSE_TOKENS="${MAX_RESPONSE_TOKENS:-4096}"
MAX_LENGTH="${MAX_LENGTH:-12288}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-analyzer_sft_qwen2.5_3b}"
# 1 wraps thinking_content in <think> and trains it with the JSON. 0 trains the JSON only.
INCLUDE_THINKING="${INCLUDE_THINKING:-0}"

IFS=',' read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
NPROC=${#GPUS[@]}

THINK_ARGS=()
if [[ "$INCLUDE_THINKING" == 1 ]]; then
    THINK_ARGS+=(--include-thinking)
fi
if [[ -n "$VAL_JSONL" ]]; then
    THINK_ARGS+=(--val-jsonl "$VAL_JSONL")
fi

python3 "$SCRIPT_DIR/prepare_analyzer_sft.py" \
    --jsonl "$JSONL_PATH" \
    --model-path "$MODEL_PATH" \
    --out-dir "$DATA_DIR" \
    --max-response-tokens "$MAX_RESPONSE_TOKENS" \
    --max-length "$MAX_LENGTH" \
    "${THINK_ARGS[@]}"

torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC" \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/val.parquet" \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    data.max_length=$MAX_LENGTH \
    data.truncation=error \
    data.train_batch_size=16 \
    data.micro_batch_size_per_gpu=1 \
    model.partial_pretrain="$MODEL_PATH" \
    model.enable_gradient_checkpointing=true \
    model.trust_remote_code=true \
    optim.lr=1e-5 \
    trainer.default_local_dir="$SAVE_DIR" \
    trainer.default_hdfs_dir=null \
    trainer.project_name=analyzer-sft \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.logger=['console','swanlab'] \
    trainer.total_epochs=3 \
    "$@"
