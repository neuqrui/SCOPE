#!/usr/bin/env bash
# Analyzer SFT on Qwen2.5-7B-Instruct for WebShop.
# Data: all_sft_data_v1.jsonl
# (messages with trajectory_summary in the assistant JSON, no thinking tags).
#
# SFT_CUDA_VISIBLE_DEVICES=6,7 bash examples/sft/webshop/run_webshop_7b.sh

set -x
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
# Same sdar env as the Qwen2.5 analyzer SFT. WEBSHOP would select sdar-webshop.
TASK_KIND=ALFWORLD
source "$PROJECT_ROOT/examples/_sdar_env.sh"

# GPUs 2,3,4,5 are occupied. Use only 6 and 7.
export CUDA_VISIBLE_DEVICES="${SFT_CUDA_VISIBLE_DEVICES:-6,7}"
unset RAY_ADDRESS
export PYTHONUNBUFFERED=1

MODEL_PATH="${SFT_MODEL_PATH:-work/models/Qwen2.5-7B-Instruct}"
JSONL_PATH="${JSONL_PATH:-$PROJECT_ROOT/all_sft_data_v1.jsonl}"
VAL_JSONL="${VAL_JSONL:-}"
DATA_DIR="${DATA_DIR:-work/sft/webshop_analyzer_qwen2.5_7b}"
SAVE_DIR="${SAVE_DIR:-$CKPT_ROOT/sft/webshop_analyzer_qwen2.5_7b}"
MAX_RESPONSE_TOKENS="${MAX_RESPONSE_TOKENS:-4096}"
MAX_LENGTH="${MAX_LENGTH:-12288}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-webshop_analyzer_sft_qwen2.5_7b}"
# 1 wraps thinking_content in <think> and trains it with the JSON. 0 trains the JSON only.
INCLUDE_THINKING="${INCLUDE_THINKING:-0}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"

IFS=',' read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
NPROC=${#GPUS[@]}

THINK_ARGS=()
if [[ "$INCLUDE_THINKING" == 1 ]]; then
    THINK_ARGS+=(--include-thinking)
fi
if [[ -n "$VAL_JSONL" ]]; then
    THINK_ARGS+=(--val-jsonl "$VAL_JSONL")
fi

mkdir -p "$SAVE_DIR" "$DATA_DIR"

python3 "$PROJECT_ROOT/examples/sft/alfworld/prepare_alfworld_sft.py" \
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
    trainer.project_name=webshop-analyzer-sft \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.logger=['console','swanlab'] \
    trainer.total_epochs=$TOTAL_EPOCHS \
    "$@"
