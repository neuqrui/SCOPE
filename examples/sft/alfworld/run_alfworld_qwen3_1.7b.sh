#!/usr/bin/env bash
# Analyzer SFT on Qwen3-1.7B for ALFWorld.
# Data: messages [system, user, assistant] with step_skills JSON (same schema
# as the WebShop analyzer SFT / alfworld_sft_data). No thinking tags by default.
#
# SFT_CUDA_VISIBLE_DEVICES=4,5 bash examples/sft/alfworld/run_alfworld_qwen3_1.7b.sh

set -x
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
TASK_KIND=ALFWORLD
source "$PROJECT_ROOT/examples/_sdar_env.sh"

export CUDA_VISIBLE_DEVICES="${SFT_CUDA_VISIBLE_DEVICES:-4,5}"
unset RAY_ADDRESS
export PYTHONUNBUFFERED=1

# Base instruct model. Do not inherit .env MODEL_PATH (may point at an analyzer ckpt).
MODEL_PATH="${SFT_MODEL_PATH:-work/models/Qwen3-1.7B}"
JSONL_PATH="${JSONL_PATH:-work/alfworld_sft_data/train.jsonl}"
VAL_JSONL="${VAL_JSONL:-}"
DATA_DIR="${DATA_DIR:-work/sft/alfworld_analyzer_qwen3_1.7b}"
SAVE_DIR="${SAVE_DIR:-$CKPT_ROOT/sft/alfworld_analyzer_qwen3_1.7b}"
MAX_RESPONSE_TOKENS="${MAX_RESPONSE_TOKENS:-4096}"
MAX_LENGTH="${MAX_LENGTH:-12288}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-alfworld_analyzer_sft_qwen3_1.7b}"
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

python3 "$SCRIPT_DIR/prepare_alfworld_sft.py" \
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
    trainer.project_name=alfworld-analyzer-sft \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.logger=['console','swanlab'] \
    trainer.total_epochs=$TOTAL_EPOCHS \
    "$@"
