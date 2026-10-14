#!/usr/bin/env bash
# Single-turn SFT on WebShop steps from GRPO step-150 rejection sampling.
# Each trajectory step is one (prompt, response) pair. The prompt already
# contains the history window used at rollout.
#
# CUDA_VISIBLE_DEVICES=0,1 bash examples/sft/webshop/run_webshop_3b.sh

set -x
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
TASK_KIND=WEBSHOP
source "$PROJECT_ROOT/examples/_sdar_env.sh"

export CUDA_VISIBLE_DEVICES="${SFT_CUDA_VISIBLE_DEVICES:-0,1}"
unset RAY_ADDRESS
export PYTHONUNBUFFERED=1

MODEL_PATH="${SFT_MODEL_PATH:-/data/qirui/SDAR/ckpt/hf/grpo_qwen2.5_3b_step150}"
SUCCESS_PATH="${SUCCESS_PATH:-/data/qirui/SDAR/reject_sample/webshop_grpo_3b_step150/success.jsonl}"
DATA_DIR="${DATA_DIR:-/data/qirui/SDAR/sft/webshop_grpo_3b_step150}"
SAVE_DIR="${SAVE_DIR:-$CKPT_ROOT/sft/webshop_grpo_3b_step150}"
MAX_LENGTH=4608

IFS=',' read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
NPROC=${#GPUS[@]}

python3 "$SCRIPT_DIR/prepare_webshop_sft.py" \
    --success-path "$SUCCESS_PATH" \
    --model-path "$MODEL_PATH" \
    --out-dir "$DATA_DIR" \
    --max-length "$MAX_LENGTH" \
    --val-trajectories 64 \
    --seed 0

torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC" \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/val.parquet" \
    data.prompt_key=prompt \
    data.response_key=response \
    data.prompt_dict_keys=null \
    data.response_dict_keys=null \
    data.max_length=$MAX_LENGTH \
    data.truncation=error \
    data.train_batch_size=64 \
    data.micro_batch_size_per_gpu=4 \
    data.multiturn.enable=false \
    model.partial_pretrain="$MODEL_PATH" \
    model.enable_gradient_checkpointing=true \
    model.trust_remote_code=true \
    optim.lr=1e-6 \
    trainer.default_local_dir="$SAVE_DIR" \
    trainer.default_hdfs_dir=null \
    trainer.project_name=verl_agent_webshopv1 \
    trainer.experiment_name=webshop_sft_grpo_3b_step150 \
    trainer.logger=['console','swanlab'] \
    trainer.total_epochs=1 \
    "$@"
