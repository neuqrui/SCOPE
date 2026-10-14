#!/usr/bin/env bash
# Analyzer SFT in the ALFWorld chat format: messages [system, user, assistant].
# Three epochs. The RL scripts load the latest global_step under the save dir.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
: "${SCOPE_TASK:?set SCOPE_TASK}"
case "$SCOPE_TASK" in
    alfworld) TASK_KIND=ALFWORLD ;;
    webshop) TASK_KIND=WEBSHOP ;;
    search) TASK_KIND=SEARCH ;;
    *) echo "SCOPE_TASK must be alfworld, webshop, or search" >&2; exit 1 ;;
esac
# shellcheck disable=SC1091
source "$PROJECT_ROOT/scope/env.sh"

: "${SCOPE_MODEL_TAG:?set SCOPE_MODEL_TAG}"
: "${SCOPE_BASE_MODEL:?set SCOPE_BASE_MODEL}"
# One jsonl per base model, next to that model's own 180-example sample.
SCOPE_SFT_JSONL="${SCOPE_SFT_JSONL:-$SCOPE_SAMPLE_ROOT/${SCOPE_TASK}_${SCOPE_MODEL_TAG}/train.jsonl}"

if [[ ! -f "$SCOPE_SFT_JSONL" ]]; then
    echo "SFT jsonl not found: $SCOPE_SFT_JSONL" >&2
    echo "Each base model is trained on its own sample. After scope/sample/run_${SCOPE_TASK}_*.sh," >&2
    echo "put this model's messages jsonl at the path above." >&2
    echo "Format: one JSON object per line with messages [system, user, assistant]." >&2
    exit 1
fi
if [[ "$SCOPE_BASE_MODEL" != Qwen/* && ! -f "$SCOPE_BASE_MODEL/config.json" ]]; then
    echo "Base model not found: $SCOPE_BASE_MODEL" >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${SFT_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0,1}}"
unset RAY_ADDRESS
export PYTHONUNBUFFERED=1

epochs="${SCOPE_SFT_EPOCHS:-3}"
data_dir="${SCOPE_SFT_DATA_DIR:-/data/qirui/SDAR/sft/scope/${SCOPE_TASK}_${SCOPE_MODEL_TAG}}"
save_dir="${SCOPE_SFT_DIR:-$SCOPE_SFT_ROOT/${SCOPE_TASK}_${SCOPE_MODEL_TAG}}"
max_response_tokens="${MAX_RESPONSE_TOKENS:-4096}"
max_length="${MAX_LENGTH:-12288}"
experiment_name="scope_sft_${SCOPE_TASK}_${SCOPE_MODEL_TAG}_epoch${epochs}"

IFS=',' read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
nproc=${#GPUS[@]}

echo "SCOPE SFT"
echo "  task:       $SCOPE_TASK"
echo "  base:       $SCOPE_BASE_MODEL"
echo "  jsonl:      $SCOPE_SFT_JSONL"
echo "  save:       $save_dir"
echo "  epochs:     $epochs"

python3 "$PROJECT_ROOT/examples/sft/alfworld/prepare_alfworld_sft.py" \
    --jsonl "$SCOPE_SFT_JSONL" \
    --model-path "$SCOPE_BASE_MODEL" \
    --out-dir "$data_dir" \
    --max-response-tokens "$max_response_tokens" \
    --max-length "$max_length"

torchrun --standalone --nnodes=1 --nproc_per_node="$nproc" \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files="$data_dir/train.parquet" \
    data.val_files="$data_dir/val.parquet" \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    data.max_length="$max_length" \
    data.truncation=error \
    data.train_batch_size=16 \
    data.micro_batch_size_per_gpu=1 \
    model.partial_pretrain="$SCOPE_BASE_MODEL" \
    model.enable_gradient_checkpointing=true \
    model.trust_remote_code=true \
    optim.lr=1e-5 \
    trainer.default_local_dir="$save_dir" \
    trainer.default_hdfs_dir=null \
    trainer.project_name=scope-sft \
    trainer.experiment_name="$experiment_name" \
    trainer.logger=['console','swanlab'] \
    trainer.total_epochs="$epochs" \
    "$@"
