#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TASK_KIND=ALFWORLD
source "$PROJECT_ROOT/examples/_sdar_env.sh"

ENGINE=${ENGINE:-vllm}
if [[ $# -gt 0 && "$1" != *=* ]]; then
    ENGINE=$1
    shift
fi

: "${TEACHER_MODEL_PATH:?Set TEACHER_MODEL_PATH to a Hugging Face teacher model path}"

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}
DATA_ROOT=${DATA_ROOT:-$DATA_ROOT/verl-agent}
DATA_DIR=${DATA_DIR:-$DATA_ROOT/text}
OPD_COEF=${OPD_COEF:-0.01}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}
TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE:-2}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-8}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-8}
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.45}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-grpo_opd_qwen2.5_3b}

if [[ -z "${ALFWORLD_DATA:-}" ]]; then
    if [[ -d "$HOME/.cache/alfworld/json_2.1.1" ]]; then
        export ALFWORLD_DATA="$HOME/.cache/alfworld"
    else
        export ALFWORLD_DATA="$HOME/data/alfworld"
    fi
fi

python3 -m examples.data_preprocess.prepare \
    --local_dir "$DATA_ROOT/verl-agent/" \
    --mode text \
    --local_dir "$DATA_ROOT" \
    --train_data_size 16 \
    --val_data_size 128

teacher_model_path=$TEACHER_MODEL_PATH
opd_coef=$OPD_COEF

python3 -m verl.trainer.main_grpo_opd_trainer \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/test.parquet" \
    data.train_batch_size=16 \
    data.val_batch_size=128 \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_SIZE_PER_GPU" \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$TENSOR_MODEL_PARALLEL_SIZE" \
    actor_rollout_ref.rollout.name="$ENGINE" \
    actor_rollout_ref.rollout.gpu_memory_utilization="$VLLM_GPU_MEMORY_UTILIZATION" \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU" \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    +algorithm.opd.opd_coef=$opd_coef \
    +algorithm.opd.teacher_model_path=$teacher_model_path \
    +algorithm.opd.teacher_log_prob_micro_batch_size_per_gpu="$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU" \
    +algorithm.opd.skills_dir=skills/alfworld \
    +algorithm.opd.skill_all=False \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=50 \
    env.rollout.n=8 \
    env.resources_per_worker.num_cpus=0.1 \
    trainer.critic_warmup=0 \
    trainer.logger="['console','wandb']" \
    trainer.project_name=verl_agent_alfworld \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.n_gpus_per_node="$N_GPUS_PER_NODE" \
    trainer.ray_wait_register_center_timeout=600 \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
    trainer.total_epochs=150 \
    trainer.val_before_train=False \
    "$@"
