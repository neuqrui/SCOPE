#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TASK_KIND=WEBSHOP
source "$PROJECT_ROOT/examples/_sdar_env.sh"

ENGINE=${ENGINE:-vllm}
if [[ $# -gt 0 && "$1" != *=* ]]; then
    ENGINE=$1
    shift
fi

export JAVA_HOME="${CONDA_PREFIX:-$ENV_PREFIX}/lib/jvm"
export JVM_PATH="$JAVA_HOME/lib/server/libjvm.so"
export PATH=$JAVA_HOME/bin:$PATH
export _JAVA_OPTIONS="-XX:+UseSerialGC -Xss512k -Xmx1g"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export DATASETS_MAX_WORKERS=1

export CUDA_VISIBLE_DEVICES=4,5

OPD_COEF=${OPD_COEF:-1}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-2}
TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE:-2}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-8}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-8}
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.45}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-opd_webshop_qwen3_1.7b}
# wandb replaced by swanlab; see .env SWANLAB_API_KEY

train_data_size=16
val_data_size=128
group_size=8

python3 -m examples.data_preprocess.prepare \
    --local_dir "$DATA_ROOT/verl-agent/" \
    --mode text \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size

teacher_model_path=./checkpoints/verl_agent_webshopv1/grpo_qwen3_1.7b/global_step_150/actor/huggingface
MODEL_PATH=work/models/Qwen3-1.7B
opd_coef=$OPD_COEF

python3 -m verl.trainer.main_opd_trainer \
    data.train_files=$DATA_ROOT/verl-agent/text/train.parquet \
    data.val_files=$DATA_ROOT/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_SIZE_PER_GPU" \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
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
    +algorithm.opd.opd_coef=$opd_coef \
    +algorithm.opd.teacher_model_path="$teacher_model_path" \
    +algorithm.opd.teacher_log_prob_micro_batch_size_per_gpu="$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU" \
    +algorithm.opd.skills_dir=skills/webshop \
    +algorithm.opd.skill_all=False \
    env.env_name=Webshop \
    env.seed=0 \
    env.max_steps=5 \
    env.val_max_steps=${VAL_MAX_STEPS:-15} \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=0.1 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','swanlab'] \
    trainer.project_name='verl_agent_webshopv1' \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.n_gpus_per_node="$N_GPUS_PER_NODE" \
    trainer.ray_wait_register_center_timeout=600 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=150 \
    trainer.val_before_train=True \
    trainer.rollout_data_dir=$CKPT_ROOT/rollouts/${EXPERIMENT_NAME} \
    trainer.validation_data_dir=$CKPT_ROOT/rollouts_val/${EXPERIMENT_NAME} \
    trainer.trajectory_dump.freq=1 \
    trainer.trajectory_dump.max_episodes=2 \
    trainer.trajectory_dump.max_teacher_examples=2 \
    trainer.trajectory_dump.write_markdown=true \
    trainer.trajectory_dump.write_html=true \
    "$@"
