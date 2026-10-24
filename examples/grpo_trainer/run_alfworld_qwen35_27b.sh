set -x

# ALFWorld GRPO for Qwen3.5-27B on 2 nodes x 8 GPUs.
# Start Ray yourself, then run this on the head. _sdar_env.sh clears RAY_ADDRESS,
# so pass it on the command line:
#   RAY_ADDRESS=<head_ip>:6379 bash examples/grpo_trainer/run_alfworld_qwen35_27b.sh
#
# Head:   ray start --head --port=6379 --num-gpus=8 --temp-dir=work/tmp/ray_af_qwen35
# Worker: ray start --address=<head_ip>:6379 --num-gpus=8
# Model weights and ALFWORLD_DATA must be visible at the same path on both nodes.

RAY_ADDRESS_SAVE="${RAY_ADDRESS:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TASK_KIND=ALFWORLD
source "$PROJECT_ROOT/examples/_sdar_env.sh"

if [[ -n "$RAY_ADDRESS_SAVE" ]]; then
    export RAY_ADDRESS="$RAY_ADDRESS_SAVE"
fi
: "${RAY_ADDRESS:?Set RAY_ADDRESS to the Ray head, for example 10.0.0.1:6379}"

export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

ENGINE=${1:-vllm}
NNODES=${NNODES:-2}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}
num_cpus_per_env_worker=0.1
max_steps=50

train_data_size=16
val_data_size=128
group_size=8
total_epochs=150

# Do not inherit .env MODEL_PATH (Qwen2.5-3B).
model_path="${QWEN35_27B_PATH:-${MODELS_ROOT:-work/models}/Qwen3.5-27B}"
if [[ ! -f "$model_path/config.json" ]]; then
    echo "Qwen3.5-27B not found: $model_path" >&2
    exit 1
fi
experiment_name="grpo_qwen3.5_27b"

python3 -m examples.data_preprocess.prepare \
    --local_dir "$DATA_ROOT/verl-agent/" \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_ROOT/verl-agent/text/train.parquet \
    data.val_files=$DATA_ROOT/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    data.trust_remote_code=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.max_model_len=2560 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=$max_steps \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.logger=['console','swanlab'] \
    trainer.project_name='verl_agent_alfworld' \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
    trainer.nnodes=$NNODES \
    trainer.ray_wait_register_center_timeout=600 \
    trainer.save_freq=50 \
    trainer.test_freq=5 \
    trainer.total_epochs=$total_epochs \
    trainer.val_before_train=True $@
