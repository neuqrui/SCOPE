set -x
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TASK_KIND=SEARCH
source "$PROJECT_ROOT/examples/_sdar_env.sh"

# Same runtime as examples/opsd_kl_trainer/run_search_3b_step_skill.sh.
# Retrieval is the already-running server; this script does not launch it.
# Model stays the Qwen3-1.7B base. Do not inherit .env MODEL_PATH.
export RAY_TMPDIR=work/tmp/ray_search_qwen3
mkdir -p "$RAY_TMPDIR"
unset RAY_ADDRESS
export CUDA_VISIBLE_DEVICES="${SEARCH_CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export HIGHLIGHT_CONFIGS='<search>:0,0,255;</search>:0,0,255;<information>:255,0,0;</information>:255,0,0'

ENGINE=${1:-vllm}
IFS=',' read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
N_GPUS=${#GPUS[@]}
MODEL_PATH_QWEN3="${SEARCH_BASE_MODEL:-work/models/Qwen3-1.7B}"

train_data_size=128
val_data_size=512
group_size=8

TRAIN_DATA="$SEARCH_PROCESSED_ROOT/train.parquet"
VAL_DATA="$SEARCH_PROCESSED_ROOT/test.parquet"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path=$MODEL_PATH_QWEN3 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.01 \
    algorithm.use_kl_in_reward=False \
    env.env_name=search \
    env.seed=0 \
    env.max_steps=4 \
    env.rollout.n=$group_size \
    env.history_length=4 \
    env.search.search_url="$SEARCH_URL" \
    trainer.critic_warmup=0 \
    trainer.logger=['console','swanlab'] \
    trainer.project_name='verl_agent_search' \
    trainer.experiment_name='grpo_qwen3_1.7b_instruct' \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.ray_wait_register_center_timeout=600 \
    trainer.nnodes=1 \
    trainer.save_freq=150 \
    trainer.test_freq=150 \
    trainer.total_training_steps=150 \
    trainer.val_before_train=False $@
