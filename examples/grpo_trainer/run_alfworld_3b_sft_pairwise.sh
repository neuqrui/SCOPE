set -x


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TASK_KIND=ALFWORLD
source "$PROJECT_ROOT/examples/_sdar_env.sh"

# Same ALFWorld layout as examples/opsd_kl_trainer/run_alfworld_3b_step_skill.sh,
# without OPSD: SFT init, GRPO, pairwise length reward.
export RAY_TMPDIR=work/tmp/ray_af_grpo_sft
mkdir -p "$RAY_TMPDIR"
unset RAY_ADDRESS
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

ENGINE=${1:-vllm}
export CUDA_VISIBLE_DEVICES=0,1,2,3
num_cpus_per_env_worker=0.1
max_steps=30
history_length=5
# Soft pairwise efficiency, tau=0.1. Score is 10+5*b on successes, then GRPO.
success_length_adv_coef=0
success_length_bonus_coef=0
success_length_pairwise_tau=0.1

train_data_size=16
val_data_size=128
group_size=8
total_epochs=180
# 3-epoch ALFWorld analyzer SFT. Do not inherit .env MODEL_PATH (base Instruct).
model_path="${SFT_MODEL_PATH:-${ALFWORLD_SFT_MODEL_PATH:-work/ckpt/sft/alfworld_analyzer_qwen2.5_3b/global_step_261}}"
if [[ ! -f "$model_path/config.json" ]]; then
    echo "ALFWorld 3B SFT checkpoint not found: $model_path" >&2
    exit 1
fi
experiment_name="grpo_qwen2.5_3b_sft_length_pairwise"

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
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    env.history_length=$history_length \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=$max_steps \
    reward_model.success_length_bonus_coef=$success_length_bonus_coef \
    reward_model.success_length_adv_coef=$success_length_adv_coef \
    reward_model.success_length_pairwise_tau=$success_length_pairwise_tau \
    reward_model.success_length_max_steps=$max_steps \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.logger=['console','swanlab'] \
    trainer.project_name='verl_agent_alfworld' \
    trainer.experiment_name=$experiment_name \
    trainer.resume_mode=disable \
    trainer.n_gpus_per_node=4 \
    trainer.ray_wait_register_center_timeout=600 \
    trainer.nnodes=1 \
    trainer.max_actor_ckpt_to_keep=4 \
    trainer.save_freq=50 \
    trainer.test_freq=5 \
    trainer.total_epochs=$total_epochs \
    trainer.val_before_train=True \
    trainer.rollout_data_dir=$CKPT_ROOT/rollouts/${experiment_name} \
    trainer.validation_data_dir=$CKPT_ROOT/rollouts_val/${experiment_name} \
    trainer.trajectory_dump.freq=1 \
    trainer.trajectory_dump.max_episodes=2 \
    trainer.trajectory_dump.max_teacher_examples=2 \
    trainer.trajectory_dump.write_markdown=true \
    trainer.trajectory_dump.write_html=true $@
