set -x
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TASK_KIND=ALFWORLD
source "$PROJECT_ROOT/examples/_sdar_env.sh"

ENGINE=${1:-vllm}
# Conda env is activated by examples/_sdar_env.sh (CONDA_ENV=sdar for ALFWORLD).
# export VLLM_ATTENTION_BACKEND=XFORMERS
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

num_cpus_per_env_worker=0.1 # The CPU resource allocated for each environment worker. If you want to use less CPU resources, you can decrease this value.
max_steps=50
# LENGTH_REWARD=none     : episode reward stays 10 or 0.
# LENGTH_REWARD=adv      : task-relative length advantage after GRPO, coef 1.
# LENGTH_REWARD=bonus    : add 5 * (max_steps - steps) / max_steps on successes only.
# LENGTH_REWARD=pairwise : soft pairwise efficiency, tau=0.1. Score is 10+5*b on successes, then GRPO.
LENGTH_REWARD=${LENGTH_REWARD:-adv}
case "$LENGTH_REWARD" in
    none)
        success_length_adv_coef=0
        success_length_bonus_coef=0
        success_length_pairwise_tau=0
        ;;
    adv)
        success_length_adv_coef=1.0
        success_length_bonus_coef=0
        success_length_pairwise_tau=0
        ;;
    bonus)
        success_length_adv_coef=0
        success_length_bonus_coef=5.0
        success_length_pairwise_tau=0
        ;;
    pairwise)
        success_length_adv_coef=0
        success_length_bonus_coef=0
        success_length_pairwise_tau=0.1
        ;;
    *)
        echo "LENGTH_REWARD must be none, adv, bonus, or pairwise, got ${LENGTH_REWARD}" >&2
        exit 1
        ;;
esac

train_data_size=16
val_data_size=128
group_size=8
# wandb replaced by swanlab; see .env SWANLAB_API_KEY
# pip3 install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir

# We only use data preparation to indicate the modality and the data size.
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
    actor_rollout_ref.model.path=$MODEL_PATH \
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
    trainer.experiment_name=grpo_qwen2.5_3b_length_${LENGTH_REWARD} \
    trainer.n_gpus_per_node=4 \
    trainer.ray_wait_register_center_timeout=600 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=180 \
    trainer.val_before_train=True $@
