set -x


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TASK_KIND=WEBSHOP
source "$PROJECT_ROOT/examples/_sdar_env.sh"

ENGINE=${1:-vllm}

num_cpus_per_env_worker=0.1

# OPSD hyperparameters (SDAR without GRPO loss)
sdar_coef=0.01
gate_beta=5.0
skill_all=false
# skill_bank: static keyword skill bank.
# seed_ana: after each trajectory, an analyzer writes one episode skill (SEED episode_only prompt).
skill_source=skill_bank
ana_backend=policy_vllm
# Same split as SEED WebShop: student prompts stay at 4096, vLLM context fits the analyzer.
ana_max_prompt_length=16384
ana_max_completion_tokens=4096
ana_max_model_len=$((ana_max_prompt_length + ana_max_completion_tokens))

train_data_size=16
val_data_size=128
group_size=8
experiment_name="opsd_qwen2.5_3b_coef${sdar_coef}_beta${gate_beta}_skillall${skill_all}_${skill_source}"
# wandb replaced by swanlab; see .env SWANLAB_API_KEY

python3 -m examples.data_preprocess.prepare \
    --local_dir "$DATA_ROOT/verl-agent/" \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size

python3 -m verl.trainer.main_sdar \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_ROOT/verl-agent/text/train.parquet \
    data.val_files=$DATA_ROOT/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.max_model_len=$ana_max_model_len \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    +actor_rollout_ref.actor.pg_loss_coef=0 \
    +algorithm.sdar.sdar_coef=$sdar_coef \
    +algorithm.sdar.gate_beta=$gate_beta \
    +algorithm.sdar.skills_dir=skills/webshop \
    +algorithm.sdar.skill_all=$skill_all \
    +algorithm.sdar.skill_source=$skill_source \
    +algorithm.sdar.ana_backend=$ana_backend \
    +algorithm.sdar.ana_max_prompt_length=$ana_max_prompt_length \
    +algorithm.sdar.ana_max_completion_tokens=$ana_max_completion_tokens \
    env.env_name=Webshop \
    env.seed=0 \
    env.max_steps=15 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.logger=['console','swanlab'] \
    trainer.project_name='verl_agent_webshopv1' \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=2 \
    trainer.ray_wait_register_center_timeout=600 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=150 \
    trainer.val_before_train=True \
    trainer.rollout_data_dir=$CKPT_ROOT/rollouts/${experiment_name} \
    trainer.validation_data_dir=$CKPT_ROOT/rollouts_val/${experiment_name} \
    trainer.trajectory_dump.freq=1 \
    trainer.trajectory_dump.max_episodes=2 \
    trainer.trajectory_dump.max_teacher_examples=2 \
    trainer.trajectory_dump.write_markdown=true \
    trainer.trajectory_dump.write_html=true $@
