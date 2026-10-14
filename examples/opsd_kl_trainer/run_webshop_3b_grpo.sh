set -x


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TASK_KIND=WEBSHOP
source "$PROJECT_ROOT/examples/_sdar_env.sh"

ENGINE=${1:-vllm}

num_cpus_per_env_worker=0.1

# GRPO + sampled-token OPSD (no SDAR gate).
# L = L_GRPO + opsd_coef * agg(-stopgrad(logπ_T - logπ_S) * logπ_S) + kl_coef * L_KL
opsd_coef=0.01
opsd_objective=sampled_kl
skill_all=false
# skill_bank: static keyword skill bank.
# seed_ana: after each trajectory, an analyzer writes one episode skill (SEED episode_only prompt).
skill_source=seed_ana
ana_backend=policy_vllm
# Same split as SEED WebShop: student prompts stay at 4096, vLLM context fits the analyzer.
ana_max_prompt_length=16384
ana_max_completion_tokens=4096
ana_max_model_len=$((ana_max_prompt_length + ana_max_completion_tokens))

train_data_size=16
val_data_size=128
group_size=8
# Folder that contains actor/. The loader appends actor/ itself.
resume_from_path=/home/liuqirui/SDAR/checkpoints/verl_agent_webshopv1/opd_webshop_qwen2.5_3b_max15/global_step_30
# One batch per epoch. The counter resumes at 30 and the next update is step 31.
# 180 leaves 150 new updates before the loop stops.
total_epochs=180
experiment_name="grpo_opsd_kl_qwen2.5_3b_coef${opsd_coef}_${opsd_objective}_skillall${skill_all}_${skill_source}_from_opd30"
# wandb replaced by swanlab; see .env SWANLAB_API_KEY

python3 -m examples.data_preprocess.prepare \
    --local_dir "$DATA_ROOT/verl-agent/" \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size

python3 -m verl.trainer.main_opsd_kl \
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
    algorithm.opsd_kl.opsd_coef=$opsd_coef \
    algorithm.opsd_kl.objective=$opsd_objective \
    +algorithm.opsd_kl.skills_dir=skills/webshop \
    +algorithm.opsd_kl.skill_all=$skill_all \
    +algorithm.opsd_kl.skill_source=$skill_source \
    +algorithm.opsd_kl.ana_backend=$ana_backend \
    +algorithm.opsd_kl.ana_max_prompt_length=$ana_max_prompt_length \
    +algorithm.opsd_kl.ana_max_completion_tokens=$ana_max_completion_tokens \
    env.env_name=Webshop \
    env.seed=0 \
    env.max_steps=15 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.logger=['console','swanlab'] \
    trainer.project_name='verl_agent_webshopv1' \
    trainer.experiment_name=$experiment_name \
    trainer.resume_mode=resume_path \
    trainer.resume_from_path=$resume_from_path \
    trainer.n_gpus_per_node=2 \
    trainer.ray_wait_register_center_timeout=600 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
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
