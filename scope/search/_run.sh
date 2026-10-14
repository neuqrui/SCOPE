#!/usr/bin/env bash
# SCOPE Search RL. Batch, KL, and environment settings match
# examples/sdar_trainer/run_search_*.sh. The method adds step_skill,
# an OPSD gate, and a length reward. Weights come from scope/sft.
# The retrieval server on SEARCH_URL must already be up. run_search_serial.sh starts it.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TASK_KIND=SEARCH
# shellcheck disable=SC1091
source "$PROJECT_ROOT/scope/env.sh"

: "${SCOPE_TASK:=search}"
: "${SCOPE_MODEL_TAG:?set SCOPE_MODEL_TAG}"
SCOPE_SFT_DIR="${SCOPE_SFT_DIR:-$SCOPE_SFT_ROOT/${SCOPE_TASK}_${SCOPE_MODEL_TAG}}"
scope_init_method

export HIGHLIGHT_CONFIGS='<search>:0,0,255;</search>:0,0,255;<information>:255,0,0;</information>:255,0,0'

ENGINE=vllm
extra_args=("$@")
if [[ ${#extra_args[@]} -gt 0 && "${extra_args[0]}" != *=* ]]; then
    ENGINE="${extra_args[0]}"
    extra_args=("${extra_args[@]:1}")
fi
n_gpus="${N_GPUS_PER_NODE:-4}"
train_data_size=128
val_data_size=512
group_size=8
max_steps=4
skill_all=false
ana_max_prompt_length=16384
ana_max_completion_tokens=512
ana_max_model_len=$((ana_max_prompt_length + ana_max_completion_tokens))
model_path="$(scope_latest_sft "$SCOPE_SFT_DIR")"
train_data="$SEARCH_PROCESSED_ROOT/train.parquet"
val_data="$SEARCH_PROCESSED_ROOT/test.parquet"

thinking_args=()
if [[ "${SCOPE_DISABLE_THINKING:-0}" == 1 ]]; then
    thinking_args+=(+data.apply_chat_template_kwargs.enable_thinking=False)
fi

echo "SCOPE Search RL"
echo "  model:      $model_path"
echo "  experiment: $experiment_name"

python3 -m verl.trainer.main_opsd_kl \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_data" \
    data.val_files="$val_data" \
    data.train_batch_size="$train_data_size" \
    data.val_batch_size="$val_data_size" \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    data.return_raw_chat=True \
    "${thinking_args[@]}" \
    actor_rollout_ref.model.path="$model_path" \
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
    actor_rollout_ref.rollout.name="$ENGINE" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.max_model_len="$ana_max_model_len" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.01 \
    algorithm.use_kl_in_reward=False \
    algorithm.opsd_kl.opsd_coef="$opsd_coef" \
    algorithm.opsd_kl.objective="$opsd_objective" \
    algorithm.opsd_kl.gate="$opsd_gate" \
    algorithm.opsd_kl.gate_beta="$opsd_gate_beta" \
    algorithm.opsd_kl.advantage_clip="$opsd_adv_clip" \
    +algorithm.opsd_kl.skills_dir=skills/search \
    +algorithm.opsd_kl.skill_all="$skill_all" \
    +algorithm.opsd_kl.skill_source="$skill_source" \
    +algorithm.opsd_kl.ana_backend="$ana_backend" \
    +algorithm.opsd_kl.ana_max_prompt_length="$ana_max_prompt_length" \
    +algorithm.opsd_kl.ana_max_completion_tokens="$ana_max_completion_tokens" \
    +algorithm.opsd_kl.ana_loss_mode="$ana_loss_mode" \
    env.env_name=search \
    env.seed=0 \
    env.max_steps="$max_steps" \
    env.history_length=4 \
    env.search.search_url='http://0.0.0.0:8000/retrieve' \
    reward_model.success_length_bonus_coef="$success_length_bonus_coef" \
    reward_model.success_length_adv_coef="$success_length_adv_coef" \
    reward_model.success_length_pairwise_tau="$success_length_pairwise_tau" \
    reward_model.success_length_max_steps="$max_steps" \
    env.rollout.n="$group_size" \
    trainer.critic_warmup=0 \
    trainer.logger=['console','swanlab'] \
    trainer.project_name='verl_agent_search' \
    trainer.experiment_name="$experiment_name" \
    trainer.n_gpus_per_node="$n_gpus" \
    trainer.ray_wait_register_center_timeout=600 \
    trainer.nnodes=1 \
    trainer.save_freq=150 \
    trainer.test_freq=150 \
    trainer.total_epochs=180 \
    trainer.total_training_steps=180 \
    trainer.val_before_train=False \
    "${extra_args[@]}"
