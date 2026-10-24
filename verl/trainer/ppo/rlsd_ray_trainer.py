"""
RLSD (Reinforcement Learning with Self-Distillation) Trainer.

Extends the standard RayPPOTrainer with a teacher forward pass that uses
privileged information (skills) to construct token-level advantages.
"""

from pprint import pprint
import time

import numpy as np
import ray
import torch
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    _timer,
    apply_invalid_action_penalty,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.rlsd_utils import SkillProvider, compute_rlsd_token_advantage
from verl.utils.metric import reduce_metrics
from verl.utils.torch_functional import masked_mean
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
)
from verl.utils.model import compute_position_id_with_mask

from agent_system.multi_turn_rollout import adjust_batch


def build_teacher_batch(
    batch: DataProto,
    skill_provider: SkillProvider,
    tokenizer,
    max_prompt_length: int,
    truncation: str = "error",
):
    """
    Build a teacher batch by prepending privileged skill info to each sample's prompt.

    The teacher sees (x, r) where r is the skill/privileged information.
    We prepend the skill text as a system message before the user prompt,
    then re-tokenize to get teacher input_ids/attention_mask/position_ids.
    The responses remain unchanged.

    Also returns per-sample skill resolution metadata (selected skills / matched tasks /
    skill text) for offline analysis dumps.

    Args:
        batch: The original student batch with prompts and responses.
        skill_provider: SkillProvider instance for loading skills.
        tokenizer: The tokenizer.
        max_prompt_length: Maximum prompt length.
        truncation: Truncation mode.

    Returns:
        teacher_batch: A DataProto with modified input_ids/attention_mask/position_ids
            but the same responses, suitable for computing teacher log probs.
        skill_meta: Dict of np.ndarray object arrays with keys
            selected_skills / matched_tasks / skill_text / skill_all.
    """
    bs = batch.batch["input_ids"].size(0)
    response_length = batch.batch["responses"].size(1)

    teacher_input_ids_list = []
    teacher_attention_mask_list = []
    teacher_position_ids_list = []
    selected_skills_list = []
    matched_tasks_list = []
    skill_text_list = []
    skill_all_list = []

    for i in range(bs):
        # Decode the original prompt (student input minus response)
        original_input_ids = batch.batch["input_ids"][i]
        original_attention_mask = batch.batch["attention_mask"][i]
        prompt_length = original_input_ids.size(0) - response_length

        prompt_ids = original_input_ids[:prompt_length]
        prompt_mask = original_attention_mask[:prompt_length]

        # Find the first non-padding token in prompt
        valid_start = prompt_mask.nonzero(as_tuple=True)[0]
        if len(valid_start) > 0:
            valid_start = valid_start[0].item()
        else:
            valid_start = 0

        valid_prompt_ids = prompt_ids[valid_start:]
        prompt_text = tokenizer.decode(valid_prompt_ids, skip_special_tokens=False)

        # Resolve privileged skill info based on gamefile, data_source, or prompt text
        gamefile = batch.non_tensor_batch.get("gamefile", None)
        data_source = batch.non_tensor_batch.get("data_source", None)
        gf = None
        ds = None
        if gamefile is not None and gamefile[i] is not None:
            gf = gamefile[i] if isinstance(gamefile[i], str) else str(gamefile[i])
        if data_source is not None and data_source[i] is not None:
            ds = data_source[i] if isinstance(data_source[i], str) else str(data_source[i])

        resolved = skill_provider.resolve(gamefile=gf, data_source=ds, prompt_text=prompt_text)
        skill_text = resolved["skill_text"]
        selected_skills_list.append(list(resolved["selected_skills"]))
        matched_tasks_list.append(list(resolved["matched_tasks"]))
        skill_text_list.append(skill_text)
        skill_all_list.append(bool(resolved["skill_all"]))

        # Construct teacher prompt: prepend skill as a system message
        skill_prefix = f"[Privileged Skill Information]\n{skill_text}\n\n"
        teacher_prompt_text = skill_prefix + prompt_text

        # Tokenize the teacher prompt
        teacher_prompt_ids = tokenizer.encode(teacher_prompt_text, add_special_tokens=False)

        # Truncate if needed (left truncation to keep the end of prompt)
        if len(teacher_prompt_ids) > max_prompt_length:
            teacher_prompt_ids = teacher_prompt_ids[-max_prompt_length:]

        teacher_prompt_ids = torch.tensor(teacher_prompt_ids, dtype=torch.long)
        actual_prompt_len = len(teacher_prompt_ids)

        # Pad to max_prompt_length (left padding)
        pad_length = max_prompt_length - actual_prompt_len
        if pad_length > 0:
            pad_ids = torch.full((pad_length,), tokenizer.pad_token_id, dtype=torch.long)
            teacher_prompt_ids = torch.cat([pad_ids, teacher_prompt_ids])
            t_prompt_mask = torch.cat([
                torch.zeros(pad_length, dtype=torch.long),
                torch.ones(actual_prompt_len, dtype=torch.long),
            ])
        else:
            t_prompt_mask = torch.ones(actual_prompt_len, dtype=torch.long)

        # Combine with response
        response_ids = batch.batch["responses"][i]
        response_mask = original_attention_mask[-response_length:]

        teacher_full_ids = torch.cat([teacher_prompt_ids, response_ids])
        teacher_full_mask = torch.cat([t_prompt_mask, response_mask])
        teacher_position_ids = compute_position_id_with_mask(teacher_full_mask.unsqueeze(0))[0]

        teacher_input_ids_list.append(teacher_full_ids)
        teacher_attention_mask_list.append(teacher_full_mask)
        teacher_position_ids_list.append(teacher_position_ids)

    teacher_input_ids = torch.stack(teacher_input_ids_list)
    teacher_attention_mask = torch.stack(teacher_attention_mask_list)
    teacher_position_ids = torch.stack(teacher_position_ids_list)

    teacher_batch = DataProto.from_dict(
        tensors={
            "input_ids": teacher_input_ids,
            "attention_mask": teacher_attention_mask,
            "position_ids": teacher_position_ids,
            "responses": batch.batch["responses"],
        },
    )
    skill_meta = {
        "selected_skills": np.array(selected_skills_list, dtype=object),
        "matched_tasks": np.array(matched_tasks_list, dtype=object),
        "skill_text": np.array(skill_text_list, dtype=object),
        "skill_all": np.array(skill_all_list, dtype=object),
    }

    return teacher_batch, skill_meta


def _pack_prompt_and_response(tokenizer, prompt_text: str, response_ids, response_mask, max_prompt_length: int):
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    if len(prompt_ids) > max_prompt_length:
        prompt_ids = prompt_ids[-max_prompt_length:]
    prompt_ids = torch.tensor(prompt_ids, dtype=torch.long)
    actual_prompt_len = len(prompt_ids)
    pad_length = max_prompt_length - actual_prompt_len
    if pad_length > 0:
        prompt_ids = torch.cat([
            torch.full((pad_length,), tokenizer.pad_token_id, dtype=torch.long),
            prompt_ids,
        ])
        prompt_mask = torch.cat([
            torch.zeros(pad_length, dtype=torch.long),
            torch.ones(actual_prompt_len, dtype=torch.long),
        ])
    else:
        prompt_mask = torch.ones(actual_prompt_len, dtype=torch.long)
    full_ids = torch.cat([prompt_ids, response_ids])
    full_mask = torch.cat([prompt_mask, response_mask])
    position_ids = compute_position_id_with_mask(full_mask.unsqueeze(0))[0]
    return full_ids, full_mask, position_ids


def _complete_openai(prompts: list[str], *, model, max_tokens: int) -> list[str]:
    import os

    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )
    model_name = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    texts = []
    for prompt in prompts:
        response = client.chat.completions.create(
            model=model_name,
            temperature=0,
            max_tokens=int(max_tokens),
            messages=[{"role": "user", "content": prompt}],
        )
        texts.append(response.choices[0].message.content or "")
    return texts


class RLSDRayTrainer(RayPPOTrainer):
    """
    RLSD trainer that extends RayPPOTrainer with self-distillation
    using privileged skill information as teacher signal.
    """

    def __init__(self, *args, skill_provider: SkillProvider = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.skill_provider = skill_provider
        # RLSD hyperparams from config
        rlsd_cfg = self.config.algorithm.get("rlsd", {})
        self.rlsd_lambda_init = rlsd_cfg.get("rlsd_lambda", 0.5)
        self.rlsd_lambda_warmdown_steps = rlsd_cfg.get("warmdown_steps", 50)
        self.rlsd_clip_eps = rlsd_cfg.get("clip_eps", 0.2)
        self._pending_ana_pg = None
        self._step_skill_metrics = {}
        self._judge_seconds = None
        self._check_seed_vllm_context()

    def _get_rlsd_lambda(self, step: int) -> float:
        """Linearly decay λ from rlsd_lambda_init to 0 over warmdown_steps."""
        if step >= self.rlsd_lambda_warmdown_steps:
            return 0.0
        return self.rlsd_lambda_init * (1.0 - step / self.rlsd_lambda_warmdown_steps)

    def fit(self):
        """
        The training loop of RLSD, extending the standard PPO/GRPO loop
        with teacher forward pass and token-level advantage computation.
        """
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()
        self._maybe_idle_align_val_envs()

        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="RLSD Training")
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "data_source"]
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                if "env_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("env_kwargs")
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    with _timer("gen", timing_raw):
                        gen_batch_output = self.traj_collector.multi_turn_loop(
                            gen_batch=gen_batch,
                            actor_rollout_wg=self.actor_rollout_wg,
                            envs=self.envs,
                            is_train=True,
                        )

                    del batch
                    batch = gen_batch_output

                    batch = adjust_batch(self.config, batch)
                    batch.batch["response_mask"] = compute_response_mask(batch)

                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with _timer("reward", timing_raw):
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    # Compute student log probs (old_log_probs)
                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_loss = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                    # ---- RLSD: Teacher forward pass ----
                    with _timer("teacher_forward", timing_raw):
                        teacher_log_probs = self._compute_teacher_log_probs(batch)
                        batch.batch["teacher_log_probs"] = teacher_log_probs

                    if self.use_reference_policy:
                        with _timer("ref", timing_raw):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        print(f"{list(reward_extra_infos_dict.keys())=}")
                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        if self.config.actor_rollout_ref.actor.get('use_invalid_action_penalty', True):
                            batch, invalid_metrics = apply_invalid_action_penalty(
                                batch,
                                invalid_action_penalty_coef=self.config.actor_rollout_ref.actor.invalid_action_penalty_coef,
                            )
                            metrics.update(invalid_metrics)

                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Compute standard GRPO sequence-level advantages
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                            use_pf_ppo=self.config.algorithm.use_pf_ppo,
                            pf_ppo_reweight_method=self.config.algorithm.pf_ppo.reweight_method,
                            pf_ppo_weight_pow=self.config.algorithm.pf_ppo.weight_pow,
                            step_advantage_w=self.config.algorithm.gigpo.step_advantage_w,
                            gigpo_mode=self.config.algorithm.gigpo.mode,
                            gigpo_enable_similarity=self.config.algorithm.gigpo.enable_similarity,
                            gigpo_similarity_thresh=self.config.algorithm.gigpo.similarity_thresh,
                        )

                        # ---- RLSD: Replace sequence-level advantage with token-level advantage ----
                        seq_advantages = batch.batch["advantages"]
                        student_log_probs = batch.batch["old_log_probs"]
                        teacher_log_probs = batch.batch["teacher_log_probs"]
                        response_mask = batch.batch["response_mask"]

                        current_lambda = self._get_rlsd_lambda(self.global_steps)
                        token_advantages = compute_rlsd_token_advantage(
                            seq_advantages=seq_advantages,
                            student_log_probs=student_log_probs,
                            teacher_log_probs=teacher_log_probs,
                            response_mask=response_mask,
                            rlsd_lambda=current_lambda,
                            rlsd_clip_eps=self.rlsd_clip_eps,
                        )

                        batch.batch["advantages"] = token_advantages

                        # Log RLSD-specific metrics
                        delta_t = (teacher_log_probs - student_log_probs) * response_mask
                        metrics["rlsd/teacher_student_gap_mean"] = masked_mean(delta_t, response_mask).item()
                        metrics["rlsd/teacher_student_gap_std"] = masked_mean(delta_t ** 2, response_mask).sqrt().item()
                        metrics["rlsd/lambda"] = current_lambda
                        metrics["rlsd/clip_eps"] = self.rlsd_clip_eps

                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with _timer("update_actor", timing_raw):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    with _timer("dump_rollout_generations", timing_raw):
                        self._dump_train_trajectories(batch, reward_extra_infos_dict, is_last_step)

                    test_start_step = self.config.trainer.get("test_start_step", 0)
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or (self.global_steps >= test_start_step and self.global_steps % self.config.trainer.test_freq == 0)):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                metrics.update({
                    "training/global_step": self.global_steps,
                    "training/epoch": epoch,
                })
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

    def _teacher_skill_cfg(self):
        """Skill-source settings live on ``opsd_kl`` or ``sdar``, whichever sets skill_source."""
        for key in ("opsd_kl", "sdar"):
            cfg = self.config.algorithm.get(key, {}) or {}
            if cfg.get("skill_source", None) not in (None, ""):
                return cfg
        return {}

    def _check_seed_vllm_context(self) -> None:
        """Match SEED: policy_vllm analysis needs max_model_len >= prompt + completion."""
        if self._skill_source() not in {"seed_ana", "step_skill"}:
            return
        cfg = self._teacher_skill_cfg()
        if str(cfg.get("ana_backend", "policy_vllm")) != "policy_vllm":
            return
        context_length = int(cfg.get("ana_max_prompt_length", 16384))
        completion_tokens = int(cfg.get("ana_max_completion_tokens", 1024))
        required = context_length + completion_tokens
        rollout = self.config.actor_rollout_ref.rollout
        configured = rollout.get("max_model_len", None)
        effective = int(configured or (int(rollout.prompt_length) + int(rollout.response_length)))
        if effective < required:
            raise ValueError(
                "policy_vllm SEED analysis requires actor_rollout_ref.rollout.max_model_len "
                f">= {required} (ana_max_prompt_length + ana_max_completion_tokens), got {effective}."
            )

    def _skill_source(self) -> str:
        source = str(self._teacher_skill_cfg().get("skill_source", "skill_bank"))
        if source not in {"skill_bank", "seed_ana", "step_skill"}:
            raise ValueError(
                "algorithm.opsd_kl.skill_source / algorithm.sdar.skill_source "
                f"must be skill_bank, seed_ana, or step_skill, got {source!r}"
            )
        return source

    def _compute_teacher_log_probs(self, batch: DataProto) -> torch.Tensor:
        """
        Compute teacher log probs by running forward pass with privileged skill info
        prepended to the prompt. Uses the same model π_θ but conditioned on (x, r).

        ``skill_source=skill_bank`` reads the static skill bank.
        ``skill_source=seed_ana`` first asks an analyzer for one SEED episode skill
        per trajectory, then inserts that skill into every step.
        ``skill_source=step_skill`` distills only the steps marked by the analyzer.

        Also attaches skill resolution metadata onto ``batch.non_tensor_batch`` for dumps.
        """
        if self._skill_source() == "seed_ana":
            return self._compute_seed_teacher_log_probs(batch)
        if self._skill_source() == "step_skill":
            return self._compute_step_teacher_log_probs(batch)

        teacher_batch, skill_meta = build_teacher_batch(
            batch=batch,
            skill_provider=self.skill_provider,
            tokenizer=self.tokenizer,
            max_prompt_length=self.config.data.max_prompt_length,
            truncation=self.config.data.get("truncation", "left"),
        )

        # Use the same actor to compute teacher log probs
        teacher_output = self.actor_rollout_wg.compute_log_prob(teacher_batch)
        teacher_log_probs = teacher_output.batch["old_log_probs"]

        for k, v in skill_meta.items():
            batch.non_tensor_batch[k] = v

        return teacher_log_probs

    def _compute_seed_teacher_log_probs(self, batch: DataProto) -> torch.Tensor:
        from verl.trainer.ppo.seed_episode_skill import (
            build_augmented_observation_text,
            build_episode_analysis_prompt,
            extract_task_description,
            parse_analysis_json,
        )

        analyses = self._generate_seed_episode_skills(batch, self._teacher_skill_cfg())
        observations = self._student_observations(batch)
        response_length = batch.batch["responses"].size(1)
        max_prompt_length = int(self.config.data.max_prompt_length)
        chat_kwargs = self.config.data.get("apply_chat_template_kwargs", {}) or {}

        teacher_input_ids_list = []
        teacher_attention_mask_list = []
        teacher_position_ids_list = []
        selected_skills_list = []
        matched_tasks_list = []
        skill_text_list = []
        skill_all_list = []
        summary_list = []
        parse_error_list = []

        traj_uids = batch.non_tensor_batch["traj_uid"]
        for i, traj_uid in enumerate(traj_uids):
            analysis = analyses.get(traj_uid, {})
            skill = str(analysis.get("episode_skill") or "")
            observation = build_augmented_observation_text(
                observation=observations[i],
                episode_skill=skill,
            )
            teacher_prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": observation}],
                add_generation_prompt=True,
                tokenize=False,
                **chat_kwargs,
            )
            packed = _pack_prompt_and_response(
                tokenizer=self.tokenizer,
                prompt_text=teacher_prompt,
                response_ids=batch.batch["responses"][i],
                response_mask=batch.batch["attention_mask"][i, -response_length:],
                max_prompt_length=max_prompt_length,
            )
            teacher_input_ids_list.append(packed[0])
            teacher_attention_mask_list.append(packed[1])
            teacher_position_ids_list.append(packed[2])
            selected_skills_list.append(["episode_skill"] if skill else [])
            matched_tasks_list.append(["seed_ana"] if skill else [])
            skill_text_list.append(skill)
            skill_all_list.append(False)
            summary_list.append(str(analysis.get("episode_summary") or ""))
            parse_error_list.append(analysis.get("parse_error"))

        teacher_batch = DataProto.from_dict(
            tensors={
                "input_ids": torch.stack(teacher_input_ids_list),
                "attention_mask": torch.stack(teacher_attention_mask_list),
                "position_ids": torch.stack(teacher_position_ids_list),
                "responses": batch.batch["responses"],
            },
        )
        teacher_output = self.actor_rollout_wg.compute_log_prob(teacher_batch)
        batch.non_tensor_batch["selected_skills"] = np.array(selected_skills_list, dtype=object)
        batch.non_tensor_batch["matched_tasks"] = np.array(matched_tasks_list, dtype=object)
        batch.non_tensor_batch["skill_text"] = np.array(skill_text_list, dtype=object)
        batch.non_tensor_batch["skill_all"] = np.array(skill_all_list, dtype=object)
        batch.non_tensor_batch["episode_summary"] = np.array(summary_list, dtype=object)
        batch.non_tensor_batch["analysis_parse_error"] = np.array(parse_error_list, dtype=object)
        n_ok = sum(1 for item in analyses.values() if item.get("episode_skill"))
        print(f"[SDAR][seed_ana] trajectories={len(analyses)} parsed_skills={n_ok}")
        return teacher_output.batch["old_log_probs"]

    def _student_observations(self, batch: DataProto) -> list[str]:
        raw_prompts = batch.non_tensor_batch.get("raw_prompt")
        if raw_prompts is not None:
            texts = []
            for item in raw_prompts:
                if isinstance(item, (list, np.ndarray)) and len(item) > 0:
                    content = item[0].get("content", "") if isinstance(item[0], dict) else str(item[0])
                elif isinstance(item, dict):
                    content = item.get("content", "")
                else:
                    content = str(item)
                texts.append(str(content))
            return texts
        response_length = batch.batch["responses"].size(1)
        texts = []
        for i in range(batch.batch["input_ids"].size(0)):
            prompt_ids = batch.batch["input_ids"][i, :-response_length]
            prompt_mask = batch.batch["attention_mask"][i, :-response_length]
            valid = prompt_mask.nonzero(as_tuple=True)[0]
            start = int(valid[0].item()) if len(valid) else 0
            texts.append(self.tokenizer.decode(prompt_ids[start:], skip_special_tokens=False))
        return texts

    def _generate_seed_episode_skills(self, batch: DataProto, sdar_cfg) -> dict:
        from verl.trainer.ppo.seed_episode_skill import (
            build_episode_analysis_prompt,
            extract_task_description,
            parse_analysis_json,
        )

        response_length = batch.batch["responses"].size(1)
        response_mask = batch.batch["attention_mask"][:, -response_length:]
        observations = self._student_observations(batch)
        anchors = batch.non_tensor_batch.get("anchor_obs")
        valids = batch.non_tensor_batch.get("is_action_valid")
        turn_steps = batch.non_tensor_batch.get("turn_step")
        rewards = batch.non_tensor_batch.get("episode_rewards")
        grouped: dict[Any, list[tuple[int, dict]]] = {}
        for i, traj_uid in enumerate(batch.non_tensor_batch["traj_uid"]):
            length = int(response_mask[i].sum().item())
            response = self.tokenizer.decode(batch.batch["responses"][i, :length], skip_special_tokens=True)
            anchor = ""
            if anchors is not None and anchors[i] is not None:
                anchor = anchors[i] if isinstance(anchors[i], str) else str(anchors[i])
            step_index = int(turn_steps[i]) if turn_steps is not None else i
            grouped.setdefault(traj_uid, []).append(
                (
                    step_index,
                    {
                        "step_index": step_index,
                        "observation": anchor or observations[i],
                        "response": response,
                        "action_valid": bool(valids[i]) if valids is not None else True,
                        "student_observation": observations[i],
                    },
                )
            )

        prompts = []
        traj_order = []
        for traj_uid, pairs in grouped.items():
            pairs.sort(key=lambda item: item[0])
            steps = [step for _, step in pairs]
            sample_index = next(
                idx for idx, uid in enumerate(batch.non_tensor_batch["traj_uid"]) if uid == traj_uid
            )
            reward = None
            if rewards is not None:
                reward = float(rewards[sample_index])
            success = None if reward is None else (1.0 if reward > 0 else 0.0)
            task = extract_task_description(steps[0]["student_observation"])
            prompts.append(
                build_episode_analysis_prompt(
                    steps=steps,
                    task_description=task,
                    episode_success=success,
                )
            )
            traj_order.append(traj_uid)

        backend = str(sdar_cfg.get("ana_backend", "policy_vllm"))
        if backend == "openai":
            raw_texts = _complete_openai(
                prompts,
                model=sdar_cfg.get("ana_model"),
                max_tokens=int(sdar_cfg.get("ana_max_completion_tokens", 1024)),
            )
        elif backend == "policy_vllm":
            raw_texts = self._complete_policy_vllm(
                prompts,
                max_prompt_length=int(sdar_cfg.get("ana_max_prompt_length", 16384)),
                max_tokens=int(sdar_cfg.get("ana_max_completion_tokens", 1024)),
            )
        else:
            raise ValueError(f"algorithm.sdar.ana_backend must be policy_vllm or openai, got {backend!r}")

        return {traj_uid: parse_analysis_json(text) for traj_uid, text in zip(traj_order, raw_texts)}

    def _complete_policy_vllm(self, prompts: list[str], *, max_prompt_length: int, max_tokens: int) -> list[str]:
        if not prompts:
            return []
        from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

        chat_kwargs = self.config.data.get("apply_chat_template_kwargs", {}) or {}
        input_ids_list = []
        attention_mask_list = []
        position_ids_list = []
        raw_prompt_ids = []
        for prompt in prompts:
            rendered = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
                **chat_kwargs,
            )
            token_ids = self.tokenizer.encode(rendered, add_special_tokens=False)
            if len(token_ids) > max_prompt_length:
                print(
                    f"[SDAR][seed_ana] analysis prompt {len(token_ids)} tokens, "
                    f"left-truncated to {max_prompt_length}"
                )
                token_ids = token_ids[-max_prompt_length:]
            raw_prompt_ids.append(token_ids)
            ids = torch.tensor(token_ids, dtype=torch.long)
            pad_length = max_prompt_length - len(token_ids)
            if pad_length > 0:
                ids = torch.cat([torch.full((pad_length,), self.tokenizer.pad_token_id, dtype=torch.long), ids])
                mask = torch.cat([
                    torch.zeros(pad_length, dtype=torch.long),
                    torch.ones(len(token_ids), dtype=torch.long),
                ])
            else:
                mask = torch.ones(len(token_ids), dtype=torch.long)
            input_ids_list.append(ids)
            attention_mask_list.append(mask)
            position_ids_list.append(compute_position_id_with_mask(mask.unsqueeze(0))[0])

        prompt_batch = DataProto.from_dict(
            tensors={
                "input_ids": torch.stack(input_ids_list),
                "attention_mask": torch.stack(attention_mask_list),
                "position_ids": torch.stack(position_ids_list),
            },
            non_tensors={"raw_prompt_ids": np.array(raw_prompt_ids, dtype=object)},
            meta_info={
                "do_sample": False,
                "validate": False,
                "sampling_kwargs": {"max_tokens": int(max_tokens), "n": 1},
            },
        )
        padded, pad_size = pad_dataproto_to_divisor(prompt_batch, self.actor_rollout_wg.world_size)
        output = self.actor_rollout_wg.generate_sequences(padded)
        output = unpad_dataproto(output, pad_size=pad_size)
        texts = []
        response_mask = output.batch["attention_mask"][:, -output.batch["responses"].size(1) :]
        for i in range(len(prompts)):
            length = int(response_mask[i].sum().item())
            texts.append(self.tokenizer.decode(output.batch["responses"][i, :length], skip_special_tokens=True))
        return texts

    def _generate_analysis_batch(
        self,
        prompts: list[str],
        *,
        max_prompt_length: int,
        max_tokens: int,
    ) -> tuple[DataProto | None, list[str]]:
        """Sample analyzer completions with the student vLLM and keep their tokens."""
        if not prompts:
            return None, []
        from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
        from verl.trainer.ppo.step_skill import ANALYZER_SYSTEM

        chat_kwargs = self.config.data.get("apply_chat_template_kwargs", {}) or {}
        temperature = float(self.config.actor_rollout_ref.rollout.temperature)
        input_ids_list = []
        attention_mask_list = []
        position_ids_list = []
        raw_prompt_ids = []
        for prompt in prompts:
            rendered = self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": ANALYZER_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                add_generation_prompt=True,
                tokenize=False,
                **chat_kwargs,
            )
            token_ids = self.tokenizer.encode(rendered, add_special_tokens=False)
            if len(token_ids) > max_prompt_length:
                print(
                    f"[SDAR][step_skill] analysis prompt {len(token_ids)} tokens, "
                    f"left-truncated to {max_prompt_length}"
                )
                token_ids = token_ids[-max_prompt_length:]
            raw_prompt_ids.append(token_ids)
            ids = torch.tensor(token_ids, dtype=torch.long)
            pad_length = max_prompt_length - len(token_ids)
            if pad_length > 0:
                ids = torch.cat([torch.full((pad_length,), self.tokenizer.pad_token_id, dtype=torch.long), ids])
                mask = torch.cat([
                    torch.zeros(pad_length, dtype=torch.long),
                    torch.ones(len(token_ids), dtype=torch.long),
                ])
            else:
                mask = torch.ones(len(token_ids), dtype=torch.long)
            input_ids_list.append(ids)
            attention_mask_list.append(mask)
            position_ids_list.append(compute_position_id_with_mask(mask.unsqueeze(0))[0])

        prompt_batch = DataProto.from_dict(
            tensors={
                "input_ids": torch.stack(input_ids_list),
                "attention_mask": torch.stack(attention_mask_list),
                "position_ids": torch.stack(position_ids_list),
            },
            non_tensors={"raw_prompt_ids": np.array(raw_prompt_ids, dtype=object)},
            meta_info={
                "do_sample": temperature > 0,
                "validate": False,
                "sampling_kwargs": {
                    "max_tokens": int(max_tokens),
                    "n": 1,
                    "temperature": temperature,
                    "top_p": 1.0,
                    "top_k": -1,
                },
            },
        )
        padded, pad_size = pad_dataproto_to_divisor(prompt_batch, self.actor_rollout_wg.world_size)
        output = self.actor_rollout_wg.generate_sequences(padded)
        output = unpad_dataproto(output, pad_size=pad_size)
        texts = []
        response_mask = output.batch["attention_mask"][:, -output.batch["responses"].size(1) :]
        for i in range(len(prompts)):
            length = int(response_mask[i].sum().item())
            texts.append(self.tokenizer.decode(output.batch["responses"][i, :length], skip_special_tokens=True))
        return output, texts

    def _compute_step_teacher_log_probs(self, batch: DataProto) -> torch.Tensor:
        from collections import defaultdict

        from verl.trainer.ppo.step_skill import (
            build_failure_analysis_prompt,
            build_failure_judge_prompt,
            build_success_analysis_prompt,
            build_success_judge_prompt,
            category_advantages,
            complete_openai_compatible,
            extract_actions,
            MAX_SUCCESS_REFERENCES,
            parse_judge_score,
            parse_step_skill_analysis,
        )

        cfg = self._teacher_skill_cfg()
        if str(cfg.get("ana_backend", "policy_vllm")) != "policy_vllm":
            raise ValueError("step_skill ana_backend must be policy_vllm so the analysis tokens stay on-policy")

        self._judge_seconds = None
        episodes = self._collect_step_episodes(batch)
        success_eps = [episode for episode in episodes if episode["success"]]
        failure_eps = [episode for episode in episodes if not episode["success"]]
        max_prompt_length = int(cfg.get("ana_max_prompt_length", 16384))
        max_tokens = int(cfg.get("ana_max_completion_tokens", 1024))

        loss_mode = str(cfg.get("ana_loss_mode", "pg")).lower()
        if loss_mode not in {"pg", "sft", "none"}:
            raise ValueError(f"step_skill ana_loss_mode must be pg, sft, or none, got {loss_mode}")
        temperature = float(self.config.actor_rollout_ref.rollout.temperature)
        success_prompts = [
            build_success_analysis_prompt(steps=episode["steps"], task_description=episode["task"])
            for episode in success_eps
        ]
        api_temperature = float(cfg.get("ana_api_temperature", 0.6))
        if loss_mode == "sft":
            success_output = None
            success_texts = self._complete_analysis_api(success_prompts, cfg, max_tokens=max_tokens)
        else:
            success_output, success_texts = self._generate_analysis_batch(
                success_prompts,
                max_prompt_length=max_prompt_length,
                max_tokens=max_tokens,
            )
        success_parsed = [
            parse_step_skill_analysis(text, {int(step["step_index"]) for step in episode["steps"]})
            for text, episode in zip(success_texts, success_eps)
        ]
        references = defaultdict(list)
        for episode in success_eps:
            bucket = references[episode["uid"]]
            if len(bucket) >= MAX_SUCCESS_REFERENCES:
                continue
            bucket.append(extract_actions(episode["steps"]))

        failure_prompts = [
            build_failure_analysis_prompt(
                steps=episode["steps"],
                task_description=episode["task"],
                reference_lists=references.get(episode["uid"], []),
            )
            for episode in failure_eps
        ]
        if loss_mode == "sft":
            failure_output = None
            failure_texts = self._complete_analysis_api(failure_prompts, cfg, max_tokens=max_tokens)
        else:
            failure_output, failure_texts = self._generate_analysis_batch(
                failure_prompts,
                max_prompt_length=max_prompt_length,
                max_tokens=max_tokens,
            )
        failure_parsed = [
            parse_step_skill_analysis(text, {int(step["step_index"]) for step in episode["steps"]})
            for text, episode in zip(failure_texts, failure_eps)
        ]

        judge_prompts = []
        judge_raw = []
        scores = []
        judge_reasons = []
        judge_errors = []
        advantages = []
        if loss_mode == "pg":
            judge_meta = []
            for episode, text in zip(success_eps, success_texts):
                judge_prompts.append(
                    build_success_judge_prompt(
                        steps=episode["steps"],
                        task_description=episode["task"],
                        analysis_text=text,
                    )
                )
                judge_meta.append(("success", episode["uid"]))
            for episode, text in zip(failure_eps, failure_texts):
                judge_prompts.append(
                    build_failure_judge_prompt(
                        steps=episode["steps"],
                        task_description=episode["task"],
                        reference_lists=references.get(episode["uid"], []),
                        analysis_text=text,
                    )
                )
                judge_meta.append(("failure", episode["uid"]))
            judge_temperature = float(cfg.get("judge_temperature", 0.6))
            judge_started = time.perf_counter()
            judge_raw = complete_openai_compatible(
                judge_prompts,
                model=cfg.get("judge_model"),
                base_url=cfg.get("judge_base_url"),
                api_key=cfg.get("judge_api_key"),
                max_tokens=int(cfg.get("judge_max_tokens", 256)),
                temperature=judge_temperature,
                top_p=float(cfg.get("judge_top_p", 0.95)),
            )
            self._judge_seconds = time.perf_counter() - judge_started
            judge_reasons = []
            for text in judge_raw:
                score, reason, error = parse_judge_score(text)
                scores.append(score)
                judge_reasons.append(reason)
                judge_errors.append(error)
            advantages = category_advantages(
                [label for label, _uid in judge_meta],
                [uid for _label, uid in judge_meta],
                scores,
            ) if judge_meta else []

        analyses = {}
        records = {}
        ordered = (
            list(zip(success_eps, success_prompts, success_texts, success_parsed, ["success"] * len(success_eps)))
            + list(zip(failure_eps, failure_prompts, failure_texts, failure_parsed, ["failure"] * len(failure_eps)))
        )
        for index, (episode, prompt, text, parsed, kind) in enumerate(ordered):
            if kind == "success":
                analyses[episode["traj_uid"]] = {"kind": "success", **parsed}
            else:
                analyses[episode["traj_uid"]] = {
                    "kind": "failure",
                    "reference_lists": references.get(episode["uid"], []),
                    **parsed,
                }
            record = {
                "loss_mode": loss_mode,
                "kind": kind,
                "analysis_source": "api" if loss_mode == "sft" else "policy_vllm",
                "temperature": api_temperature if loss_mode == "sft" else temperature,
                "max_tokens": max_tokens,
                "analysis_prompt": prompt,
                "analysis_output": text,
                "parse_error": parsed.get("parse_error"),
                "step_skills": parsed.get("step_skills") or [],
                "trajectory_summary": parsed.get("trajectory_summary") or "",
                "reference_actions": references.get(episode["uid"], []) if kind == "failure" else [],
            }
            if loss_mode == "pg":
                record.update(
                    {
                        "judge_prompt": judge_prompts[index] if index < len(judge_prompts) else "",
                        "judge_output": judge_raw[index] if index < len(judge_raw) else "",
                        "judge_reason": judge_reasons[index] if index < len(judge_reasons) else "",
                        "judge_temperature": float(cfg.get("judge_temperature", 0.6)),
                        "judge_score": scores[index] if index < len(scores) else None,
                        "advantage": advantages[index] if index < len(advantages) else None,
                    }
                )
            elif loss_mode == "sft":
                record["sft_target"] = text
                record["api_model"] = cfg.get("ana_api_model") or cfg.get("judge_model")
                record["api_base_url"] = cfg.get("ana_api_base_url") or cfg.get("judge_base_url")
            records[episode["traj_uid"]] = record
        if loss_mode == "none":
            self._pending_ana_pg = None
        elif loss_mode == "sft":
            self._stash_ana_sft_batch(
                prompts=success_prompts + failure_prompts,
                texts=success_texts + failure_texts,
                max_prompt_length=max_prompt_length,
                max_tokens=max_tokens,
                cfg=cfg,
            )
        else:
            self._stash_ana_pg_batch(
                success_output=success_output,
                failure_output=failure_output,
                advantages=advantages,
                n_success=len(success_eps),
                cfg=cfg,
            )
        teacher_log_probs = self._teacher_log_probs_for_marked_steps(batch, analyses)
        self._attach_step_skill_meta(batch, analyses, records)
        self._step_skill_metrics = self._step_skill_log_metrics(
            loss_mode=loss_mode,
            success_parsed=success_parsed,
            failure_parsed=failure_parsed,
            scores=scores,
            advantages=advantages,
            judge_errors=judge_errors,
        )
        n_success_opsd = sum(1 for item in success_parsed if item["step_skills"])
        n_failure_opsd = sum(1 for item in failure_parsed if item["step_skills"])
        score_text = f"judge_score_mean={sum(scores) / len(scores):.3f}" if scores else "judge_score_mean=n/a"
        judge_seconds = getattr(self, "_judge_seconds", None)
        judge_time_text = f" judge_time={judge_seconds:.1f}s" if judge_seconds is not None else ""
        print(
            f"[SDAR][step_skill] mode={loss_mode} trajectories={len(episodes)} success={len(success_eps)} "
            f"success_opsd={n_success_opsd} failure={len(failure_eps)} failure_opsd={n_failure_opsd} "
            f"{score_text} judge_errors={sum(error is not None for error in judge_errors)}{judge_time_text}"
        )
        return teacher_log_probs

    def _complete_analysis_api(self, prompts, cfg, *, max_tokens: int) -> list[str]:
        from verl.trainer.ppo.step_skill import ANALYZER_SYSTEM, complete_openai_compatible

        if not prompts:
            return []

        return complete_openai_compatible(
            prompts,
            model=cfg.get("ana_api_model") or cfg.get("judge_model"),
            base_url=cfg.get("ana_api_base_url") or cfg.get("judge_base_url"),
            api_key=cfg.get("ana_api_key") or cfg.get("judge_api_key"),
            max_tokens=max_tokens,
            temperature=float(cfg.get("ana_api_temperature", 0.6)),
            top_p=float(cfg.get("ana_api_top_p", 0.95)),
            system=ANALYZER_SYSTEM,
        )

    def _step_skill_log_metrics(self, *, loss_mode, success_parsed, failure_parsed, scores, advantages, judge_errors) -> dict:
        parsed = list(success_parsed) + list(failure_parsed)
        metrics = {}
        if parsed:
            metrics["step_skill/parse_error_rate"] = sum(item.get("parse_error") is not None for item in parsed) / len(parsed)
        if loss_mode != "pg":
            return metrics
        if scores:
            metrics["step_skill/judge_score_mean"] = float(sum(scores) / len(scores))
            success_scores = scores[: len(success_parsed)]
            failure_scores = scores[len(success_parsed) :]
            if success_scores:
                metrics["step_skill/judge_score_success"] = float(sum(success_scores) / len(success_scores))
            if failure_scores:
                metrics["step_skill/judge_score_failure"] = float(sum(failure_scores) / len(failure_scores))
        if advantages:
            metrics["step_skill/advantage_mean"] = float(sum(advantages) / len(advantages))
        if judge_errors:
            metrics["step_skill/judge_error_rate"] = sum(error is not None for error in judge_errors) / len(judge_errors)
        return metrics

    def _collect_step_episodes(self, batch: DataProto) -> list[dict]:
        from verl.trainer.ppo.seed_episode_skill import extract_task_description

        response_length = batch.batch["responses"].size(1)
        response_mask = batch.batch["attention_mask"][:, -response_length:]
        observations = self._student_observations(batch)
        anchors = batch.non_tensor_batch.get("anchor_obs")
        valids = batch.non_tensor_batch.get("is_action_valid")
        turn_steps = batch.non_tensor_batch.get("turn_step")
        rewards = batch.non_tensor_batch.get("episode_rewards")
        uids = batch.non_tensor_batch.get("uid")
        grouped: dict[Any, list[tuple[int, int, dict]]] = {}
        for i, traj_uid in enumerate(batch.non_tensor_batch["traj_uid"]):
            length = int(response_mask[i].sum().item())
            response = self.tokenizer.decode(batch.batch["responses"][i, :length], skip_special_tokens=True)
            anchor = ""
            if anchors is not None and anchors[i] is not None:
                anchor = anchors[i] if isinstance(anchors[i], str) else str(anchors[i])
            step_index = int(turn_steps[i]) if turn_steps is not None else i
            grouped.setdefault(traj_uid, []).append(
                (
                    step_index,
                    i,
                    {
                        "step_index": step_index,
                        "observation": anchor or observations[i],
                        "response": response,
                        "action_valid": bool(valids[i]) if valids is not None else True,
                        "student_observation": observations[i],
                    },
                )
            )
        episodes = []
        for traj_uid, triples in grouped.items():
            triples.sort(key=lambda item: item[0])
            sample_index = triples[0][1]
            reward = float(rewards[sample_index]) if rewards is not None else 0.0
            uid = uids[sample_index] if uids is not None else traj_uid
            steps = [step for _step, _row, step in triples]
            episodes.append(
                {
                    "traj_uid": traj_uid,
                    "uid": uid,
                    "success": reward > 0,
                    "task": extract_task_description(steps[0]["student_observation"]),
                    "steps": steps,
                    "row_indices": [row for _step, row, _step_dict in triples],
                }
            )
        return episodes

    def _stash_ana_pg_batch(
        self,
        *,
        success_output: DataProto | None,
        failure_output: DataProto | None,
        advantages: list[float],
        n_success: int,
        cfg,
    ) -> None:
        pieces = [piece for piece in (success_output, failure_output) if piece is not None and len(piece) > 0]
        coef = float(cfg.get("ana_pg_coef", 0.01))
        if not pieces or coef == 0 or not advantages:
            self._pending_ana_pg = None
            return
        ana_batch = DataProto.concat(pieces) if len(pieces) > 1 else pieces[0]
        if len(ana_batch) != len(advantages):
            raise RuntimeError(
                f"step_skill analysis batch {len(ana_batch)} does not match advantages {len(advantages)}"
            )
        response_length = ana_batch.batch["responses"].size(1)
        response_mask = ana_batch.batch["attention_mask"][:, -response_length:]
        advantage_rows = torch.tensor(advantages, dtype=torch.float32).unsqueeze(-1) * response_mask.to(torch.float32)
        is_success = torch.zeros(len(ana_batch), response_length, dtype=torch.float32)
        is_success[:n_success] = response_mask[:n_success].to(torch.float32)
        logprob_batch = DataProto.from_dict(
            tensors={
                "responses": ana_batch.batch["responses"],
                "input_ids": ana_batch.batch["input_ids"],
                "attention_mask": ana_batch.batch["attention_mask"],
                "position_ids": ana_batch.batch["position_ids"],
            },
            meta_info={"ana_log_prob": True, "ana_micro_batch_size": 1},
        )
        from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

        padded, pad_size = pad_dataproto_to_divisor(logprob_batch, self.actor_rollout_wg.world_size)
        old_output = self.actor_rollout_wg.compute_log_prob(padded)
        old_output = unpad_dataproto(old_output, pad_size=pad_size)
        pg_batch = DataProto.from_dict(
            tensors={
                "responses": ana_batch.batch["responses"],
                "input_ids": ana_batch.batch["input_ids"],
                "attention_mask": ana_batch.batch["attention_mask"],
                "position_ids": ana_batch.batch["position_ids"],
                "old_log_probs": old_output.batch["old_log_probs"],
                "advantages": advantage_rows,
                "ana_is_success": is_success,
            },
            meta_info={
                "temperature": float(self.config.actor_rollout_ref.rollout.temperature),
                "ana_loss_mode": "pg",
                "ana_pg_coef": coef,
                "ana_alpha_pos": float(cfg.get("ana_alpha_pos", 1.0)),
                "ana_alpha_neg": float(cfg.get("ana_alpha_neg", 1.0)),
                "ana_micro_batch_size": int(cfg.get("ana_micro_batch_size", 1)),
                "ana_real_rows": len(ana_batch),
                "global_token_num": ana_batch.batch["attention_mask"].sum(dim=-1).tolist(),
            },
        )
        self._pending_ana_pg = self._pad_inactive_ana_rows(pg_batch, self.actor_rollout_wg.world_size)

    def _stash_ana_sft_batch(self, *, prompts, texts, max_prompt_length, max_tokens, cfg) -> None:
        coef = float(cfg.get("ana_pg_coef", 0.01))
        if not prompts or coef == 0:
            self._pending_ana_pg = None
            return
        ana_batch = self._pack_api_analysis_batch(
            prompts,
            texts,
            max_prompt_length=max_prompt_length,
            max_response_length=max_tokens,
        )
        pg_batch = DataProto.from_dict(
            tensors={
                "responses": ana_batch.batch["responses"],
                "input_ids": ana_batch.batch["input_ids"],
                "attention_mask": ana_batch.batch["attention_mask"],
                "position_ids": ana_batch.batch["position_ids"],
            },
            meta_info={
                "temperature": float(self.config.actor_rollout_ref.rollout.temperature),
                "ana_loss_mode": "sft",
                "ana_pg_coef": coef,
                "ana_micro_batch_size": int(cfg.get("ana_micro_batch_size", 1)),
                "ana_real_rows": len(ana_batch),
                "global_token_num": ana_batch.batch["attention_mask"].sum(dim=-1).tolist(),
            },
        )
        self._pending_ana_pg = self._pad_inactive_ana_rows(pg_batch, self.actor_rollout_wg.world_size)

    def _pack_api_analysis_batch(self, prompts, texts, *, max_prompt_length, max_response_length) -> DataProto:
        chat_kwargs = self.config.data.get("apply_chat_template_kwargs", {}) or {}
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else eos_id
        input_ids_list = []
        attention_list = []
        position_list = []
        response_list = []
        for prompt, text in zip(prompts, texts):
            rendered = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
                **chat_kwargs,
            )
            prompt_ids = self.tokenizer.encode(rendered, add_special_tokens=False)
            if len(prompt_ids) > max_prompt_length:
                prompt_ids = prompt_ids[-max_prompt_length:]
            response_ids = self.tokenizer.encode(str(text or ""), add_special_tokens=False)
            if len(response_ids) < max_response_length and eos_id is not None and (not response_ids or response_ids[-1] != eos_id):
                response_ids.append(eos_id)
            response_ids = response_ids[:max_response_length]
            response_mask = [1] * len(response_ids) + [0] * (max_response_length - len(response_ids))
            response_ids = response_ids + [pad_id] * (max_response_length - len(response_ids))
            prompt_pad = max_prompt_length - len(prompt_ids)
            full_ids = [pad_id] * prompt_pad + prompt_ids + response_ids
            full_mask = [0] * prompt_pad + [1] * len(prompt_ids) + response_mask
            ids = torch.tensor(full_ids, dtype=torch.long)
            mask = torch.tensor(full_mask, dtype=torch.long)
            input_ids_list.append(ids)
            attention_list.append(mask)
            position_list.append(compute_position_id_with_mask(mask.unsqueeze(0))[0])
            response_list.append(torch.tensor(response_ids, dtype=torch.long))
        return DataProto.from_dict(
            tensors={
                "input_ids": torch.stack(input_ids_list),
                "attention_mask": torch.stack(attention_list),
                "position_ids": torch.stack(position_list),
                "responses": torch.stack(response_list),
            }
        )

    def _pad_inactive_ana_rows(self, batch: DataProto, divisor: int) -> DataProto:
        remainder = len(batch) % divisor
        if remainder == 0:
            return batch
        pad = divisor - remainder
        extra = DataProto.concat([batch[:1] for _ in range(pad)])
        cloned = {key: extra.batch[key].clone() for key in extra.batch.keys()}
        response_length = cloned["responses"].size(1)
        cloned["attention_mask"][:, -response_length:] = 0
        for key in ("advantages", "ana_is_success", "old_log_probs"):
            if key in cloned:
                cloned[key].zero_()
        extra = DataProto.from_dict(tensors=cloned, meta_info=dict(batch.meta_info))
        padded = DataProto.concat([batch, extra])
        padded.meta_info["global_token_num"] = padded.batch["attention_mask"].sum(dim=-1).tolist()
        return padded

    def _teacher_log_probs_for_marked_steps(self, batch: DataProto, analyses: dict) -> torch.Tensor:
        from verl.trainer.ppo.step_skill import insert_privilege_text

        response_length = batch.batch["responses"].size(1)
        teacher_log_probs = torch.zeros(batch.batch["responses"].shape, dtype=torch.float32)
        observations = self._student_observations(batch)
        turn_steps = batch.non_tensor_batch.get("turn_step")
        max_prompt_length = int(self.config.data.max_prompt_length)
        chat_kwargs = self.config.data.get("apply_chat_template_kwargs", {}) or {}
        selected_rows = []
        input_ids_list = []
        attention_mask_list = []
        position_ids_list = []
        response_list = []
        response_mask_list = []
        for i, traj_uid in enumerate(batch.non_tensor_batch["traj_uid"]):
            analysis = analyses.get(traj_uid)
            step_index = int(turn_steps[i]) if turn_steps is not None else i
            if analysis is None or not self._step_uses_opsd(analysis, step_index):
                continue
            privilege = self._privilege_text(analysis, step_index)
            observation = insert_privilege_text(observation=observations[i], privilege=privilege)
            teacher_prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": observation}],
                add_generation_prompt=True,
                tokenize=False,
                **chat_kwargs,
            )
            packed = _pack_prompt_and_response(
                tokenizer=self.tokenizer,
                prompt_text=teacher_prompt,
                response_ids=batch.batch["responses"][i],
                response_mask=batch.batch["attention_mask"][i, -response_length:],
                max_prompt_length=max_prompt_length,
            )
            selected_rows.append(i)
            input_ids_list.append(packed[0])
            attention_mask_list.append(packed[1])
            position_ids_list.append(packed[2])
            response_list.append(batch.batch["responses"][i])
            response_mask_list.append(batch.batch["attention_mask"][i, -response_length:])
        opsd_mask = torch.zeros(batch.batch["responses"].shape, dtype=torch.float32)
        for row in selected_rows:
            opsd_mask[row] = 1
        batch.batch["opsd_token_mask"] = opsd_mask
        if not selected_rows:
            return teacher_log_probs
        teacher_batch = DataProto.from_dict(
            tensors={
                "input_ids": torch.stack(input_ids_list),
                "attention_mask": torch.stack(attention_mask_list),
                "position_ids": torch.stack(position_ids_list),
                "responses": torch.stack(response_list),
            },
        )
        from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

        padded, pad_size = pad_dataproto_to_divisor(teacher_batch, self.actor_rollout_wg.world_size)
        teacher_output = self.actor_rollout_wg.compute_log_prob(padded)
        teacher_output = unpad_dataproto(teacher_output, pad_size=pad_size)
        for local_index, row in enumerate(selected_rows):
            teacher_log_probs[row] = teacher_output.batch["old_log_probs"][local_index]
        return teacher_log_probs

    def _step_uses_opsd(self, analysis: dict, step_index: int) -> bool:
        from verl.trainer.ppo.step_skill import skill_for_step

        return bool(skill_for_step(analysis, step_index))

    def _privilege_text(self, analysis: dict, step_index: int) -> str:
        from verl.trainer.ppo.step_skill import privilege_for_step

        return privilege_for_step(analysis, step_index)

    def _attach_step_skill_meta(self, batch: DataProto, analyses: dict, records: dict | None = None) -> None:
        from verl.trainer.ppo.step_skill import skill_for_step

        skill_text = []
        matched = []
        errors = []
        turn_steps = batch.non_tensor_batch.get("turn_step")
        for i, traj_uid in enumerate(batch.non_tensor_batch["traj_uid"]):
            analysis = analyses.get(traj_uid, {})
            step_index = int(turn_steps[i]) if turn_steps is not None else i
            skill = skill_for_step(analysis, step_index)
            groups = [
                list(item.get("target_steps") or [])
                for item in analysis.get("step_skills") or []
                if step_index in set(item.get("target_steps") or [])
            ]
            skill_text.append(skill)
            matched.append(groups[0] if groups else [])
            errors.append(analysis.get("parse_error"))
        batch.non_tensor_batch["skill_text"] = np.array(skill_text, dtype=object)
        batch.non_tensor_batch["matched_tasks"] = np.array(matched, dtype=object)
        batch.non_tensor_batch["analysis_parse_error"] = np.array(errors, dtype=object)
        batch.non_tensor_batch["selected_skills"] = np.array(
            [["step_skill"] if text else [] for text in skill_text],
            dtype=object,
        )
        if records is not None:
            batch.non_tensor_batch["step_skill_record"] = np.array(
                [records.get(traj_uid) for traj_uid in batch.non_tensor_batch["traj_uid"]],
                dtype=object,
            )

    def _maybe_update_ana_policy(self) -> dict:
        metrics = dict(self._step_skill_metrics or {})
        self._step_skill_metrics = {}
        pending = self._pending_ana_pg
        self._pending_ana_pg = None
        if pending is None:
            return metrics
        pending.meta_info["accumulate_into_existing_grad"] = True
        pending.meta_info["finalize_optimizer_step"] = True
        output = self.actor_rollout_wg.update_ana_actor(pending)
        ana_metrics = reduce_metrics(output.meta_info["metrics"])
        metrics.update(
            {
                key: value
                for key, value in ana_metrics.items()
                if key.startswith(("ana_pg/", "ana_sft/")) or key == "actor/grad_norm"
            }
        )
        return metrics
