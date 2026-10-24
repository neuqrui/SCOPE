"""Dedicated fixed-coefficient K3-IS OPD trainer."""

from pprint import pprint

import numpy as np
import torch
from tqdm import tqdm

from agent_system.multi_turn_rollout import adjust_batch
from verl import DataProto
from verl.trainer.ppo.metric_utils import compute_throughout_metrics, compute_timing_metrics
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, _timer, compute_response_mask
from verl.trainer.ppo.rlsd_ray_trainer import build_teacher_batch
from verl.utils.metric import reduce_metrics


def compute_opd_data_metrics(batch: DataProto) -> dict[str, float]:
    """Return rollout statistics that do not depend on rewards or advantages."""

    max_response_length = batch.batch["responses"].shape[-1]
    response_mask = batch.batch["response_mask"].bool()
    prompt_mask = batch.batch["attention_mask"][:, :-max_response_length].bool()
    response_length = response_mask.sum(-1).float()
    prompt_length = prompt_mask.sum(-1).float()

    metrics = {
        "response_length/mean": float(response_length.mean().item()),
        "response_length/max": float(response_length.max().item()),
        "response_length/min": float(response_length.min().item()),
        "response_length/clip_ratio": float((response_length == max_response_length).float().mean().item()),
        "prompt_length/mean": float(prompt_length.mean().item()),
        "prompt_length/max": float(prompt_length.max().item()),
        "prompt_length/min": float(prompt_length.min().item()),
    }

    traj_uids = batch.non_tensor_batch.get("traj_uid")
    if traj_uids is None:
        return metrics
    _, unique_idx = np.unique(np.asarray(traj_uids), return_index=True)
    for metric_name, batch_key in (
        ("episode/reward", "episode_rewards"),
        ("episode/length", "episode_lengths"),
        ("episode/tool_call_count", "tool_callings"),
    ):
        if batch_key not in batch.non_tensor_batch:
            continue
        values = np.asarray(batch.non_tensor_batch[batch_key], dtype=float)[unique_idx]
        metrics[f"{metric_name}/mean"] = float(values.mean())
        metrics[f"{metric_name}/max"] = float(values.max())
        metrics[f"{metric_name}/min"] = float(values.min())
    for key, values in batch.non_tensor_batch.items():
        if "success_rate" in key:
            metrics[f"episode/{key}"] = float(np.asarray(values, dtype=float)[unique_idx].mean())
    return metrics


class OPDTrainer(RayPPOTrainer):
    """Train only fixed-coefficient sampled K3-IS policy distillation."""

    def __init__(self, *args, skill_provider=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.skill_provider = skill_provider
        self.opd_coef = float(self.config.algorithm.opd.opd_coef)
        if self.opd_coef < 0:
            raise ValueError("algorithm.opd.opd_coef must be nonnegative")

    def _compute_teacher_log_probs(self, batch: DataProto) -> torch.Tensor:
        teacher_batch, skill_meta = build_teacher_batch(
            batch=batch,
            skill_provider=self.skill_provider,
            tokenizer=self.tokenizer,
            max_prompt_length=self.config.data.max_prompt_length,
            truncation=self.config.data.get("truncation", "left"),
        )
        if not hasattr(self, "teacher_policy_wg"):
            raise RuntimeError("OPDTrainer requires a framework-managed teacher policy worker")
        teacher_output = self.teacher_policy_wg.compute_ref_log_prob(teacher_batch)
        for k, v in skill_meta.items():
            batch.non_tensor_batch[k] = v
        return teacher_output.batch["ref_log_prob"]

    def fit(self):
        """Run rollout, frozen-teacher scoring, K3-IS updates, and validation."""

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

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="OPD training")
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                batch = DataProto.from_single_dict(batch_dict)

                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "data_source"]
                for key in ("multi_modal_data", "raw_prompt", "tools_kwargs", "env_kwargs"):
                    if key in batch.non_tensor_batch:
                        non_tensor_batch_keys_to_pop.append(key)
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )
                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    with _timer("gen", timing_raw):
                        batch = self.traj_collector.multi_turn_loop(
                            gen_batch=gen_batch,
                            actor_rollout_wg=self.actor_rollout_wg,
                            envs=self.envs,
                            is_train=True,
                        )
                    batch = adjust_batch(self.config, batch)
                    batch.batch["response_mask"] = compute_response_mask(batch)
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                    with _timer("teacher_forward", timing_raw):
                        batch.batch["teacher_log_probs"] = self._compute_teacher_log_probs(batch)

                    with _timer("update_actor", timing_raw):
                        batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                        actor_output = self.actor_rollout_wg.update_opd_actor(batch)
                    metrics.update(reduce_metrics(actor_output.meta_info["metrics"]))

                    test_start_step = self.config.trainer.get("test_start_step", 0)
                    should_test = self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or (self.global_steps >= test_start_step and self.global_steps % self.config.trainer.test_freq == 0))
                    if should_test:
                        with _timer("testing", timing_raw):
                            val_metrics = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                metrics.update({"training/global_step": self.global_steps, "training/epoch": epoch, "opd/coef": self.opd_coef})
                metrics.update(compute_opd_data_metrics(batch))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                metrics.update(
                    compute_throughout_metrics(
                        batch=batch,
                        timing_raw=timing_raw,
                        n_gpus=self.resource_pool_manager.get_n_gpus(),
                    )
                )
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return
