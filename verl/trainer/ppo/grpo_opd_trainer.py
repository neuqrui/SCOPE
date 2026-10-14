"""GRPO training with a fixed-coefficient OPD auxiliary objective."""

import math

import torch

from verl import DataProto
from verl.trainer.ppo.rlsd_ray_trainer import build_teacher_batch
from verl.trainer.ppo.skillsd_ray_trainer import SkillSDRayTrainer


class GRPOOPDTrainer(SkillSDRayTrainer):
    """Keep standard GRPO advantages and add OPD from a frozen teacher."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.opd_coef = float(self.config.algorithm.opd.opd_coef)
        if not math.isfinite(self.opd_coef) or self.opd_coef < 0:
            raise ValueError("algorithm.opd.opd_coef must be finite and nonnegative")

    def _get_sdl_lambda(self, step: int) -> float:
        """Return the fixed OPD coefficient used by every training step."""

        return self.opd_coef

    def _compute_teacher_log_probs(self, batch: DataProto) -> torch.Tensor:
        teacher_batch, skill_meta = build_teacher_batch(
            batch=batch,
            skill_provider=self.skill_provider,
            tokenizer=self.tokenizer,
            max_prompt_length=self.config.data.max_prompt_length,
            truncation=self.config.data.get("truncation", "left"),
        )
        if not hasattr(self, "teacher_policy_wg"):
            raise RuntimeError("GRPOOPDTrainer requires a framework-managed teacher policy worker")
        teacher_output = self.teacher_policy_wg.compute_ref_log_prob(teacher_batch)
        for k, v in skill_meta.items():
            batch.non_tensor_batch[k] = v
        return teacher_output.batch["ref_log_prob"]
