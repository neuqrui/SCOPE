"""GRPO+OPD trainer with competence-aware OPD retirement."""

import math
import os

from verl import DataProto
from verl.trainer.ppo.grpo_opd_trainer import GRPOOPDTrainer
from verl.trainer.ppo.retire_opd import (
    RetireOPDConfig,
    RetireOPDScheduler,
    load_retirement_state,
    save_retirement_state,
)


def get_validation_scalar(val_metrics: dict, metric_name: str) -> float:
    """Read one finite scalar from validation metrics."""

    if metric_name not in val_metrics:
        raise KeyError(f"Validation metric {metric_name!r} is required for RetireOPD")
    value = float(val_metrics[metric_name])
    if not math.isfinite(value):
        raise ValueError(f"Validation metric {metric_name!r} must be finite")
    return value


def _prefix_metrics(metrics: dict[str, float | int]) -> dict[str, float | int]:
    return {f"retire_opd/{key}": value for key, value in metrics.items()}


class RetireOPDTrainer(GRPOOPDTrainer):
    """Retire the fixed OPD auxiliary term once both criteria are satisfied."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        opd_cfg = self.config.algorithm.opd
        self.success_metric = str(opd_cfg.success_metric)
        self.retirement = RetireOPDScheduler(
            RetireOPDConfig(
                opd_coef=self.opd_coef,
                teacher_performance=float(opd_cfg.teacher_performance),
                window_size=opd_cfg.get("retirement_window_size", 5),
                alignment_threshold=float(opd_cfg.get("alignment_threshold", 0.0)),
                competence_threshold=float(opd_cfg.get("competence_threshold", 0.9)),
                denominator_epsilon=float(opd_cfg.get("denominator_epsilon", 1e-8)),
            )
        )

    def _get_sdl_lambda(self, step: int) -> float:
        return self.retirement.opd_coef

    def _get_training_teacher_log_probs(self, batch: DataProto):
        if self.retirement.retired:
            placeholder = batch.batch["old_log_probs"].detach().clone()
            return placeholder, {"retire_opd/teacher_forward_skipped": 1.0}
        return self._compute_teacher_log_probs(batch), {"retire_opd/teacher_forward_skipped": 0.0}

    def _after_teacher_student_gap(self, step: int, gap_mean: float) -> dict[str, float | int]:
        if self.retirement.retired:
            return _prefix_metrics(self.retirement.metrics())
        return _prefix_metrics(self.retirement.observe_gap(step, gap_mean))

    def _after_validation(self, step: int, val_metrics: dict) -> dict[str, float | int]:
        performance = get_validation_scalar(val_metrics, self.success_metric)
        return _prefix_metrics(self.retirement.observe_validation(step, performance))

    def _checkpoint_folder(self, *, for_resume: bool = False) -> str:
        if for_resume and self.config.trainer.resume_mode == "resume_path":
            return os.path.abspath(self.config.trainer.resume_from_path)
        return os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")

    def _save_checkpoint(self):
        super()._save_checkpoint()
        save_retirement_state(self.retirement, self._checkpoint_folder())

    def _load_checkpoint(self):
        result = super()._load_checkpoint()
        checkpoint_folder = self._checkpoint_folder(for_resume=True)
        if self.global_steps > 0 and not load_retirement_state(self.retirement, checkpoint_folder):
            raise FileNotFoundError(f"RetireOPD state is missing from {checkpoint_folder}")
        return result
