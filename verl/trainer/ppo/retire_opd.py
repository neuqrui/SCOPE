"""State machine for competence-aware retirement of the OPD objective."""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

RETIREMENT_STATE_FILENAME = "retire_opd_state.json"


@dataclass(frozen=True)
class RetireOPDConfig:
    """Configuration for the non-overlapping-window retirement criterion."""

    opd_coef: float
    teacher_performance: float
    window_size: int = 5
    alignment_threshold: float = 0.0
    competence_threshold: float = 0.9
    denominator_epsilon: float = 1e-8

    def __post_init__(self) -> None:
        if not math.isfinite(self.opd_coef) or self.opd_coef < 0:
            raise ValueError("opd_coef must be finite and nonnegative")
        if not math.isfinite(self.teacher_performance) or self.teacher_performance <= 0:
            raise ValueError("teacher_performance must be finite and positive")
        if isinstance(self.window_size, bool) or int(self.window_size) != self.window_size or self.window_size <= 0:
            raise ValueError("window_size must be a positive integer")
        if not math.isfinite(self.alignment_threshold):
            raise ValueError("alignment_threshold must be finite")
        if not math.isfinite(self.competence_threshold) or self.competence_threshold < 0:
            raise ValueError("competence_threshold must be finite and nonnegative")
        if not math.isfinite(self.denominator_epsilon) or self.denominator_epsilon < 0:
            raise ValueError("denominator_epsilon must be finite and nonnegative")


class RetireOPDScheduler:
    """Retire OPD when alignment stalls after the student becomes competent."""

    def __init__(self, config: RetireOPDConfig):
        self.config = config
        self.opd_coef = float(config.opd_coef)
        self.pending_k_values: list[float] = []
        self.completed_windows: list[dict[str, float | int]] = []
        self.num_evaluated_windows = 0
        self.previous_window_mean: float | None = None
        self.previous_performance: float | None = None
        self.retired = False
        self.trigger_step: int | None = None

    def observe_gap(self, step: int, teacher_student_gap_mean: float) -> dict[str, float | int]:
        """Convert the mean teacher-student log-probability gap to K and buffer it."""

        gap_mean = float(teacher_student_gap_mean)
        if not math.isfinite(gap_mean):
            raise ValueError("teacher_student_gap_mean must be finite")

        k_step = -gap_mean
        self.pending_k_values.append(k_step)
        completed_mean = math.nan
        if len(self.pending_k_values) == self.config.window_size:
            completed_mean = sum(self.pending_k_values) / self.config.window_size
            self.completed_windows.append({"end_step": int(step), "mean": completed_mean})
            self.pending_k_values = []

        return {
            "k_step": k_step,
            "k_window_mean": completed_mean,
            "window_index": len(self.completed_windows),
            "window_fill": len(self.pending_k_values),
        }

    def observe_validation(self, step: int, student_performance: float) -> dict[str, float | int]:
        """Evaluate the next completed window against alignment and competence criteria."""

        performance = float(student_performance)
        if not math.isfinite(performance):
            raise ValueError("student_performance must be finite")

        alignment_progress = math.nan
        competence_ratio = math.nan
        criterion_valid = False
        if self.num_evaluated_windows < len(self.completed_windows):
            window = self.completed_windows[self.num_evaluated_windows]
            current_window_mean = float(window["mean"])
            if self.previous_window_mean is not None and self.previous_performance is not None:
                denominator = self.previous_window_mean
                if abs(denominator) > self.config.denominator_epsilon:
                    alignment_progress = (denominator - current_window_mean) / denominator
                    competence_ratio = (performance + self.previous_performance) / (
                        2.0 * self.config.teacher_performance
                    )
                    criterion_valid = math.isfinite(alignment_progress) and math.isfinite(competence_ratio)

            self.previous_window_mean = current_window_mean
            self.previous_performance = performance
            self.num_evaluated_windows += 1

            if (
                not self.retired
                and criterion_valid
                and alignment_progress <= self.config.alignment_threshold
                and competence_ratio >= self.config.competence_threshold
            ):
                self.retired = True
                self.trigger_step = int(step)
                self.opd_coef = 0.0

        return self.metrics(
            student_performance=performance,
            alignment_progress=alignment_progress,
            competence_ratio=competence_ratio,
            criterion_valid=criterion_valid,
        )

    def metrics(
        self,
        *,
        student_performance: float = math.nan,
        alignment_progress: float = math.nan,
        competence_ratio: float = math.nan,
        criterion_valid: bool = False,
    ) -> dict[str, float | int]:
        latest_window = self.completed_windows[-1]["mean"] if self.completed_windows else math.nan
        return {
            "k_window_mean": float(latest_window),
            "alignment_progress": float(alignment_progress),
            "student_performance": float(student_performance),
            "teacher_performance": self.config.teacher_performance,
            "competence_ratio": float(competence_ratio),
            "alignment_threshold": self.config.alignment_threshold,
            "competence_threshold": self.config.competence_threshold,
            "criterion_valid": float(criterion_valid),
            "retired": float(self.retired),
            "trigger_step": -1 if self.trigger_step is None else self.trigger_step,
            "opd_coef": self.opd_coef,
            "window_index": len(self.completed_windows),
            "window_fill": len(self.pending_k_values),
            "num_evaluated_windows": self.num_evaluated_windows,
        }

    def state_dict(self) -> dict:
        return {
            "version": 1,
            "config": asdict(self.config),
            "opd_coef": self.opd_coef,
            "pending_k_values": self.pending_k_values,
            "completed_windows": self.completed_windows,
            "num_evaluated_windows": self.num_evaluated_windows,
            "previous_window_mean": self.previous_window_mean,
            "previous_performance": self.previous_performance,
            "retired": self.retired,
            "trigger_step": self.trigger_step,
        }

    def load_state_dict(self, state: Mapping) -> None:
        if int(state.get("version", -1)) != 1:
            raise ValueError(f"Unsupported RetireOPD state version: {state.get('version')}")
        if state.get("config") != asdict(self.config):
            raise ValueError("RetireOPD checkpoint config does not match current config")

        self.opd_coef = float(state["opd_coef"])
        self.pending_k_values = [float(value) for value in state["pending_k_values"]]
        self.completed_windows = [dict(window) for window in state["completed_windows"]]
        self.num_evaluated_windows = int(state["num_evaluated_windows"])
        previous_mean = state.get("previous_window_mean")
        previous_performance = state.get("previous_performance")
        self.previous_window_mean = None if previous_mean is None else float(previous_mean)
        self.previous_performance = None if previous_performance is None else float(previous_performance)
        self.retired = bool(state["retired"])
        trigger_step = state.get("trigger_step")
        self.trigger_step = None if trigger_step is None else int(trigger_step)


def save_retirement_state(scheduler: RetireOPDScheduler, checkpoint_folder: str | os.PathLike) -> Path:
    """Atomically save retirement state beside a trainer checkpoint."""

    checkpoint_folder = Path(checkpoint_folder)
    checkpoint_folder.mkdir(parents=True, exist_ok=True)
    state_path = checkpoint_folder / RETIREMENT_STATE_FILENAME
    temporary_path = checkpoint_folder / f".{RETIREMENT_STATE_FILENAME}.tmp"
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(scheduler.state_dict(), handle, indent=2, sort_keys=True, allow_nan=False)
    os.replace(temporary_path, state_path)
    return state_path


def load_retirement_state(scheduler: RetireOPDScheduler, checkpoint_folder: str | os.PathLike) -> bool:
    """Load retirement state if it exists in a trainer checkpoint."""

    state_path = Path(checkpoint_folder) / RETIREMENT_STATE_FILENAME
    if not state_path.exists():
        return False
    with state_path.open(encoding="utf-8") as handle:
        scheduler.load_state_dict(json.load(handle))
    return True

