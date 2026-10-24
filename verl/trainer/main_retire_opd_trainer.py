"""Entry point for GRPO+OPD with competence-aware OPD retirement."""

import hydra
import ray
from omegaconf import OmegaConf

from verl.trainer.main_grpo_opd_trainer import configure_grpo_opd_trainer, run_configured_grpo_opd_task
from verl.trainer.ppo.retire_opd import RetireOPDConfig


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_retire_opd_trainer(config)


def configure_retire_opd_trainer(config) -> dict:
    """Configure GRPO+OPD and validate the mandatory retirement inputs."""

    settings = configure_grpo_opd_trainer(config)
    opd_cfg = config.algorithm.opd
    if "teacher_performance" not in opd_cfg:
        raise ValueError("algorithm.opd.teacher_performance is required")
    if not opd_cfg.get("success_metric"):
        raise ValueError("algorithm.opd.success_metric is required")

    retirement_config = RetireOPDConfig(
        opd_coef=settings["opd_coef"],
        teacher_performance=float(opd_cfg.teacher_performance),
        window_size=opd_cfg.get("retirement_window_size", 5),
        alignment_threshold=float(opd_cfg.get("alignment_threshold", 0.0)),
        competence_threshold=float(opd_cfg.get("competence_threshold", 0.9)),
        denominator_epsilon=float(opd_cfg.get("denominator_epsilon", 1e-8)),
    )
    success_metric = str(opd_cfg.success_metric)
    if not success_metric.strip():
        raise ValueError("algorithm.opd.success_metric is required")

    test_freq = config.trainer.test_freq
    if (
        isinstance(test_freq, bool)
        or int(test_freq) != test_freq
        or test_freq <= 0
        or test_freq > retirement_config.window_size
    ):
        raise ValueError(
            "trainer.test_freq must be a positive integer no larger than "
            "algorithm.opd.retirement_window_size"
        )

    return {
        **settings,
        "teacher_performance": retirement_config.teacher_performance,
        "success_metric": success_metric,
    }


def run_retire_opd_trainer(config) -> None:
    if not ray.is_initialized():
        from verl.trainer.constants_ppo import get_ppo_ray_runtime_env

        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.get("ray_init", {})
        runtime_env = OmegaConf.merge(default_runtime_env, ray_init_kwargs.get("runtime_env", {}))
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    runner = RetireOPDTaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)
class RetireOPDTaskRunner:
    def run(self, config):
        settings = configure_retire_opd_trainer(config)
        from verl.trainer.ppo.retire_opd_trainer import RetireOPDTrainer

        print(f"[RetireOPD] teacher_performance: {settings['teacher_performance']}")
        print(f"[RetireOPD] success_metric: {settings['success_metric']}")
        run_configured_grpo_opd_task(config, RetireOPDTrainer, settings, "RetireOPD")


if __name__ == "__main__":
    main()
