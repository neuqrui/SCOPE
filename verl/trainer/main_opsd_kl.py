"""
Main entry point for true OPSD (ungated sampled-token KL distillation).

Reuses SkillSDRayTrainer for the privileged teacher forward pass, but enables
``use_opsd_kl_loss`` in the actor instead of SDAR's gated loss.

Sampled-token objective (default)::

    A_t = stopgrad(log π_T - log π_S)
    L   = agg(-A_t * log π_S)

Config namespace: ``algorithm.opsd_kl.*``
  - opsd_coef: weight on the distillation loss
  - objective: sampled_kl | reverse_kl (default sampled_kl)
  - skills_dir / skill_all: SkillProvider settings

Not to be confused with ``main_opsd`` (AgentOPSD Bayesian credit assignment).
"""

import hydra
import ray
from omegaconf import OmegaConf


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_opsd_kl(config)


def run_opsd_kl(config) -> None:
    if not ray.is_initialized():
        from verl.trainer.constants_ppo import get_ppo_ray_runtime_env

        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})

        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    runner = OPSDKlTaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)
class OPSDKlTaskRunner:
    def run(self, config):
        from pprint import pprint

        from omegaconf import OmegaConf, open_dict

        from verl.utils.fs import copy_to_local

        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        opsd_kl_cfg = config.algorithm.get("opsd_kl", {})
        with open_dict(config):
            config.actor_rollout_ref.actor.use_opsd_kl_loss = True
            config.actor_rollout_ref.actor.use_sdar_loss = False
            config.actor_rollout_ref.actor.use_sdl_loss = False
            config.actor_rollout_ref.actor.log_opsd_kl_reference = False
            config.actor_rollout_ref.actor.opsd_kl_loss_coef = opsd_kl_cfg.get("opsd_coef", 0.1)
            config.actor_rollout_ref.actor.opsd_kl_objective = opsd_kl_cfg.get("objective", "sampled_kl")
            config.actor_rollout_ref.actor.opsd_kl_gate = opsd_kl_cfg.get("gate", "none")
            config.actor_rollout_ref.actor.opsd_kl_gate_beta = opsd_kl_cfg.get("gate_beta", 5.0)
            config.actor_rollout_ref.actor.opsd_kl_advantage_clip = float(opsd_kl_cfg.get("advantage_clip", 10.0))

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )

        from agent_system.environments import make_envs

        envs, val_envs = make_envs(config)

        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        if config.actor_rollout_ref.rollout.name in ["vllm"]:
            from verl.utils.vllm_utils import is_version_ge

            if config.actor_rollout_ref.model.get("lora_rank", 0) > 0:
                if not is_version_ge(pkg="vllm", minver="0.7.3"):
                    raise NotImplementedError("PPO LoRA is not supported before vllm 0.7.3")

        if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
            assert config.critic.strategy in ["fsdp", "fsdp2"]
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = ActorRolloutRefWorker
            ray_worker_group_cls = NVMegatronRayWorkerGroup

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        if config.reward_model.enable:
            if config.reward_model.strategy in ["fsdp", "fsdp2"]:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        reward_manager_name = config.reward_model.get("reward_manager", "episode")
        if reward_manager_name == "episode":
            from agent_system.reward_manager import EpisodeRewardManager

            reward_manager_cls = EpisodeRewardManager
        else:
            raise NotImplementedError

        reward_kwargs = dict(
            normalize_by_length=False,
            success_length_bonus_coef=config.reward_model.get("success_length_bonus_coef", 0.0),
            success_length_max_steps=config.reward_model.get("success_length_max_steps", 15),
            success_length_pairwise_tau=config.reward_model.get("success_length_pairwise_tau", 0.0),
            success_length_pairwise_coef=config.reward_model.get("success_length_pairwise_coef", 5.0),
        )
        reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=0, **reward_kwargs)
        # Validation keeps the original episode reward. Length terms are training-only.
        val_reward_fn = reward_manager_cls(
            tokenizer=tokenizer,
            num_examine=1,
            normalize_by_length=False,
            success_length_bonus_coef=0.0,
            success_length_max_steps=config.reward_model.get("success_length_max_steps", 15),
            success_length_pairwise_tau=0.0,
        )

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        assert config.actor_rollout_ref.rollout.n == 1, (
            "In verl, actor_rollout_ref.rollout.n>1 is for GRPO. "
            "In verl+env, we keep n=1, and achieve GRPO by env.rollout.n"
        )

        from agent_system.multi_turn_rollout import TrajectoryCollector

        traj_collector = TrajectoryCollector(config=config, tokenizer=tokenizer, processor=processor)

        from verl.utils.dataset.rl_dataset import collate_fn
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
        train_sampler = create_rl_sampler(config.data, train_dataset)

        from verl.trainer.ppo.rlsd_utils import SkillProvider

        skills_dir = opsd_kl_cfg.get("skills_dir", "skills/alfworld")
        skill_all = opsd_kl_cfg.get("skill_all", False)
        skill_provider = SkillProvider(skills_dir=skills_dir, skill_all=skill_all)
        skill_source = str(opsd_kl_cfg.get("skill_source", "skill_bank"))
        print(f"[OPSD-KL] skill_source: {skill_source}")
        print(f"[OPSD-KL] Loaded skills from {skills_dir}")
        print(f"[OPSD-KL] Available skills: {list(skill_provider.skill_contents.keys())}")
        print(f"[OPSD-KL] Task-to-skill mapping: {skill_provider.task_to_skill}")
        print(f"[OPSD-KL] opsd_coef: {config.actor_rollout_ref.actor.opsd_kl_loss_coef}")
        print(f"[OPSD-KL] objective: {config.actor_rollout_ref.actor.opsd_kl_objective}")
        print(f"[OPSD-KL] gate: {config.actor_rollout_ref.actor.opsd_kl_gate}")
        print(f"[OPSD-KL] gate_beta: {config.actor_rollout_ref.actor.opsd_kl_gate_beta}")
        print(f"[OPSD-KL] advantage_clip: {config.actor_rollout_ref.actor.opsd_kl_advantage_clip}")

        from verl.trainer.ppo.skillsd_ray_trainer import SkillSDRayTrainer

        trainer = SkillSDRayTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=config.trainer.device,
            traj_collector=traj_collector,
            envs=envs,
            val_envs=val_envs,
            skill_provider=skill_provider,
        )
        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    main()
