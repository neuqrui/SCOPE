"""Single-env WebShop loop.

The environment is SDAR's ``WebshopEnvironmentManager``: the student prompt is the
same history-window text used in training. Only the student's action is executed.
The teacher never steps the env.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import ray
from omegaconf import OmegaConf

from agent_system.environments.env_manager import WebshopEnvironmentManager
from agent_system.environments.env_package.webshop import build_webshop_envs, webshop_projection


def extract_action(text: str) -> str:
    match = re.search(r"<action>(.*?)</action>", str(text or ""), flags=re.IGNORECASE | re.DOTALL)
    if match is None:
        return ""
    return " ".join(match.group(1).strip().split())


def make_webshop_env(
    *,
    seed: int = 0,
    max_steps: int = 15,
    history_length: int = 2,
    use_small: bool = True,
    human_goals: bool = False,
    goal_idx: int | None = None,
    num_cpus: float = 0.5,
):
    if not ray.is_initialized():
        ray.init(num_cpus=max(int(num_cpus) + 1, 2), ignore_reinit_error=True, include_dashboard=False)

    import os

    root = os.path.join(os.path.dirname(__file__), "..", "..", "agent_system", "environments")
    root = os.path.abspath(root)
    if use_small:
        file_path = os.path.join(root, "env_package/webshop/webshop/data/items_shuffle_1000.json")
        attr_path = os.path.join(root, "env_package/webshop/webshop/data/items_ins_v2_1000.json")
    else:
        file_path = os.path.join(root, "env_package/webshop/webshop/data/items_shuffle.json")
        attr_path = os.path.join(root, "env_package/webshop/webshop/data/items_ins_v2.json")

    raw_env = build_webshop_envs(
        seed=seed,
        env_num=1,
        group_n=1,
        is_train=True,
        resources_per_worker={"num_cpus": num_cpus, "num_gpus": 0},
        env_kwargs={
            "observation_mode": "text",
            "num_products": None,
            "human_goals": human_goals,
            "file_path": file_path,
            "attr_path": attr_path,
        },
    )
    if goal_idx is not None:
        raw_env.goal_idxs = np.asarray([int(goal_idx)])
        raw_env._rng = np.random.RandomState(seed)

    config = OmegaConf.create(
        {
            "env": {
                "env_name": "Webshop",
                "seed": seed,
                "max_steps": max_steps,
                "history_length": history_length,
            }
        }
    )
    manager = WebshopEnvironmentManager(raw_env, webshop_projection, config)
    return manager


def run_episode(env, policy, teacher, *, max_steps: int, episode_idx: int = 0) -> dict[str, Any]:
    observations, infos = env.reset({})
    task = env.tasks[0] if getattr(env, "tasks", None) else ""
    steps = []
    done = False
    for turn in range(max_steps):
        student_prompt = str(observations["text"][0])
        anchor = observations["anchor"][0]
        anchor_text = anchor if isinstance(anchor, str) else str(anchor)
        student = policy.generate(student_prompt)
        teacher_step = (
            teacher.act(student_prompt, task=task, observation=anchor_text) if teacher is not None else {}
        )
        next_obs, rewards, dones, step_infos = env.step([student["response"]])
        info = step_infos[0] if step_infos else {}
        reward = float(np.asarray(rewards[0]).item())
        done = bool(np.asarray(dones[0]).item())
        available = None
        if isinstance(infos, (list, tuple)) and infos:
            available = (infos[0] or {}).get("available_actions")
        steps.append(
            {
                "turn": turn,
                "anchor_obs": anchor_text,
                "student_prompt": student_prompt,
                "student_prompt_tokens": student["prompt_tokens"],
                "student_response": student["response"],
                "student_response_ids": student["response_ids"],
                "student_action": extract_action(student["response"]),
                "teacher_action": extract_action(teacher_step.get("teacher_response") or ""),
                "reward": reward,
                "done": done,
                "is_action_valid": bool(info.get("is_action_valid", False)),
                "won": bool(info.get("won", False)),
                "task_score": info.get("task_score"),
                "available_actions": available,
                **teacher_step,
            }
        )
        observations, infos = next_obs, step_infos
        if done:
            break

    episode = {
        "episode_idx": episode_idx,
        "teacher_mode": getattr(teacher, "name", None),
        "task": task,
        "history_length": int(env.config.env.history_length),
        "episode_length": len(steps),
        "episode_reward": float(sum(step["reward"] for step in steps)),
        "won": any(step["won"] for step in steps),
        "task_score": steps[-1]["task_score"] if steps else None,
        "steps": steps,
    }
    if teacher is not None:
        teacher.finalize(episode)
        for step in episode["steps"]:
            step["teacher_action"] = extract_action(step.get("teacher_response") or "")
    return episode
