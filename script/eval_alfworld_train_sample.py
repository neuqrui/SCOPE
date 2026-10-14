"""Sample 180 ALFWorld training games with Qwen2.5-3B-Instruct.

The 180 games cover the six training task types evenly: 30 each. Settings
match examples/grpo_trainer/run_alfworld.sh: 50 steps, history length 2,
prompts capped at 2048 tokens, completions at 512, temperature 1.0.
Each game is rolled out 8 times. A success scores 10, a failure scores 0.
"""

from __future__ import annotations

import os

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import argparse
import json

import numpy as np
import ray
import torch
from transformers import AutoTokenizer

from verl.trainer.ppo.seed_episode_skill import format_episode_steps


N_TASKS = 180
N_ROLLOUTS = 8
MAX_STEPS = 50
MAX_PROMPT_LENGTH = 2048
MAX_RESPONSE_LENGTH = 512
HISTORY_LENGTH = 2
TEMPERATURE = 1.0
ENV_SEED = 0
PER_TYPE = N_TASKS // 6
TASK_TYPES = (
    "pick_and_place_simple",
    "look_at_obj_in_light",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_two_obj_and_place",
)
BASE_MODEL = "/data/qirui/SDAR/models/Qwen2.5-3B-Instruct"


def encode_prompts(tokenizer, texts: list[str]) -> tuple[list[list[int]], int]:
    truncated = 0
    token_ids = []
    for text in texts:
        template_kwargs = {}
        if os.environ.get("SCOPE_DISABLE_THINKING") == "1":
            template_kwargs["enable_thinking"] = False
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            add_generation_prompt=True,
            tokenize=False,
            **template_kwargs,
        )
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(ids) > MAX_PROMPT_LENGTH:
            ids = ids[-MAX_PROMPT_LENGTH:]
            truncated += 1
        token_ids.append(ids)
    return token_ids, truncated


def alf_config_path() -> str:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(here, "agent_system/environments/env_package/alfworld/configs/config_tw.yaml")


def task_type_of(game_file: str) -> str:
    traj_path = os.path.join(os.path.dirname(game_file), "traj_data.json")
    with open(traj_path) as handle:
        task_type = json.load(handle)["task_type"]
    if task_type not in TASK_TYPES:
        raise ValueError(f"Unexpected task type {task_type} in {traj_path}")
    return task_type


def select_games(env_num: int) -> list[dict]:
    """30 solvable training games from each of the six task types."""
    from agent_system.environments.env_package.alfworld.alfworld.agents.environment.alfred_tw_env import AlfredTWEnv
    from agent_system.environments.env_package.alfworld.envs import load_config_file

    if N_TASKS % env_num != 0:
        raise ValueError(f"env_num={env_num} must divide {N_TASKS}")
    config = load_config_file(alf_config_path())
    pool = AlfredTWEnv(config, train_eval="train").game_files
    grouped = {task_type: [] for task_type in TASK_TYPES}
    for game_file in pool:
        grouped[task_type_of(game_file)].append(game_file)
    rng = np.random.RandomState(ENV_SEED)
    chosen = {task_type: [] for task_type in TASK_TYPES}
    for task_type in TASK_TYPES:
        files = sorted(grouped[task_type])
        if len(files) < PER_TYPE:
            raise RuntimeError(f"{task_type} has {len(files)} solvable train games, need {PER_TYPE}")
        pick = rng.choice(len(files), size=PER_TYPE, replace=False)
        chosen[task_type] = [files[int(i)] for i in sorted(pick)]
    # Round-robin so each worker shard contains every type.
    ordered = []
    for index in range(PER_TYPE):
        for task_type in TASK_TYPES:
            game_file = chosen[task_type][index]
            ordered.append(
                {
                    "game_index": len(ordered),
                    "task_type": task_type,
                    "game_file": game_file,
                }
            )
    return ordered


def make_manager(games: list[dict], env_num: int):
    from functools import partial
    from types import SimpleNamespace

    from agent_system.environments.env_manager import AlfWorldEnvironmentManager
    from agent_system.environments.env_package.alfworld import alfworld_projection
    from agent_system.environments.env_package.alfworld.envs import AlfworldEnvs

    if len(games) % env_num != 0:
        raise ValueError(f"env_num={env_num} must divide {len(games)} games")
    shard_size = len(games) // env_num
    groups = []
    for shard in range(env_num):
        groups.append([item["game_file"] for item in games[shard * shard_size : (shard + 1) * shard_size]])
    vector = AlfworldEnvs(
        alf_config_path=alf_config_path(),
        seed=ENV_SEED,
        env_num=env_num,
        group_n=1,
        resources_per_worker={"num_cpus": 0.1, "num_gpus": 0},
        is_train=True,
        game_file_groups=groups,
    )
    config = SimpleNamespace(env=SimpleNamespace(history_length=HISTORY_LENGTH))
    return AlfWorldEnvironmentManager(vector, partial(alfworld_projection), config), shard_size


def rollout(llm, tokenizer, manager) -> tuple[list[dict], int]:
    from vllm import SamplingParams

    sampling = SamplingParams(
        temperature=TEMPERATURE,
        top_p=1.0,
        top_k=-1,
        max_tokens=MAX_RESPONSE_LENGTH,
    )
    obs, infos = manager.reset(kwargs=None)
    n = len(obs["text"])
    game_files = [info.get("extra.gamefile") for info in infos]
    tasks = list(manager.tasks)
    done = np.zeros(n, dtype=bool)
    traces = [[] for _ in range(n)]
    truncated = 0
    for step_i in range(MAX_STEPS):
        active = np.flatnonzero(~done)
        if len(active) == 0:
            break
        prompts = [obs["text"][i] for i in active]
        anchors = list(obs["anchor"])
        token_ids, n_trunc = encode_prompts(tokenizer, prompts)
        truncated += n_trunc
        outputs = llm.generate(
            [{"prompt_token_ids": ids} for ids in token_ids],
            sampling_params=sampling,
            use_tqdm=False,
        )
        actions = [""] * n
        responses = [""] * n
        for slot, output in zip(active, outputs):
            responses[int(slot)] = output.outputs[0].text
            actions[int(slot)] = responses[int(slot)]
        next_obs, rewards, dones, next_infos = manager.step(actions)
        rewards = np.asarray(rewards).reshape(-1)
        dones = np.asarray(dones).reshape(-1).astype(bool)
        for slot in active:
            slot = int(slot)
            info = next_infos[slot]
            won = bool(info.get("won"))
            executed = manager.memory[slot][-1]["action"]
            traces[slot].append(
                {
                    "step_index": len(traces[slot]),
                    "observation": anchors[slot],
                    "response": responses[slot],
                    "action": executed,
                    "action_valid": bool(info.get("is_action_valid")),
                    "reward": float(rewards[slot]),
                    "done": bool(dones[slot]),
                    "won": won,
                }
            )
            if dones[slot] or won or step_i == MAX_STEPS - 1:
                done[slot] = True
        obs = next_obs
    records = []
    for i in range(n):
        steps = traces[i]
        success = any(step["won"] for step in steps)
        episode_reward = 10.0 if success else 0.0
        records.append(
            {
                "game_file": game_files[i],
                "task_type": task_type_of(game_files[i]) if game_files[i] else "",
                "task": tasks[i],
                "episode_reward": episode_reward,
                "success": bool(success),
                "length": len(steps),
                "n_invalid_actions": int(sum(not step["action_valid"] for step in steps)),
                "trajectory": format_episode_steps(steps),
                "steps": steps,
            }
        )
    return records, truncated


def load_llm(model_path: str):
    from vllm import LLM

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    utilization = float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.6"))
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        dtype="bfloat16",
        gpu_memory_utilization=utilization,
        max_model_len=MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH,
        max_num_seqs=16,
        enforce_eager=True,
        trust_remote_code=True,
        disable_log_stats=True,
    )
    return llm, tokenizer


def config_block(games: list[dict], env_num: int) -> dict:
    counts = {task_type: sum(item["task_type"] == task_type for item in games) for task_type in TASK_TYPES}
    return {
        "model": BASE_MODEL,
        "temperature": TEMPERATURE,
        "top_p": 1.0,
        "top_k": -1,
        "n_rollouts": N_ROLLOUTS,
        "n_tasks": N_TASKS,
        "per_type": PER_TYPE,
        "task_counts": counts,
        "max_steps": MAX_STEPS,
        "max_prompt_length": MAX_PROMPT_LENGTH,
        "max_response_length": MAX_RESPONSE_LENGTH,
        "history_length": HISTORY_LENGTH,
        "env_seed": ENV_SEED,
        "env_num": env_num,
        "split": "train",
        "task_types": list(TASK_TYPES),
        "selection": "30 games from each train task type, RandomState(0).choice within the type",
        "episode_reward": "10 if won else 0",
        "success": "info won is true",
        "games": games,
    }


def summarize(tasks: list[dict], truncated: int, model_path: str, config: dict) -> dict:
    flat = [ep for task in tasks for ep in task["rollouts"]]
    return {
        "model_name": "qwen2.5_3b",
        "model_path": model_path,
        "config": config,
        "prompt_truncations": truncated,
        "summary": {
            "n_episodes": len(flat),
            "success_rate": float(np.mean([ep["success"] for ep in flat])),
            "mean_episode_reward": float(np.mean([ep["episode_reward"] for ep in flat])),
            "mean_length": float(np.mean([ep["length"] for ep in flat])),
            "success_rate_by_type": {
                task_type: float(np.mean([ep["success"] for ep in flat if ep["task_type"] == task_type]))
                for task_type in TASK_TYPES
            },
        },
        "tasks": tasks,
    }


def tasks_from_rollouts(games: list[dict], by_file: dict) -> list[dict]:
    tasks = []
    for item in games:
        episodes = by_file[item["game_file"]]
        if len(episodes) != N_ROLLOUTS:
            raise RuntimeError(f"{item['game_file']} has {len(episodes)} rollouts, expected {N_ROLLOUTS}")
        for rollout_index, episode in enumerate(episodes):
            episode["rollout_index"] = rollout_index
            episode["game_index"] = item["game_index"]
        tasks.append(
            {
                "game_index": item["game_index"],
                "game_file": item["game_file"],
                "task_type": item["task_type"],
                "task": episodes[0]["task"],
                "success_rate": float(np.mean([ep["success"] for ep in episodes])),
                "mean_episode_reward": float(np.mean([ep["episode_reward"] for ep in episodes])),
                "mean_length": float(np.mean([ep["length"] for ep in episodes])),
                "rollouts": episodes,
            }
        )
    return tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path", default=BASE_MODEL)
    parser.add_argument("--env-num", type=int, default=15)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--merge-shards", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    games_path = os.path.join(args.output_dir, "games.json")
    if os.path.isfile(games_path):
        with open(games_path) as handle:
            config = json.load(handle)
        games = config["games"]
    else:
        games = select_games(args.env_num)
        config = config_block(games, args.env_num)
        with open(games_path, "w") as handle:
            json.dump(config, handle, indent=2)
        print("task counts", config["task_counts"], flush=True)
    if args.prepare_only:
        return

    output_path = os.path.join(args.output_dir, "qwen2.5_3b.json")
    if args.merge_shards:
        if len(games) % args.num_shards != 0:
            raise ValueError(f"num_shards={args.num_shards} must divide {len(games)} games")
        shard_dir = os.path.join(args.output_dir, "shards")
        parts = []
        truncated = 0
        for shard_index in range(args.num_shards):
            shard_path = os.path.join(shard_dir, f"shard{shard_index}.json")
            with open(shard_path) as handle:
                part = json.load(handle)
            parts.append(part)
            truncated += int(part["prompt_truncations"])
        by_index = {task["game_index"]: task for part in parts for task in part["tasks"]}
        if len(by_index) != len(games):
            raise RuntimeError(f"merged {len(by_index)} games, expected {len(games)}")
        tasks = [by_index[item["game_index"]] for item in games]
        report = summarize(tasks, truncated, args.model_path, config)
        with open(output_path, "w") as handle:
            json.dump(report, handle)
        print(json.dumps(report["summary"], indent=2), flush=True)
        print(f"wrote {output_path}", flush=True)
        return

    if args.num_shards == 1:
        shard_path = output_path
        shard_games = games
    else:
        if len(games) % args.num_shards != 0:
            raise ValueError(f"num_shards={args.num_shards} must divide {len(games)} games")
        if not 0 <= args.shard_index < args.num_shards:
            raise ValueError(f"shard_index={args.shard_index} out of range")
        size = len(games) // args.num_shards
        shard_games = games[args.shard_index * size : (args.shard_index + 1) * size]
        shard_dir = os.path.join(args.output_dir, "shards")
        os.makedirs(shard_dir, exist_ok=True)
        shard_path = os.path.join(shard_dir, f"shard{args.shard_index}.json")
    if os.path.isfile(shard_path):
        print(f"Skip existing {shard_path}", flush=True)
        return

    os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
    if not ray.is_initialized():
        ray.init(num_cpus=max(8, args.env_num), ignore_reinit_error=True)
    manager, shard_size = make_manager(shard_games, args.env_num)
    llm, tokenizer = load_llm(args.model_path)
    by_file = {item["game_file"]: [] for item in shard_games}
    truncated = 0
    passes = N_ROLLOUTS * shard_size
    for index in range(passes):
        print(f"shard {args.shard_index} pass {index + 1}/{passes}", flush=True)
        records, n_trunc = rollout(llm, tokenizer, manager)
        truncated += n_trunc
        for record in records:
            by_file[record["game_file"]].append(record)
    tasks = tasks_from_rollouts(shard_games, by_file)
    report = summarize(tasks, truncated, args.model_path, config)
    with open(shard_path, "w") as handle:
        json.dump(report, handle)
    print(json.dumps(report["summary"], indent=2), flush=True)
    print(f"wrote {shard_path}", flush=True)
    manager.envs.close()


if __name__ == "__main__":
    main()
