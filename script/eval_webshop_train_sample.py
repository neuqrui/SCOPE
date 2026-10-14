"""Sample WebShop training goals with several checkpoints.

Five models: Qwen2.5-3B-Instruct plus global_step 50/100/150/180 from the
length-bonus GRPO run. 180 goals are drawn once from the training pool
(indices 500 and above on the small catalog, the same split as training).
Each goal is rolled out 8 times at temperature 0.7.

Environment settings match examples/grpo_trainer/run_webshop.sh: 15 steps,
history length 2, prompts capped at 4096 tokens, completions at 512,
human_goals off, env seed 0. The training reward adds
coef * (max_steps - steps) / max_steps on successes only.
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

from script.eval_webshop_passk import merge_fsdp_checkpoint
from verl.trainer.ppo.seed_episode_skill import format_episode_steps


N_TASKS = 180
N_ROLLOUTS = 8
MAX_STEPS = 15
MAX_PROMPT_LENGTH = 4096
MAX_RESPONSE_LENGTH = 512
HISTORY_LENGTH = 2
TEMPERATURE = 0.7
ENV_SEED = 0
LENGTH_BONUS_COEF = 5.0
CKPT_ROOT = (
    "/home/liuqirui/SDAR/checkpoints/verl_agent_webshopv1/"
    "grpo_qwen2.5_3b_from_opd30_length_bonus5.0"
)


def training_score(episode_reward: float, length: int) -> float:
    if episode_reward <= 0:
        return 0.0
    efficiency = max(0.0, (MAX_STEPS - length) / MAX_STEPS)
    return float(episode_reward) + LENGTH_BONUS_COEF * efficiency


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


def make_manager(env_num: int):
    from functools import partial
    from types import SimpleNamespace

    from agent_system.environments.env_manager import WebshopEnvironmentManager
    from agent_system.environments.env_package.webshop import build_webshop_envs, webshop_projection

    webshop_root = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "agent_system/environments/env_package/webshop/webshop/data",
    )
    env_kwargs = {
        "observation_mode": "text",
        "num_products": None,
        "human_goals": False,
        "file_path": os.path.join(webshop_root, "items_shuffle_1000.json"),
        "attr_path": os.path.join(webshop_root, "items_ins_v2_1000.json"),
    }
    vector = build_webshop_envs(
        seed=ENV_SEED,
        env_num=env_num,
        group_n=1,
        resources_per_worker={"num_cpus": 0.1, "num_gpus": 0},
        is_train=True,
        env_kwargs=env_kwargs,
    )
    config = SimpleNamespace(env=SimpleNamespace(history_length=HISTORY_LENGTH))
    return WebshopEnvironmentManager(vector, partial(webshop_projection), config)


def build_envs(env_num: int):
    manager = make_manager(env_num)
    pool = [int(i) for i in manager.envs.goal_idxs]
    rng = np.random.RandomState(ENV_SEED)
    chosen = sorted(int(i) for i in rng.choice(pool, size=N_TASKS, replace=False))
    return manager, pool, chosen


def rollout(llm, tokenizer, manager, indices: list[int]) -> tuple[list[dict], int]:
    from vllm import SamplingParams

    sampling = SamplingParams(
        temperature=TEMPERATURE,
        top_p=1.0,
        top_k=-1,
        max_tokens=MAX_RESPONSE_LENGTH,
    )
    obs, _infos = manager.reset(kwargs=None, indices=indices)
    tasks = list(manager.tasks)
    n = len(indices)
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
            prompts=None,
            sampling_params=sampling,
            prompt_token_ids=token_ids,
            use_tqdm=False,
        )
        actions = [""] * n
        responses = [""] * n
        for slot, output in zip(active, outputs):
            responses[int(slot)] = output.outputs[0].text
            actions[int(slot)] = responses[int(slot)]
        next_obs, rewards, dones, infos = manager.step(actions)
        rewards = np.asarray(rewards).reshape(-1)
        dones = np.asarray(dones).reshape(-1).astype(bool)
        for slot in active:
            slot = int(slot)
            info = infos[slot]
            won = bool(info.get("won"))
            traces[slot].append(
                {
                    "step_index": len(traces[slot]),
                    "observation": anchors[slot],
                    "response": responses[slot],
                    "action": actions[slot],
                    "action_valid": bool(info.get("is_action_valid")),
                    "reward": float(rewards[slot]),
                    "done": bool(dones[slot]),
                    "won": won,
                    "task_score": float(info.get("task_score") or 0.0),
                }
            )
            if dones[slot] or won or step_i == MAX_STEPS - 1:
                done[slot] = True
        obs = next_obs
    records = []
    for i in range(n):
        steps = traces[i]
        episode_reward = max((step["reward"] for step in steps), default=0.0)
        success = episode_reward >= 10.0 or any(step["won"] for step in steps)
        task_score = steps[-1]["task_score"] if steps else 0.0
        length = len(steps)
        records.append(
            {
                "goal_index": int(indices[i]),
                "task": tasks[i],
                "episode_reward": float(episode_reward),
                "success": bool(success),
                "task_score": float(task_score),
                "length": int(length),
                "training_score": training_score(episode_reward, length),
                "n_invalid_actions": int(sum(not step["action_valid"] for step in steps)),
                "trajectory": format_episode_steps(steps),
                "steps": steps,
            }
        )
    return records, truncated


def summarize(episodes: list[dict]) -> dict:
    rewards = [ep["episode_reward"] for ep in episodes]
    scores = [ep["training_score"] for ep in episodes]
    lengths = [ep["length"] for ep in episodes]
    task_scores = [ep["task_score"] for ep in episodes]
    success = [1.0 if ep["success"] else 0.0 for ep in episodes]
    return {
        "n_episodes": len(episodes),
        "success_rate": float(np.mean(success)) if success else 0.0,
        "mean_episode_reward": float(np.mean(rewards)) if rewards else 0.0,
        "mean_training_score": float(np.mean(scores)) if scores else 0.0,
        "mean_task_score": float(np.mean(task_scores)) if task_scores else 0.0,
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
    }


def group_tasks(rollouts: list[list[dict]]) -> list[dict]:
    tasks = []
    n = len(rollouts[0])
    for i in range(n):
        episodes = []
        for r, batch in enumerate(rollouts):
            episode = dict(batch[i])
            episode["rollout_index"] = r
            episodes.append(episode)
        tasks.append(
            {
                "goal_index": episodes[0]["goal_index"],
                "task": episodes[0]["task"],
                "success_rate": float(np.mean([ep["success"] for ep in episodes])),
                "mean_episode_reward": float(np.mean([ep["episode_reward"] for ep in episodes])),
                "mean_training_score": float(np.mean([ep["training_score"] for ep in episodes])),
                "mean_task_score": float(np.mean([ep["task_score"] for ep in episodes])),
                "mean_length": float(np.mean([ep["length"] for ep in episodes])),
                "rollouts": episodes,
            }
        )
    return tasks


def config_block(goal_pool: list[int], goal_indices: list[int]) -> dict:
    return {
        "temperature": TEMPERATURE,
        "top_p": 1.0,
        "top_k": -1,
        "n_rollouts": N_ROLLOUTS,
        "n_tasks": N_TASKS,
        "max_steps": MAX_STEPS,
        "max_prompt_length": MAX_PROMPT_LENGTH,
        "max_response_length": MAX_RESPONSE_LENGTH,
        "history_length": HISTORY_LENGTH,
        "env_seed": ENV_SEED,
        "human_goals": False,
        "use_small": True,
        "split": "train",
        "train_goal_pool_size": len(goal_pool),
        "train_goal_pool_first": goal_pool[0],
        "train_goal_pool_last": goal_pool[-1],
        "goal_selection": "RandomState(env_seed).choice(train_pool, 180, replace=False), then sorted",
        "goal_indices": goal_indices,
        "success_length_bonus_coef": LENGTH_BONUS_COEF,
        "success_length_max_steps": MAX_STEPS,
        "episode_reward": "10 on a WebShop score of 1, otherwise 0",
        "training_score": "episode_reward + coef * (max_steps - length) / max_steps when episode_reward > 0, else 0",
        "success": "episode_reward >= 10",
    }


def shared_gpu_utilization(reserve_gib: float = 28.0) -> float:
    """Leave the other job's allocation in place and add a fixed KV budget.

    vLLM sets KV memory to ``utilization * total - peak``, and peak includes
    every other process on the card. A fixed utilization is either too small
    to cover that peak, or large enough to crowd the trainer off the card.
    """
    override = os.environ.get("VLLM_GPU_MEMORY_UTILIZATION")
    if override:
        return float(override)
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    used_bytes = total_bytes - free_bytes
    target = used_bytes + reserve_gib * (1024 ** 3)
    return float(min(0.95, max(0.20, target / total_bytes)))


def resolve_model(name: str, path: str, merged_root: str, base_model: str) -> str:
    actor_shard = os.path.join(path, "actor", "model_world_size_2_rank_0.pt")
    shard = os.path.join(path, "model_world_size_2_rank_0.pt")
    if os.path.isfile(actor_shard) or os.path.isfile(shard):
        merged = os.path.join(merged_root, name)
        actor_dir = os.path.join(path, "actor") if os.path.isfile(actor_shard) else path
        return merge_fsdp_checkpoint(actor_dir, merged, base_model)
    return path


def load_llm(model_path: str):
    import time

    from vllm import LLM

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    last_error = None
    for attempt in range(12):
        utilization = shared_gpu_utilization()
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        print(
            f"vLLM attempt {attempt + 1} gpu_memory_utilization={utilization:.3f} "
            f"free_gib={free_bytes / (1024 ** 3):.1f} model={model_path}",
            flush=True,
        )
        try:
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
        except (AssertionError, RuntimeError, ValueError) as exc:
            last_error = exc
            text = str(exc)
            transient = (
                "memory profiling" in text
                or "No available memory" in text
                or "Engine core initialization failed" in text
            )
            if not transient or attempt == 11:
                raise
            print(f"vLLM init raced the other job, retrying: {text.splitlines()[-1]}", flush=True)
            torch.cuda.empty_cache()
            time.sleep(8)
    raise last_error


def evaluate_model(llm, tokenizer, manager, indices: list[int], meta: dict, output_path: str):
    rollouts = []
    truncated = 0
    chunk = 60
    for r in range(N_ROLLOUTS):
        print(f"  rollout {r + 1}/{N_ROLLOUTS}", flush=True)
        pieces = []
        for start in range(0, len(indices), chunk):
            batch, n_trunc = rollout(llm, tokenizer, manager, indices[start : start + chunk])
            pieces.extend(batch)
            truncated += n_trunc
        rollouts.append(pieces)

    tasks = group_tasks(rollouts)
    episodes = [ep for task in tasks for ep in task["rollouts"]]
    report = {
        "model_name": meta["name"],
        "model_path": meta["resolved_path"],
        "checkpoint_step": meta["step"],
        "config": meta["config"],
        "prompt_truncations": truncated,
        "summary": summarize(episodes),
        "tasks": tasks,
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as handle:
        json.dump(report, handle)
    summary = report["summary"]
    print(
        f"{meta['name']} success={summary['success_rate']:.4f} "
        f"reward={summary['mean_episode_reward']:.3f} "
        f"training_score={summary['mean_training_score']:.3f} "
        f"task_score={summary['mean_task_score']:.3f} "
        f"length={summary['mean_length']:.2f} -> {output_path}",
        flush=True,
    )


def model_specs(base_model: str, ckpt_root: str) -> list[dict]:
    return [
        {"name": "base", "path": base_model, "step": 0},
        {"name": "global_step_50", "path": os.path.join(ckpt_root, "global_step_50"), "step": 50},
        {"name": "global_step_100", "path": os.path.join(ckpt_root, "global_step_100"), "step": 100},
        {"name": "global_step_150", "path": os.path.join(ckpt_root, "global_step_150"), "step": 150},
        {"name": "global_step_180", "path": os.path.join(ckpt_root, "global_step_180"), "step": 180},
    ]


def shard_indices(all_indices: list[int], model_index: int, tasks_per_model: int, shard_index: int, num_shards: int) -> list[int]:
    model_goals = all_indices[model_index * tasks_per_model : (model_index + 1) * tasks_per_model]
    if len(model_goals) != tasks_per_model:
        raise ValueError(f"Model {model_index} expected {tasks_per_model} goals, got {len(model_goals)}")
    shard = model_goals[shard_index::num_shards]
    if not shard:
        raise ValueError(f"Empty shard {shard_index}/{num_shards}")
    return shard


def run_shard(args):
    goals_path = os.path.join(args.output_dir, "goals.json")
    with open(goals_path) as handle:
        goals = json.load(handle)
    all_indices = [int(i) for i in goals["config"]["goal_indices"]]
    specs = model_specs(args.base_model, args.ckpt_root)
    meta = dict(specs[args.model_index])
    indices = shard_indices(all_indices, args.model_index, args.tasks_per_model, args.shard_index, args.num_shards)
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "shards", f"{meta['name']}_shard{args.shard_index}.json")
    if os.path.isfile(output_path):
        print(f"Skip existing {output_path}", flush=True)
        return
    os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
    if not ray.is_initialized():
        ray.init(num_cpus=8, ignore_reinit_error=True, include_dashboard=False)
    resolved = resolve_model(meta["name"], meta["path"], args.merged_dir, args.base_model)
    llm, tokenizer = load_llm(resolved)
    manager = make_manager(len(indices))
    config = dict(goals["config"])
    config["tasks_per_model"] = args.tasks_per_model
    config["num_data_parallel_shards"] = args.num_shards
    config["model_index"] = args.model_index
    config["shard_index"] = args.shard_index
    config["goal_indices"] = indices
    meta["config"] = config
    meta["resolved_path"] = resolved
    print(
        f"{meta['name']} shard {args.shard_index}/{args.num_shards} goals={indices}",
        flush=True,
    )
    try:
        evaluate_model(llm, tokenizer, manager, indices, meta, output_path)
    finally:
        del llm, tokenizer
        torch.cuda.empty_cache()
        manager.envs.close()


def merge_shards(args):
    specs = model_specs(args.base_model, args.ckpt_root)
    meta = specs[args.model_index]
    shard_dir = os.path.join(args.output_dir, "shards")
    paths = [os.path.join(shard_dir, f"{meta['name']}_shard{i}.json") for i in range(args.num_shards)]
    reports = []
    for path in paths:
        with open(path) as handle:
            reports.append(json.load(handle))
    tasks = [task for report in reports for task in report["tasks"]]
    # Each shard stored only its own indices. Restore model order from goals.json.
    with open(os.path.join(args.output_dir, "goals.json")) as handle:
        all_indices = [int(i) for i in json.load(handle)["config"]["goal_indices"]]
    model_goals = all_indices[args.model_index * args.tasks_per_model : (args.model_index + 1) * args.tasks_per_model]
    rank = {goal: pos for pos, goal in enumerate(model_goals)}
    tasks.sort(key=lambda task: rank[task["goal_index"]])
    episodes = [ep for task in tasks for ep in task["rollouts"]]
    config = dict(reports[0]["config"])
    config["goal_indices"] = model_goals
    config["n_tasks"] = len(model_goals)
    config.pop("shard_index", None)
    report = {
        "model_name": meta["name"],
        "model_path": reports[0]["model_path"],
        "checkpoint_step": meta["step"],
        "config": config,
        "prompt_truncations": int(sum(item["prompt_truncations"] for item in reports)),
        "summary": summarize(episodes),
        "tasks": tasks,
    }
    output_path = os.path.join(args.output_dir, f"{meta['name']}.json")
    with open(output_path, "w") as handle:
        json.dump(report, handle)
    summary = report["summary"]
    print(
        f"merged {meta['name']} n={len(tasks)} success={summary['success_rate']:.4f} "
        f"reward={summary['mean_episode_reward']:.3f} -> {output_path}",
        flush=True,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="/data/qirui/SDAR/ckpt/rollouts_val/webshop_train180_t0.7",
    )
    parser.add_argument("--model-index", type=int, default=None)
    parser.add_argument("--model-path", default=None, help="Sample all 180 goals with this single model.")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--tasks-per-model", type=int, default=36)
    parser.add_argument("--merge-shards", action="store_true")
    parser.add_argument(
        "--merged-dir",
        default="/data/qirui/SDAR/ckpt/hf/grpo_qwen2.5_3b_from_opd30_length_bonus5.0",
    )
    parser.add_argument(
        "--base-model",
        default=os.environ.get("MODEL_PATH", "/data/qirui/SDAR/models/Qwen2.5-3B-Instruct"),
    )
    parser.add_argument("--ckpt-root", default=CKPT_ROOT)
    return parser.parse_args()


def run_one_model(args):
    """Roll all 180 training goals with one checkpoint."""
    os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
    if not ray.is_initialized():
        ray.init(num_cpus=24, ignore_reinit_error=True, include_dashboard=False)
    name = args.model_name or os.path.basename(os.path.normpath(args.model_path))
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{name}.json")
    manager, pool, indices = build_envs(60)
    config = config_block(pool, indices)
    with open(os.path.join(args.output_dir, "goals.json"), "w") as handle:
        json.dump({"config": config}, handle, indent=2)
    print(f"Selected {len(indices)} train goals for {name}", flush=True)
    llm, tokenizer = load_llm(args.model_path)
    meta = {
        "name": name,
        "path": args.model_path,
        "step": 0,
        "config": config,
        "resolved_path": args.model_path,
    }
    try:
        evaluate_model(llm, tokenizer, manager, indices, meta, output_path)
    finally:
        del llm, tokenizer
        torch.cuda.empty_cache()
        manager.envs.close()


def main():
    args = parse_args()
    if args.model_path:
        run_one_model(args)
        return
    if args.merge_shards:
        merge_shards(args)
        return
    if args.model_index is not None:
        run_shard(args)
        return
    os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
    if not ray.is_initialized():
        ray.init(num_cpus=24, ignore_reinit_error=True, include_dashboard=False)
    models = [
        {"name": "base", "path": args.base_model, "step": 0},
        {"name": "global_step_50", "path": os.path.join(args.ckpt_root, "global_step_50"), "step": 50},
        {"name": "global_step_100", "path": os.path.join(args.ckpt_root, "global_step_100"), "step": 100},
        {"name": "global_step_150", "path": os.path.join(args.ckpt_root, "global_step_150"), "step": 150},
        {"name": "global_step_180", "path": os.path.join(args.ckpt_root, "global_step_180"), "step": 180},
    ]
    os.makedirs(args.output_dir, exist_ok=True)
    pending = []
    for meta in models:
        output_path = os.path.join(args.output_dir, f"{meta['name']}.json")
        if os.path.isfile(output_path):
            print(f"Skip existing {output_path}", flush=True)
            continue
        pending.append((meta, output_path))
    if not pending:
        print("All model files already exist", flush=True)
        return

    # Grab the GPU before the WebShop workers start, so the memory budget
    # matches the card's current occupant.
    first_meta, first_output = pending[0]
    first_path = resolve_model(first_meta["name"], first_meta["path"], args.merged_dir, args.base_model)
    llm, tokenizer = load_llm(first_path)
    manager, pool, indices = build_envs(60)
    config = config_block(pool, indices)
    with open(os.path.join(args.output_dir, "goals.json"), "w") as handle:
        json.dump({"config": config}, handle, indent=2)
    print(f"Selected {len(indices)} train goals from pool {pool[0]}..{pool[-1]} (n={len(pool)})", flush=True)
    try:
        first_meta = dict(first_meta)
        first_meta["config"] = config
        first_meta["resolved_path"] = first_path
        evaluate_model(llm, tokenizer, manager, indices, first_meta, first_output)
        del llm, tokenizer
        torch.cuda.empty_cache()
        for meta, output_path in pending[1:]:
            resolved = resolve_model(meta["name"], meta["path"], args.merged_dir, args.base_model)
            llm, tokenizer = load_llm(resolved)
            meta = dict(meta)
            meta["config"] = config
            meta["resolved_path"] = resolved
            evaluate_model(llm, tokenizer, manager, indices, meta, output_path)
            del llm, tokenizer
            torch.cuda.empty_cache()
    finally:
        manager.envs.close()


if __name__ == "__main__":
    main()
