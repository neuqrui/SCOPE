"""Sample 180 Search-R1 training questions, eight rollouts each.

Settings match examples/sdar_trainer/run_search_3b.sh: 4 steps, history
length 4, prompts capped at 4096 tokens from the left, completions at 512.
The retrieval server must already be listening on SEARCH_URL.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from verl.trainer.ppo.seed_episode_skill import format_episode_steps


N_TASKS = 180
N_ROLLOUTS = 8
MAX_STEPS = 4
MAX_PROMPT_LENGTH = 4096
MAX_RESPONSE_LENGTH = 512
HISTORY_LENGTH = 4
TEMPERATURE = 1.0
ENV_SEED = 0
ENV_NUM = 16


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def encode_prompts(tokenizer, texts: list[str]) -> tuple[list[list[int]], int]:
    truncated = 0
    token_ids = []
    template_kwargs = {}
    if os.environ.get("SCOPE_DISABLE_THINKING") == "1":
        template_kwargs["enable_thinking"] = False
    for text in texts:
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


def select_rows(parquet_path: str) -> list[dict]:
    frame = pd.read_parquet(parquet_path, columns=["data_source", "env_kwargs"])
    if len(frame) < N_TASKS:
        raise RuntimeError(f"{parquet_path} has {len(frame)} rows, need {N_TASKS}")
    rng = np.random.RandomState(ENV_SEED)
    picked = sorted(int(i) for i in rng.choice(len(frame), size=N_TASKS, replace=False))
    rows = []
    for row_index in picked:
        item = frame.iloc[row_index]
        env_kwargs = dict(item["env_kwargs"])
        rows.append(
            {
                "row_index": row_index,
                "question": str(env_kwargs["question"]),
                "ground_truth": env_kwargs["ground_truth"],
                "data_source": str(env_kwargs.get("data_source") or item["data_source"]),
            }
        )
    return rows


def make_manager():
    from functools import partial
    from types import SimpleNamespace

    from agent_system.environments.env_manager import SearchEnvironmentManager
    from agent_system.environments.env_package.search import build_search_envs, search_projection

    search_url = os.environ.get("SEARCH_URL", "http://127.0.0.1:8000/retrieve")
    env_config = OmegaConf.create(
        {
            "max_steps": MAX_STEPS,
            "history_length": HISTORY_LENGTH,
            "search": {
                "search_url": search_url,
                "topk": 3,
                "timeout": 60,
                "log_requests": False,
            },
        }
    )
    vector = build_search_envs(seed=ENV_SEED, env_num=ENV_NUM, group_n=1, is_train=True, env_config=env_config)
    config = SimpleNamespace(env=SimpleNamespace(history_length=HISTORY_LENGTH))
    return SearchEnvironmentManager(vector, partial(search_projection), config)


def rollout(llm, tokenizer, manager, rows: list[dict]) -> tuple[list[dict], int]:
    from vllm import SamplingParams

    sampling = SamplingParams(temperature=TEMPERATURE, top_p=1.0, top_k=-1, max_tokens=MAX_RESPONSE_LENGTH)
    kwargs = [
        {
            "question": row["question"],
            "ground_truth": row["ground_truth"],
            "data_source": row["data_source"],
        }
        for row in rows
    ]
    obs, _infos = manager.reset(kwargs=kwargs)
    n = len(rows)
    done = np.zeros(n, dtype=bool)
    traces = [[] for _ in range(n)]
    rewards = np.zeros(n, dtype=np.float64)
    truncated = 0
    for step_i in range(MAX_STEPS):
        active = np.flatnonzero(~done)
        if len(active) == 0:
            break
        prompts = [obs["text"][i] for i in active]
        anchors = list(obs["anchor"])
        token_ids, n_trunc = encode_prompts(tokenizer, prompts)
        truncated += n_trunc
        outputs = llm.generate(prompts=None, sampling_params=sampling, prompt_token_ids=token_ids, use_tqdm=False)
        actions = [""] * n
        responses = [""] * n
        for slot, output in zip(active, outputs):
            responses[int(slot)] = output.outputs[0].text
            actions[int(slot)] = responses[int(slot)]
        next_obs, step_rewards, dones, infos = manager.step(actions)
        step_rewards = np.asarray(step_rewards).reshape(-1)
        dones = np.asarray(dones).reshape(-1).astype(bool)
        for slot in active:
            slot = int(slot)
            info = infos[slot]
            rewards[slot] = float(step_rewards[slot])
            traces[slot].append(
                {
                    "step_index": len(traces[slot]),
                    "observation": anchors[slot],
                    "response": responses[slot],
                    "action": actions[slot],
                    "action_valid": bool(info.get("is_action_valid")),
                    "reward": float(step_rewards[slot]),
                    "done": bool(dones[slot]),
                    "won": bool(info.get("won")),
                }
            )
            if dones[slot] or step_i == MAX_STEPS - 1:
                done[slot] = True
        obs = next_obs
    records = []
    for i, row in enumerate(rows):
        steps = traces[i]
        success = bool(rewards[i] >= 1.0 or any(step["won"] for step in steps))
        records.append(
            {
                "row_index": row["row_index"],
                "data_source": row["data_source"],
                "question": row["question"],
                "ground_truth": _jsonable(row["ground_truth"]),
                "task": row["question"],
                "episode_reward": float(rewards[i]),
                "success": success,
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
    utilization = float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.5"))
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name", default="model")
    parser.add_argument("--train-parquet", required=True)
    args = parser.parse_args()

    rows = select_rows(args.train_parquet)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "questions.json"), "w") as handle:
        json.dump([{"row_index": row["row_index"], "question": row["question"]} for row in rows], handle, indent=2)

    llm, tokenizer = load_llm(args.model_path)
    manager = make_manager()
    rollouts = []
    truncated = 0
    try:
        for rollout_index in range(N_ROLLOUTS):
            print(f"rollout {rollout_index + 1}/{N_ROLLOUTS}", flush=True)
            pieces = []
            for start in range(0, len(rows), ENV_NUM):
                batch, n_trunc = rollout(llm, tokenizer, manager, rows[start : start + ENV_NUM])
                pieces.extend(batch)
                truncated += n_trunc
            for episode in pieces:
                episode["rollout_index"] = rollout_index
            rollouts.append(pieces)
    finally:
        manager.envs.close()

    tasks = []
    for index, row in enumerate(rows):
        episodes = [batch[index] for batch in rollouts]
        tasks.append(
            {
                "row_index": row["row_index"],
                "question": row["question"],
                "data_source": row["data_source"],
                "success_rate": float(np.mean([ep["success"] for ep in episodes])),
                "mean_episode_reward": float(np.mean([ep["episode_reward"] for ep in episodes])),
                "mean_length": float(np.mean([ep["length"] for ep in episodes])),
                "rollouts": episodes,
            }
        )
    flat = [ep for task in tasks for ep in task["rollouts"]]
    report = {
        "model_name": args.model_name,
        "model_path": args.model_path,
        "config": {
            "temperature": TEMPERATURE,
            "n_rollouts": N_ROLLOUTS,
            "n_tasks": N_TASKS,
            "max_steps": MAX_STEPS,
            "history_length": HISTORY_LENGTH,
            "max_prompt_length": MAX_PROMPT_LENGTH,
            "max_response_length": MAX_RESPONSE_LENGTH,
            "env_seed": ENV_SEED,
            "selection": "RandomState(0).choice of 180 train rows, then sorted",
            "search_url": os.environ.get("SEARCH_URL", "http://127.0.0.1:8000/retrieve"),
        },
        "prompt_truncations": truncated,
        "summary": {
            "n_episodes": len(flat),
            "success_rate": float(np.mean([ep["success"] for ep in flat])),
            "mean_episode_reward": float(np.mean([ep["episode_reward"] for ep in flat])),
            "mean_length": float(np.mean([ep["length"] for ep in flat])),
        },
        "tasks": tasks,
    }
    output_path = os.path.join(args.output_dir, f"{args.model_name}.json")
    with open(output_path, "w") as handle:
        json.dump(report, handle)
    summary = report["summary"]
    print(
        f"{args.model_name} success={summary['success_rate']:.4f} "
        f"reward={summary['mean_episode_reward']:.3f} length={summary['mean_length']:.2f} -> {output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
