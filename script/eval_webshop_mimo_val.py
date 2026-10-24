"""Score MiMo on the WebShop validation batch used by run_webshop_3b.sh.

``val_before_train`` builds the val env with ``seed=env.seed+1000`` and
``val_batch_size`` workers. The first reset draws 128 indices from the
held-out prefix ``range(500)``. Worker ``i`` is shuffled with seed
``1000+i`` and reset to that index. Sampling matches the script:
temperature 0.4, top_p 1.0, top_k -1, max 512 new tokens, 15 steps,
history window 2.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field

import numpy as np
import ray
import requests
from tqdm import tqdm
from transformers import AutoTokenizer

from agent_system.environments.env_package.webshop.envs import WebshopWorker
from agent_system.environments.env_package.webshop.projection import webshop_projection
from agent_system.environments.prompts.webshop import WEBSHOP_TEMPLATE, WEBSHOP_TEMPLATE_NO_HIS


def _parse_args():
    parser = argparse.ArgumentParser(description="MiMo WebShop validation score")
    parser.add_argument(
        "--model-path",
        default="work/models/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8100/v1")
    parser.add_argument("--out-dir", default="work/reject_sample/webshop_mimo_val")
    parser.add_argument("--val-size", type=int, default=128)
    parser.add_argument("--heldout", type=int, default=500)
    parser.add_argument("--env-seed", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--history-length", type=int, default=2)
    parser.add_argument("--max-prompt-length", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--num-cpus-per-env", type=float, default=0.1)
    return parser.parse_args()


def _catalog_paths():
    root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "agent_system", "environments")
    )
    return (
        os.path.join(root, "env_package/webshop/webshop/data/items_shuffle_1000.json"),
        os.path.join(root, "env_package/webshop/webshop/data/items_ins_v2_1000.json"),
    )


def _format_obs(task: str, text_obs: str) -> str:
    parts = text_obs.split(" [SEP] ")
    try:
        index = parts.index(task)
        return " [SEP] ".join(f"'{p}'" for p in parts[index + 1 :])
    except ValueError:
        return text_obs


def _extract_task(text_obs: str) -> str:
    parts = text_obs.split(" [SEP] ")
    if len(parts) < 3 or parts[1] != "Instruction:":
        raise RuntimeError(f"Unexpected WebShop observation: {text_obs[:240]!r}")
    return parts[2]


def _format_avail_actions(avail: dict) -> str:
    actions = []
    if avail.get("has_search_bar"):
        actions.append("search[<your query>]")
    for txt in avail["clickables"]:
        actions.append(f"click[{txt}]")
    return "\n".join(f"'{s}'," for s in actions)


def _history_text(history: list[dict], history_length: int) -> tuple[str, int]:
    recent = history[-history_length:]
    valid_len = len(recent)
    start_idx = len(history) - valid_len
    lines = []
    for j, rec in enumerate(recent):
        step_num = start_idx + j + 1
        lines.append(
            f"[Observation {step_num}: '{rec['text_obs']}', Action {step_num}: '{rec['action']}']"
        )
    return "\n".join(lines), valid_len


def build_user_prompt(slot: "Slot", history_length: int) -> str:
    avail = _format_avail_actions(slot.info["available_actions"])
    if history_length <= 0 or not slot.history:
        return WEBSHOP_TEMPLATE_NO_HIS.format(
            task_description=slot.task,
            current_observation=slot.pre_text_obs,
            available_actions=avail,
        )
    memory, valid_len = _history_text(slot.history, history_length)
    obs = WEBSHOP_TEMPLATE.format(
        task_description=slot.task,
        step_count=len(slot.history),
        history_length=valid_len,
        action_history=memory,
        current_step=len(slot.history) + 1,
        current_observation=slot.pre_text_obs,
        available_actions=avail,
    )
    if len(obs) > 13000:
        obs = WEBSHOP_TEMPLATE_NO_HIS.format(
            task_description=slot.task,
            current_observation=slot.pre_text_obs,
            available_actions=avail,
        )
    return obs


def first_val_tasks(val_size: int, heldout: int, env_seed: int) -> list[tuple[int, int]]:
    """Same draw as the first ``val_envs.reset()`` in the training script."""
    rng = np.random.RandomState(env_seed)
    indices = rng.choice(np.arange(heldout), size=val_size, replace=False)
    return [(env_seed + i, int(indices[i])) for i in range(val_size)]


@dataclass
class Slot:
    worker_seed: int
    goal_idx: int
    task: str
    pre_text_obs: str
    info: dict
    history: list = field(default_factory=list)
    steps: list = field(default_factory=list)


def _complete(base_url: str, model: str, prompts: list[list[int]], args) -> list[str]:
    payload = {
        "model": model,
        "prompt": prompts,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "stop": ["<|im_end|>"],
    }
    response = requests.post(f"{base_url}/completions", json=payload, timeout=600)
    response.raise_for_status()
    data = response.json()
    choices = sorted(data["choices"], key=lambda item: item["index"])
    return [item["text"] for item in choices]


def main():
    args = _parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tasks = first_val_tasks(args.val_size, args.heldout, args.env_seed)
    with open(os.path.join(args.out_dir, "val_tasks.json"), "w", encoding="utf-8") as f:
        json.dump(
            [{"worker_seed": seed, "goal_idx": idx} for seed, idx in tasks],
            f,
            indent=2,
        )

    ray_tmp = "work/tmp/mimo_val"
    os.makedirs(ray_tmp, exist_ok=True)
    os.environ["RAY_TMPDIR"] = ray_tmp
    if not ray.is_initialized():
        ray.init(num_gpus=0, include_dashboard=False, ignore_reinit_error=True, _temp_dir=ray_tmp)
    file_path, attr_path = _catalog_paths()
    env_kwargs = {
        "observation_mode": "text",
        "num_products": None,
        "human_goals": False,
        "file_path": file_path,
        "attr_path": attr_path,
    }
    worker_cls = ray.remote(
        num_cpus=args.num_cpus_per_env,
        num_gpus=0,
        runtime_env={"env_vars": {"CUDA_VISIBLE_DEVICES": ""}},
    )(WebshopWorker)

    model = args.model_path
    records = []
    pbar = tqdm(total=len(tasks), desc="mimo val", unit="task")
    for start in range(0, len(tasks), args.concurrency):
        batch = tasks[start : start + args.concurrency]
        workers = [
            worker_cls.remote(seed, {**env_kwargs, "seed": seed}) for seed, _idx in batch
        ]
        resets = ray.get(
            [worker.reset.remote(idx) for worker, (_seed, idx) in zip(workers, batch)]
        )
        slots = []
        for (seed, idx), (obs, info) in zip(batch, resets):
            task = _extract_task(obs)
            slots.append(
                Slot(
                    worker_seed=seed,
                    goal_idx=idx,
                    task=task,
                    pre_text_obs=_format_obs(task, obs),
                    info=info,
                )
            )
        active = list(range(len(slots)))
        while active:
            prompts = []
            for i in active:
                user = build_user_prompt(slots[i], args.history_length)
                ids = tokenizer.apply_chat_template(
                    [{"role": "user", "content": user}],
                    add_generation_prompt=True,
                    tokenize=True,
                )
                if len(ids) > args.max_prompt_length:
                    ids = ids[-args.max_prompt_length :]
                prompts.append(list(ids))
            responses = _complete(args.base_url, model, prompts, args)
            projected, valids = webshop_projection(list(responses))
            results = ray.get(
                [workers[i].step.remote(projected[k]) for k, i in enumerate(active)]
            )
            still = []
            for k, i in enumerate(active):
                slot = slots[i]
                obs, reward, episode_done, info = results[k]
                episode_done = bool(episode_done)
                slot.steps.append(
                    {
                        "turn": len(slot.steps),
                        "observation": slot.pre_text_obs,
                        "response": responses[k],
                        "action": projected[k],
                        "is_action_valid": bool(valids[k]),
                        "reward": float(reward),
                        "done": episode_done,
                        "won": bool(info.get("won", False)),
                        "task_score": float(info.get("task_score", 0.0) or 0.0),
                    }
                )
                slot.history.append({"text_obs": slot.pre_text_obs, "action": projected[k]})
                if episode_done or len(slot.steps) >= args.max_steps:
                    continue
                slot.pre_text_obs = _format_obs(slot.task, obs)
                slot.info = info
                still.append(i)
            active = still
        for slot, worker in zip(slots, workers):
            won = any(step["won"] for step in slot.steps)
            task_score = slot.steps[-1]["task_score"] if slot.steps else 0.0
            records.append(
                {
                    "worker_seed": slot.worker_seed,
                    "goal_idx": slot.goal_idx,
                    "task": slot.task,
                    "won": won,
                    "n_steps": len(slot.steps),
                    "task_score": task_score,
                    "steps": slot.steps,
                }
            )
            try:
                ray.get(worker.close.remote())
            except Exception:
                pass
            ray.kill(worker)
        pbar.update(len(batch))
        pbar.set_postfix(success=sum(row["won"] for row in records) / len(records))
    pbar.close()

    success_rate = float(np.mean([row["won"] for row in records])) if records else 0.0
    mean_score = float(np.mean([row["task_score"] for row in records])) if records else 0.0
    summary = {
        "model_path": args.model_path,
        "n": len(records),
        "success_rate": success_rate,
        "mean_task_score": mean_score,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_tokens": args.max_tokens,
        "max_steps": args.max_steps,
        "history_length": args.history_length,
        "env_seed": args.env_seed,
        "val_size": args.val_size,
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out_dir, "episodes.jsonl"), "w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    ray.shutdown()


if __name__ == "__main__":
    main()
