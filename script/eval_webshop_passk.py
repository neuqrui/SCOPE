"""WebShop pass@k / avg@k evaluation with vLLM.

The validation split matches training: seed 1000, 128 goals drawn without
replacement from the first 500 synthetic goals, history length 2, at most 15
steps, prompts capped at 4096 tokens and completions at 512. Temperature is
0.7. Each goal is rolled out 8 times.

avg@k is the mean success rate of the first k rollouts.
pass@k is the unbiased estimate from all 8 rollouts
(1 - C(n - c, k) / C(n, k)). Success means the WebShop score is exactly 1.
"""

from __future__ import annotations

import os

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import argparse
import json
import math
import os
import re
from functools import partial
from types import SimpleNamespace

import numpy as np
import ray
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


N_ROLLOUTS = 8
VAL_SIZE = 128
GOAL_POOL = 500
MAX_STEPS = 15
MAX_PROMPT_LENGTH = 4096
MAX_RESPONSE_LENGTH = 512
HISTORY_LENGTH = 2
TEMPERATURE = 0.7
VAL_SEED = 1000


def _norm_task(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def pass_at_k(n: int, c: int, k: int) -> float:
    if n < k:
        raise ValueError(f"Need at least {k} samples, got {n}")
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def summarize(records: list[dict], k_list: tuple[int, ...] = (1, 4, 8)) -> dict:
    metrics = {"n_tasks": len(records)}
    if not records:
        return metrics
    n = len(records[0]["success"])
    for k in k_list:
        if k == 1 or k in (4, 8):
            metrics[f"pass@{k}"] = float(np.mean([pass_at_k(n, int(sum(r["success"])), k) for r in records]))
        if k in (4, 8):
            metrics[f"avg@{k}"] = float(np.mean([np.mean(r["success"][:k]) for r in records]))
            metrics[f"task_score_avg@{k}"] = float(np.mean([np.mean(r["task_score"][:k]) for r in records]))
    return metrics


def load_failure_tasks(path: str) -> set[str]:
    tasks = set()
    with open(path) as handle:
        for line in handle:
            if not line.strip():
                continue
            episode = json.loads(line)
            if float(episode.get("episode_reward") or 0.0) > 0:
                continue
            tasks.add(_norm_task(episode.get("task_query")))
    return tasks


def _local_tensor(value):
    if isinstance(value, torch.Tensor) and type(value).__name__ == "DTensor":
        return value._local_tensor.detach().cpu()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    raise TypeError(f"Unsupported checkpoint value {type(value)}")


def _shard_dim(value) -> int | None:
    placements = getattr(value, "placements", None)
    if not placements:
        return None
    dims = [placement.dim for placement in placements if type(placement).__name__ == "Shard"]
    if not dims:
        return None
    return int(dims[0])


def merge_fsdp_checkpoint(actor_dir: str, output_dir: str, base_model: str) -> str:
    if os.path.isfile(os.path.join(output_dir, "config.json")) and (
        os.path.isfile(os.path.join(output_dir, "model.safetensors"))
        or os.path.isfile(os.path.join(output_dir, "model.safetensors.index.json"))
    ):
        print(f"Using merged weights at {output_dir}")
        return output_dir
    rank_paths = [
        os.path.join(actor_dir, "model_world_size_2_rank_0.pt"),
        os.path.join(actor_dir, "model_world_size_2_rank_1.pt"),
    ]
    shards = [torch.load(path, map_location="cpu", weights_only=False) for path in rank_paths]
    merged = {}
    for key in shards[0]:
        dim = _shard_dim(shards[0][key])
        locals_ = [_local_tensor(shard[key]) for shard in shards]
        if dim is None:
            merged[key] = locals_[0].to(torch.bfloat16)
        else:
            merged[key] = torch.cat(locals_, dim=dim).to(torch.bfloat16)
    os.makedirs(output_dir, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16)
    missing, unexpected = model.load_state_dict(merged, strict=False)
    missing = [key for key in missing if key != "lm_head.weight"]
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint merge mismatch missing={missing[:8]} unexpected={unexpected[:8]}")
    model.save_pretrained(output_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained(actor_dir).save_pretrained(output_dir)
    print(f"Merged FSDP checkpoint to {output_dir}")
    return output_dir


def build_envs():
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
    resources = {"num_cpus": 0.1, "num_gpus": 0}
    vector = build_webshop_envs(
        seed=VAL_SEED,
        env_num=VAL_SIZE,
        group_n=1,
        resources_per_worker=resources,
        is_train=False,
        env_kwargs=env_kwargs,
    )
    config = SimpleNamespace(env=SimpleNamespace(history_length=HISTORY_LENGTH))
    manager = WebshopEnvironmentManager(vector, partial(webshop_projection), config)
    rng = np.random.RandomState(VAL_SEED)
    indices = rng.choice(range(GOAL_POOL), size=VAL_SIZE, replace=False)
    return manager, [int(i) for i in indices]


def encode_prompts(tokenizer, texts: list[str]) -> tuple[list[list[int]], int]:
    truncated = 0
    token_ids = []
    for text in texts:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            add_generation_prompt=True,
            tokenize=False,
        )
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(ids) > MAX_PROMPT_LENGTH:
            ids = ids[-MAX_PROMPT_LENGTH:]
            truncated += 1
        token_ids.append(ids)
    return token_ids, truncated


def rollout(llm, tokenizer, manager, indices) -> list[dict]:
    from vllm import SamplingParams

    sampling = SamplingParams(
        temperature=TEMPERATURE,
        top_p=1.0,
        top_k=-1,
        max_tokens=MAX_RESPONSE_LENGTH,
    )
    obs, _infos = manager.reset(kwargs=None, indices=indices)
    tasks = list(manager.tasks)
    done = np.zeros(VAL_SIZE, dtype=bool)
    success = np.zeros(VAL_SIZE, dtype=np.float32)
    task_score = np.zeros(VAL_SIZE, dtype=np.float32)
    lengths = np.zeros(VAL_SIZE, dtype=np.int32)
    truncated = 0
    for _step in range(MAX_STEPS):
        active = np.flatnonzero(~done)
        if len(active) == 0:
            break
        prompts = [obs["text"][i] for i in active]
        token_ids, n_trunc = encode_prompts(tokenizer, prompts)
        truncated += n_trunc
        outputs = llm.generate(
            prompts=None,
            sampling_params=sampling,
            prompt_token_ids=token_ids,
            use_tqdm=False,
        )
        actions = [""] * VAL_SIZE
        for slot, output in zip(active, outputs):
            actions[slot] = output.outputs[0].text
        next_obs, rewards, dones, infos = manager.step(actions)
        rewards = np.asarray(rewards).reshape(-1)
        dones = np.asarray(dones).reshape(-1).astype(bool)
        for slot in active:
            lengths[slot] += 1
            if dones[slot] or bool(infos[slot].get("won")):
                done[slot] = True
                success[slot] = 1.0 if rewards[slot] >= 10.0 or bool(infos[slot].get("won")) else 0.0
                task_score[slot] = float(infos[slot].get("task_score") or 0.0)
            elif _step == MAX_STEPS - 1:
                done[slot] = True
        obs = next_obs
    if truncated:
        print(f"left-truncated {truncated} prompts to {MAX_PROMPT_LENGTH} tokens")
    return [
        {
            "task": tasks[i],
            "success": float(success[i]),
            "task_score": float(task_score[i]),
            "length": int(lengths[i]),
        }
        for i in range(VAL_SIZE)
    ]


def evaluate_model(model_path: str, manager, indices, failure_tasks: set[str], output_path: str):
    from vllm import LLM

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        dtype="bfloat16",
        gpu_memory_utilization=0.5,
        max_model_len=MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH,
        trust_remote_code=True,
        disable_log_stats=True,
    )
    per_rollout = [rollout(llm, tokenizer, manager, indices) for _ in range(N_ROLLOUTS)]
    del llm
    torch.cuda.empty_cache()

    records = []
    for i in range(VAL_SIZE):
        records.append(
            {
                "task": per_rollout[0][i]["task"],
                "goal_index": indices[i],
                "success": [per_rollout[r][i]["success"] for r in range(N_ROLLOUTS)],
                "task_score": [per_rollout[r][i]["task_score"] for r in range(N_ROLLOUTS)],
                "length": [per_rollout[r][i]["length"] for r in range(N_ROLLOUTS)],
            }
        )
    matched = {_norm_task(record["task"]) for record in records}
    subset = [record for record in records if _norm_task(record["task"]) in failure_tasks]
    missing = len(failure_tasks - matched)
    report = {
        "model": model_path,
        "temperature": TEMPERATURE,
        "n_rollouts": N_ROLLOUTS,
        "val_size": VAL_SIZE,
        "metrics": {
            "all": summarize(records),
            "prev_fail_subset": summarize(subset),
        },
        "prev_fail_subset_size": len(subset),
        "prev_fail_unmatched": missing,
        "records": records,
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as handle:
        json.dump(report, handle)
    print_report(model_path, report)
    return report


def print_report(model_path: str, report: dict) -> None:
    print(f"\n== {model_path}")
    for split in ("all", "prev_fail_subset"):
        metrics = report["metrics"][split]
        print(
            f"{split} n={metrics.get('n_tasks', 0)} "
            f"avg@4={metrics.get('avg@4', float('nan')):.4f} "
            f"avg@8={metrics.get('avg@8', float('nan')):.4f} "
            f"pass@1={metrics.get('pass@1', float('nan')):.4f} "
            f"pass@4={metrics.get('pass@4', float('nan')):.4f} "
            f"pass@8={metrics.get('pass@8', float('nan')):.4f} "
            f"task_score_avg@4={metrics.get('task_score_avg@4', float('nan')):.4f} "
            f"task_score_avg@8={metrics.get('task_score_avg@8', float('nan')):.4f}"
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True, help="HF model dirs, or an FSDP actor dir")
    parser.add_argument(
        "--failure-jsonl",
        default="work/ckpt/rollouts_val/grpo_qwen2.5_3b_step150_eval/episodes/step_000150_val/episodes.jsonl",
    )
    parser.add_argument("--output-dir", default="work/ckpt/rollouts_val/passk_t0.7")
    parser.add_argument("--merged-dir", default="work/ckpt/hf/grpo_qwen2.5_3b_step150")
    parser.add_argument("--base-model", default=os.environ.get("MODEL_PATH", "work/models/Qwen2.5-3B-Instruct"))
    return parser.parse_args()


def resolve_model(path: str, merged_dir: str, base_model: str) -> str:
    shard = os.path.join(path, "model_world_size_2_rank_0.pt")
    actor_shard = os.path.join(path, "actor", "model_world_size_2_rank_0.pt")
    if os.path.isfile(shard):
        return merge_fsdp_checkpoint(path, merged_dir, base_model)
    if os.path.isfile(actor_shard):
        return merge_fsdp_checkpoint(os.path.join(path, "actor"), merged_dir, base_model)
    return path


def main():
    args = parse_args()
    os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
    if not ray.is_initialized():
        ray.init(num_cpus=32, ignore_reinit_error=True, include_dashboard=False)
    failure_tasks = load_failure_tasks(args.failure_jsonl)
    print(f"Previous failure tasks: {len(failure_tasks)}")
    manager, indices = build_envs()
    try:
        for model in args.models:
            resolved = resolve_model(model, args.merged_dir, args.base_model)
            name = os.path.basename(resolved.rstrip("/"))
            output_path = os.path.join(args.output_dir, f"{name}.json")
            evaluate_model(resolved, manager, indices, failure_tasks, output_path)
    finally:
        manager.envs.close()


if __name__ == "__main__":
    main()
