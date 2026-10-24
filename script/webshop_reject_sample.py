"""Rejection sampling on the WebShop tasks used by training.

Train split matches ``WebshopMultiProcessEnv`` with the training defaults:
``env.webshop.use_small=true``, ``human_goals=false``, ``env.seed=0``.
Goals are shuffled once with that seed; indices ``[500, n_goals)`` are the
train pool (indices ``[0, 500)`` are the held-out prefix). Prompts use the
same history window (2) and step cap (15) as ``examples/opsd_kl_trainer/run_webshop_3b_grpo.sh``.

Each task is sampled ``--samples-per-round`` times (default 4). Among successes
(env ``won``, i.e. raw WebShop reward 1.0), the trajectory with the fewest
steps is kept. If none succeed, the task is sampled another round. Tasks that
fail every attempt are written to ``failed_goals.jsonl``.

``--draw-steps N`` replays the training env RNG: N steps, each drawing
``--env-num`` indices from ``[train_start, n_goals)`` with ``RandomState(base_seed)``.
Index ``g`` is the goal seen by worker group ``g``, whose WebShop seed is
``base_seed + g`` (goals are shuffled per seed). Duplicate draws collapse to
one task.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field

from tqdm import tqdm

import ray
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from agent_system.environments.env_package.webshop.envs import WebshopWorker
from agent_system.environments.env_package.webshop.projection import webshop_projection
from agent_system.environments.prompts.webshop import WEBSHOP_TEMPLATE, WEBSHOP_TEMPLATE_NO_HIS


def _parse_args():
    parser = argparse.ArgumentParser(description="WebShop rejection sampling with vLLM")
    parser.add_argument("--model-path", default="work/models/Qwen3-32B")
    parser.add_argument("--out-dir", default="work/reject_sample/webshop_qwen3_32b")
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-prompt-length", type=int, default=4096)
    parser.add_argument("--max-model-len", type=int, default=4608)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-num-seqs", type=int, default=0, help="0 = concurrency")
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--samples-per-round", type=int, default=4)
    parser.add_argument("--max-rounds", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--history-length", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-start", type=int, default=500)
    parser.add_argument("--limit", type=int, default=0, help="Only the first N train goals. 0 = all.")
    parser.add_argument(
        "--draw-steps",
        type=int,
        default=0,
        help="Replay this many training resets and reject-sample the unique tasks. 0 = every train index.",
    )
    parser.add_argument("--env-num", type=int, default=16, help="Goals drawn per training step.")
    parser.add_argument("--base-seed", type=int, default=0, help="Trainer env.seed. Group g uses base_seed + g.")
    parser.add_argument("--num-cpus-per-env", type=float, default=0.1)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
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
    for key in avail.keys():
        if key not in ["has_search_bar", "clickables"]:
            raise ValueError(f"Unknown key in available actions: {key}")
    if avail["has_search_bar"]:
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
        obs = WEBSHOP_TEMPLATE_NO_HIS.format(
            task_description=slot.task,
            current_observation=slot.pre_text_obs,
            available_actions=avail,
        )
    else:
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


@dataclass
class Slot:
    task_id: str
    goal_idx: int
    attempt: int
    task: str
    pre_text_obs: str
    info: dict
    history: list = field(default_factory=list)
    steps: list = field(default_factory=list)


@dataclass
class GoalState:
    task_id: str
    goal_idx: int
    env_seed: int
    finished: int = 0
    launched: int = 0
    successes: list = field(default_factory=list)
    attempt_summaries: list = field(default_factory=list)
    task: str = ""


def _load_done(path: str) -> set[str]:
    """Task ids already written. A truncated last line is ignored so a killed run can resume."""
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                done.add(str(row["task_id"] if "task_id" in row else row["goal_idx"]))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return done


def _append(path, record):
    with open(path, "a", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
        fcntl.flock(f, fcntl.LOCK_UN)


def _prompt_ids(tokenizer, user_text: str, max_prompt_length: int) -> tuple[list[int], bool]:
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        add_generation_prompt=True,
        tokenize=True,
    )
    truncated = len(ids) > max_prompt_length
    if truncated:
        ids = ids[-max_prompt_length:]
    return list(ids), truncated


def _finish_goal(state: GoalState, args, success_path: str, fail_path: str) -> str:
    """Close a goal only after a full round of samples has finished."""
    if state.finished % args.samples_per_round != 0:
        return "pending"
    if state.successes:
        best = min(state.successes, key=lambda traj: (traj["n_steps"], traj["attempt"]))
        _append(
            success_path,
            {
                "task_id": state.task_id,
                "env_seed": state.env_seed,
                "goal_idx": state.goal_idx,
                "task": best["task"],
                "won": True,
                "n_steps": best["n_steps"],
                "task_score": best["task_score"],
                "chosen_attempt": best["attempt"],
                "n_attempts": state.finished,
                "n_successes": len(state.successes),
                "success_step_counts": [traj["n_steps"] for traj in state.successes],
                "steps": best["steps"],
            },
        )
        return "success"
    if state.finished >= args.samples_per_round * args.max_rounds:
        _append(
            fail_path,
            {
                "task_id": state.task_id,
                "env_seed": state.env_seed,
                "goal_idx": state.goal_idx,
                "task": state.task,
                "won": False,
                "n_attempts": state.finished,
                "attempts": state.attempt_summaries,
            },
        )
        return "fail"
    return "continue"


def _select_tasks(n_goals: int, args) -> list[dict]:
    """Train indices, or the unique tasks actually drawn across ``draw_steps`` resets."""
    import numpy as np

    if args.draw_steps <= 0:
        ids = list(range(args.train_start, n_goals))
        if args.limit > 0:
            ids = ids[: args.limit]
        return [{"task_id": str(idx), "env_seed": args.seed, "goal_idx": idx} for idx in ids]

    pool = np.arange(args.train_start, n_goals)
    if args.env_num > len(pool):
        raise ValueError(f"env_num={args.env_num} exceeds train pool {len(pool)}")
    rng = np.random.RandomState(args.base_seed)
    seen: dict[str, dict] = {}
    for _ in range(args.draw_steps):
        chosen = rng.choice(pool, size=args.env_num, replace=False)
        for group, goal_idx in enumerate(int(value) for value in chosen.tolist()):
            env_seed = args.base_seed + group
            task_id = f"{env_seed}:{goal_idx}"
            seen.setdefault(
                task_id,
                {"task_id": task_id, "env_seed": env_seed, "goal_idx": goal_idx, "group": group},
            )
    tasks = [seen[key] for key in sorted(seen)]
    if args.limit > 0:
        tasks = tasks[: args.limit]
    return tasks


def main():
    args = _parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    success_path = os.path.join(args.out_dir, "success.jsonl")
    fail_path = os.path.join(args.out_dir, "failed_goals.jsonl")
    if args.max_num_seqs <= 0:
        args.max_num_seqs = args.concurrency

    print(f"loading tokenizer and vLLM from {args.model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        distributed_executor_backend="mp",
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        enable_chunked_prefill=True,
        seed=args.seed,
    )
    stop_ids = [tokenizer.eos_token_id]
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end, int) and im_end >= 0 and im_end != tokenizer.eos_token_id:
        stop_ids.append(im_end)
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        stop_token_ids=stop_ids,
    )

    if not ray.is_initialized():
        # AF_UNIX socket paths must stay under 108 bytes. The output dir is too long.
        ray_tmp = f"work/tmp/rs{args.shard_id}"
        os.makedirs(ray_tmp, exist_ok=True)
        os.environ["RAY_TMPDIR"] = ray_tmp
        ray.init(
            num_gpus=0,
            include_dashboard=False,
            ignore_reinit_error=True,
            _temp_dir=ray_tmp,
        )
    file_path, attr_path = _catalog_paths()
    env_kwargs = {
        "observation_mode": "text",
        "num_products": None,
        "human_goals": False,
        "file_path": file_path,
        "attr_path": attr_path,
        "seed": args.seed,
    }
    worker_cls = ray.remote(
        num_cpus=args.num_cpus_per_env,
        num_gpus=0,
        runtime_env={"env_vars": {"CUDA_VISIBLE_DEVICES": ""}},
    )(WebshopWorker)
    probe = worker_cls.remote(args.base_seed if args.draw_steps > 0 else args.seed, dict(env_kwargs))
    n_goals = len(ray.get(probe.get_goals.remote()))
    ray.get(probe.close.remote())
    ray.kill(probe)
    tasks = _select_tasks(n_goals, args)
    task_set = {task["task_id"] for task in tasks}
    done_success = _load_done(success_path) & task_set
    done_fail = (_load_done(fail_path) & task_set) - done_success
    finished_ids = done_success | done_fail
    shard_tasks = tasks[args.shard_id :: args.num_shards]
    pending_tasks = [task for task in shard_tasks if task["task_id"] not in finished_ids]
    shard_ids = [task["task_id"] for task in shard_tasks]
    train_ids = [task["task_id"] for task in pending_tasks]
    manifest = {
        "model_path": args.model_path,
        "catalog": "items_shuffle_1000",
        "human_goals": False,
        "shuffle_seed": args.seed,
        "base_seed": args.base_seed,
        "draw_steps": args.draw_steps,
        "env_num": args.env_num,
        "train_index_range": [args.train_start, n_goals],
        "n_goals": n_goals,
        "n_tasks": len(tasks),
        "n_pending": len(train_ids),
        "n_already_done": len(finished_ids),
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "samples_per_round": args.samples_per_round,
        "max_rounds": args.max_rounds,
        "max_steps": args.max_steps,
        "history_length": args.history_length,
        "temperature": args.temperature,
        "tensor_parallel_size": args.tensor_parallel_size,
        "concurrency": args.concurrency,
        "max_prompt_length": args.max_prompt_length,
        "max_tokens": args.max_tokens,
    }
    manifest_name = "manifest.json" if args.num_shards == 1 else f"manifest_shard{args.shard_id}.json"
    with open(os.path.join(args.out_dir, manifest_name), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    shard_done = len(shard_ids) - len(train_ids)
    print(
        f"shard={args.shard_id}/{args.num_shards} goals={n_goals} "
        f"shard_total={len(shard_ids)} pending={len(train_ids)} already_done={shard_done}",
        flush=True,
    )
    progress_path = os.path.join(args.out_dir, f"progress_shard{args.shard_id}.json")
    pbar = tqdm(
        total=len(shard_ids),
        initial=shard_done,
        desc=f"shard {args.shard_id}",
        unit="goal",
        position=args.shard_id,
        dynamic_ncols=True,
        mininterval=0.5,
        file=sys.stdout,
        disable=False,
    )

    def _report(n_success, n_fail):
        finished = shard_done + n_success + n_fail
        shard_ok = sum(1 for idx in shard_ids if idx in done_success) + n_success
        shard_bad = sum(1 for idx in shard_ids if idx in done_fail) + n_fail
        n_ok = shard_ok
        n_bad = shard_bad
        pbar.set_postfix(success=n_ok, fail=n_bad, refresh=False)
        pbar.refresh()
        payload = {
            "finished": finished,
            "total": len(shard_ids),
            "success": n_ok,
            "fail": n_bad,
            "resumed": shard_done,
            "pending": len(shard_ids) - finished,
            "shard_id": args.shard_id,
        }
        with open(progress_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
            f.write("\n")
        print(
            f"[shard {args.shard_id} {finished}/{len(shard_ids)}] success={n_ok} fail={n_bad} pending={payload['pending']}",
            flush=True,
        )

    by_seed: dict[int, list[dict]] = {}
    for task in pending_tasks:
        by_seed.setdefault(task["env_seed"], []).append(task)
    states = {
        task["task_id"]: GoalState(task_id=task["task_id"], goal_idx=task["goal_idx"], env_seed=task["env_seed"])
        for task in pending_tasks
    }
    n_success = 0
    n_fail = 0
    started = time.time()
    _report(0, 0)
    workers: list = []

    def _close_workers():
        for worker in workers:
            try:
                ray.get(worker.close.remote())
            except Exception:
                pass
            try:
                ray.kill(worker)
            except Exception:
                pass
        workers.clear()

    for env_seed in sorted(by_seed):
        seed_tasks = by_seed[env_seed]
        print(
            f"shard={args.shard_id} env_seed={env_seed} tasks={len(seed_tasks)} starting workers",
            flush=True,
        )
        workers.extend(worker_cls.remote(env_seed, dict(env_kwargs)) for _ in range(args.concurrency))
        queue = deque()
        for task in seed_tasks:
            queue.extend([task["task_id"]] * args.samples_per_round)
        slots: list[Slot | None] = [None] * args.concurrency

        def assign_free():
            pending = []
            for i, slot in enumerate(slots):
                if slot is not None or not queue:
                    continue
                task_id = queue.popleft()
                state = states[task_id]
                state.launched += 1
                pending.append((i, task_id, state.goal_idx, state.launched))
            if not pending:
                return
            futures = [workers[i].reset.remote(int(goal_idx)) for i, _task_id, goal_idx, _attempt in pending]
            results = ray.get(futures)
            for (i, task_id, _goal_idx, attempt), (obs, info) in zip(pending, results):
                task = _extract_task(obs)
                slots[i] = Slot(
                    task_id=task_id,
                    goal_idx=states[task_id].goal_idx,
                    attempt=attempt,
                    task=task,
                    pre_text_obs=_format_obs(task, obs),
                    info=info,
                )
                if not states[task_id].task:
                    states[task_id].task = task

        assign_free()
        while any(slot is not None for slot in slots):
            active = [i for i, slot in enumerate(slots) if slot is not None]
            user_prompts = [build_user_prompt(slots[i], args.history_length) for i in active]
            id_batch = []
            truncated = []
            for text in user_prompts:
                ids, was_truncated = _prompt_ids(tokenizer, text, args.max_prompt_length)
                id_batch.append(ids)
                truncated.append(was_truncated)
            outputs = llm.generate(prompt_token_ids=id_batch, sampling_params=sampling, use_tqdm=False)
            responses = [out.outputs[0].text for out in outputs]
            projected, valids = webshop_projection(list(responses))

            futures = [workers[i].step.remote(projected[k]) for k, i in enumerate(active)]
            results = ray.get(futures)
            finished_slots = []
            for k, i in enumerate(active):
                slot = slots[i]
                obs, reward, episode_done, info = results[k]
                reward = float(reward)
                episode_done = bool(episode_done)
                won = bool(info.get("won", False))
                slot.steps.append(
                    {
                        "turn": len(slot.steps),
                        "observation": slot.pre_text_obs,
                        "prompt": user_prompts[k],
                        "response": responses[k],
                        "action": projected[k],
                        "is_action_valid": bool(valids[k]),
                        "prompt_truncated": bool(truncated[k]),
                        "reward": reward,
                        "done": episode_done,
                        "won": won,
                        "task_score": float(info.get("task_score", 0.0) or 0.0),
                        "available_actions": slot.info.get("available_actions"),
                    }
                )
                slot.history.append({"text_obs": slot.pre_text_obs, "action": projected[k]})
                if episode_done or len(slot.steps) >= args.max_steps:
                    finished_slots.append(i)
                else:
                    slot.pre_text_obs = _format_obs(slot.task, obs)
                    slot.info = info

            for i in finished_slots:
                slot = slots[i]
                state = states[slot.task_id]
                won = any(step["won"] for step in slot.steps)
                task_score = slot.steps[-1]["task_score"] if slot.steps else 0.0
                summary = {
                    "attempt": slot.attempt,
                    "won": won,
                    "n_steps": len(slot.steps),
                    "task_score": task_score,
                }
                state.attempt_summaries.append(summary)
                state.finished += 1
                state.task = slot.task or state.task
                if won:
                    state.successes.append(
                        {
                            "attempt": slot.attempt,
                            "task": slot.task,
                            "n_steps": len(slot.steps),
                            "task_score": task_score,
                            "steps": slot.steps,
                        }
                    )
                status = _finish_goal(state, args, success_path, fail_path)
                if status == "continue":
                    queue.extend([slot.task_id] * args.samples_per_round)
                elif status == "success":
                    n_success += 1
                elif status == "fail":
                    n_fail += 1
                slots[i] = None
                if status in ("success", "fail"):
                    pbar.update(1)
                    _report(n_success, n_fail)
            assign_free()
        _close_workers()

    pbar.close()
    elapsed = time.time() - started
    summary = {
        "new_success": n_success,
        "new_fail": n_fail,
        "elapsed_sec": elapsed,
        "success_path": success_path,
        "fail_path": fail_path,
    }
    summary_name = "run_summary.json" if args.num_shards == 1 else f"run_summary_shard{args.shard_id}.json"
    with open(os.path.join(args.out_dir, summary_name), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    _close_workers()
    ray.shutdown()


if __name__ == "__main__":
    main()
