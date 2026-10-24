"""Build success/failure trajectory pairs for the intervention analyzer.

Preference order for a failure's matched success:
1. Same training group (same uid) from a GRPO rollout dump.
2. Same task in a GRPO training dump.
3. Same task in the GRPO step-150 rejection-sample successes.

Each output row contains one failure trajectory and one success trajectory.
Observations and admissible actions are kept; chain-of-thought is dropped.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict


# Plain GRPO dumps only keep two episodes per step, so they cannot fill 500
# same-task pairs. Other WebShop training dumps are used only after those.
GRPO_EXPERIMENTS = (
    "grpo_qwen2.5_3b",
    "grpo_qwen2.5_3b_from_opd30",
)
EXTRA_TRAINING_EXPERIMENTS = (
    "grpo_opsd_qwen2.5_3b_coef0.01_beta0.0_skillallfalse",
    "grpo_opsd_kl_qwen2.5_3b_coef0.01_sampled_kl_skillallfalse",
    "grpo_opsd_kl_qwen2.5_3b_coef0.01_sampled_kl_skillallfalse_seed_ana",
    "grpo_opsd_kl_qwen2.5_3b_coef0.01_sampled_kl_skillallfalse_seed_ana_from_opd30",
    "grpo_opsd_kl_qwen2.5_3b_coef0.01_sampled_kl_step_skill_none",
    "grpo_opsd_kl_qwen2.5_3b_coef0.01_sampled_kl_step_skill_none_from_opd30",
    "grpo_opsd_kl_qwen2.5_3b_coef0.01_sampled_kl_step_skill_pg",
    "grpo_opsd_kl_qwen2.5_3b_coef0.01_sampled_kl_step_skill_sft",
    "grpo_opsd_kl_qwen2.5_3b_coef0.1_sampled_kl_skillallfalse_seed_ana",
    "opsd_qwen2.5_3b_coef0.01_beta5.0_skillallfalse",
)
REJECT_SUCCESS = "work/reject_sample/webshop_grpo_3b_step150/success.jsonl"


def norm_task(text: str) -> str:
    return " ".join(str(text or "").split()).lower()


def extract_action(step: dict) -> str:
    action = step.get("action")
    if action:
        return " ".join(str(action).split())
    response = str(step.get("response") or "")
    match = re.search(r"<action>(.*?)</action>", response, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return " ".join(match.group(1).split())
    return ""


def extract_admissible(step: dict) -> list[str]:
    raw = step.get("available_actions")
    if isinstance(raw, dict):
        actions = []
        if raw.get("has_search_bar"):
            actions.append("search[<your query>]")
        for item in raw.get("clickables") or []:
            actions.append(f"click[{item}]")
        return actions
    if isinstance(raw, list) and raw:
        return [str(item) for item in raw]
    prompt = str(step.get("prompt") or "")
    # The list ends at "]." ; inner click[...] brackets must not close it.
    match = re.search(
        r"admissible actions of the current situation are:\s*\[(.*?)\]\s*\.",
        prompt,
        flags=re.DOTALL,
    )
    if not match:
        return []
    return [item.strip() for item in re.findall(r"'([^']*)'", match.group(1)) if item.strip()]


def compact_steps(steps: list[dict]) -> list[dict]:
    compact = []
    for index, step in enumerate(steps):
        observation = str(step.get("observation") or "")
        if len(observation) > 1500:
            observation = observation[:1500] + " ...[truncated]"
        compact.append(
            {
                "step": int(step.get("turn", step.get("step", index))),
                "observation": observation,
                "admissible_actions": extract_admissible(step),
                "action": extract_action(step),
                "action_valid": bool(step.get("is_action_valid", True)),
            }
        )
    return compact


def load_training_episodes(rollout_root: str, experiments: tuple[str, ...]) -> list[dict]:
    episodes = []
    for experiment in experiments:
        episode_root = os.path.join(rollout_root, experiment, "episodes")
        if not os.path.isdir(episode_root):
            continue
        for step_name in sorted(os.listdir(episode_root)):
            path = os.path.join(episode_root, step_name, "episodes.jsonl")
            if not os.path.exists(path):
                continue
            global_step = int(step_name.split("_")[1])
            with open(path) as handle:
                for line in handle:
                    raw = json.loads(line)
                    reward = float(raw.get("episode_reward") or 0.0)
                    episodes.append(
                        {
                            "source": experiment,
                            "global_step": global_step,
                            "traj_uid": raw.get("traj_uid"),
                            "uid": raw.get("uid") or "",
                            "task": " ".join(str(raw.get("task_query") or "").split()),
                            "task_key": norm_task(raw.get("task_query")),
                            "success": reward > 0,
                            "episode_reward": reward,
                            "steps": compact_steps(raw.get("steps") or []),
                        }
                    )
    return episodes


def load_reject_successes(path: str) -> list[dict]:
    episodes = []
    with open(path) as handle:
        for line in handle:
            raw = json.loads(line)
            episodes.append(
                {
                    "source": "grpo_step150_reject_sample",
                    "global_step": 150,
                    "traj_uid": f"reject-{raw.get('task_id', raw.get('goal_idx'))}",
                    "uid": "",
                    "task": " ".join(str(raw.get("task") or "").split()),
                    "task_key": norm_task(raw.get("task")),
                    "success": True,
                    "episode_reward": 10.0 if raw.get("won") else 0.0,
                    "task_score": raw.get("task_score"),
                    "steps": compact_steps(raw.get("steps") or []),
                }
            )
    return episodes


def success_rank(failure: dict, success: dict) -> tuple:
    same_group = bool(failure.get("uid")) and failure.get("uid") == success.get("uid")
    same_step = (
        failure.get("source") == success.get("source")
        and failure.get("global_step") == success.get("global_step")
    )
    from_training = success.get("source") != "grpo_step150_reject_sample"
    return (same_group, same_step, from_training, -len(success.get("steps") or []))


def build_pairs(failures: list[dict], successes: list[dict], limit: int) -> list[dict]:
    by_task: dict[str, list[dict]] = defaultdict(list)
    for episode in successes:
        if episode.get("steps"):
            by_task[episode["task_key"]].append(episode)

    # One failure per task first, so the 500 are not dominated by repeated tasks.
    failures = [item for item in failures if item.get("steps") and by_task.get(item["task_key"])]
    def _priority(source: str) -> int:
        if source == "grpo_qwen2.5_3b":
            return 0
        if source == "grpo_qwen2.5_3b_from_opd30":
            return 1
        return 2

    failures.sort(key=lambda item: (_priority(item["source"]), item["global_step"], item["traj_uid"] or ""))
    seen_fail = set()
    unique = []
    for item in failures:
        key = item.get("traj_uid") or id(item)
        if key in seen_fail:
            continue
        seen_fail.add(key)
        unique.append(item)

    first = []
    rest = []
    seen_task = set()
    for item in unique:
        if item["task_key"] in seen_task:
            rest.append(item)
        else:
            seen_task.add(item["task_key"])
            first.append(item)
    ordered = first + rest

    pairs = []
    for failure in ordered:
        candidates = by_task[failure["task_key"]]
        success = max(candidates, key=lambda item: success_rank(failure, item))
        same_group = bool(failure.get("uid")) and failure.get("uid") == success.get("uid")
        pairs.append(
            {
                "pair_id": f"{failure.get('source')}-s{failure.get('global_step')}-{failure.get('traj_uid')}",
                "task": failure["task"],
                "same_group": same_group,
                "success_source": success["source"],
                "failure_source": failure["source"],
                "success": {
                    "source": success["source"],
                    "global_step": success["global_step"],
                    "traj_uid": success["traj_uid"],
                    "uid": success.get("uid") or "",
                    "episode_reward": success["episode_reward"],
                    "steps": success["steps"],
                },
                "failure": {
                    "source": failure["source"],
                    "global_step": failure["global_step"],
                    "traj_uid": failure["traj_uid"],
                    "uid": failure.get("uid") or "",
                    "episode_reward": failure["episode_reward"],
                    "steps": failure["steps"],
                },
            }
        )
        if len(pairs) >= limit:
            break
    return pairs


def episode_for_prompt(pair: dict, case: str) -> dict:
    chosen = pair[case]
    payload = {
        "case": case,
        "task": pair["task"],
        "final_result": "success" if case == "success" else "failure",
        "episode_reward": chosen["episode_reward"],
        "steps": chosen["steps"],
    }
    if case == "failure":
        payload["successful_reference_actions"] = [
            [step["action"] for step in pair["success"]["steps"] if step.get("action")]
        ]
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-root", default="work/ckpt/rollouts")
    parser.add_argument("--reject-success", default=REJECT_SUCCESS)
    parser.add_argument("--out-dir", default="work/analyzer_step_skill")
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args()

    primary = load_training_episodes(args.rollout_root, GRPO_EXPERIMENTS)
    primary_failures = [item for item in primary if not item["success"]]
    primary_successes = [item for item in primary if item["success"]]
    reject_successes = load_reject_successes(args.reject_success)
    pairs = build_pairs(primary_failures, primary_successes + reject_successes, args.limit)
    extra_failures = []
    extra_successes = []
    if len(pairs) < args.limit:
        extra = load_training_episodes(args.rollout_root, EXTRA_TRAINING_EXPERIMENTS)
        extra_failures = [item for item in extra if not item["success"]]
        extra_successes = [item for item in extra if item["success"]]
        used = {item["pair_id"] for item in pairs}
        more = build_pairs(
            extra_failures,
            primary_successes + extra_successes + reject_successes,
            args.limit - len(pairs),
        )
        for item in more:
            if item["pair_id"] not in used:
                pairs.append(item)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "pairs.jsonl")
    with open(out_path, "w") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair, ensure_ascii=False) + "\n")

    same_group = sum(1 for pair in pairs if pair["same_group"])
    reject_matched = sum(1 for pair in pairs if pair["success_source"] == "grpo_step150_reject_sample")
    train_matched = len(pairs) - reject_matched
    summary = {
        "grpo_episodes": len(primary),
        "grpo_failures": len(primary_failures),
        "grpo_successes": len(primary_successes),
        "extra_failures": len(extra_failures),
        "extra_successes": len(extra_successes),
        "reject_successes": len(reject_successes),
        "pairs": len(pairs),
        "same_group_pairs": same_group,
        "success_from_training": train_matched,
        "success_from_reject_sample": reject_matched,
        "unique_tasks": len({pair["task"] for pair in pairs}),
        "out_path": out_path,
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
