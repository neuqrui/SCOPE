"""Contrastive prompt for the frozen trajectory analyzer.

The analyzer sees one failed WebShop trajectory and the successful trajectories
of the same task. Successful trajectories are evidence, not a script to copy.
"""

from __future__ import annotations

import json
import os
import re


ANALYZER_INSTRUCTIONS = """You are a contrastive trajectory analyzer used at training time.

Input: one failed agent trajectory and zero or more verified successful trajectories for the SAME task. Each trajectory contains step-indexed observations, admissible actions, executed actions, subsequent observations, and the final outcome. Successful trajectories are evidence, not flawless demonstrations: they may contain detours, and different action sequences may both be valid.

Your job is narrow: identify failed-trajectory decisions for which a successful reference provides a concrete, state-relevant alternative. Then write one local, transferable skill for each selected decision. Prefer abstaining to unsupported causal claims.

Decision procedure:

1. Check each successful reference's task and outcome. Align trajectories by the task subgoal, the visible environment state, and the actions available BEFORE the decision. Never align solely by step number or action-string equality.
2. Find the earliest failed-trajectory step with a well-supported, actionable contrast. A different action is not by itself an error. If the same action succeeds in a comparable reference state, do not call that action the cause of failure without evidence explaining the difference; inspect later steps instead.
3. Starting at that step, select at most 5 distinct intervention points. Select fewer when the evidence is weak. Do not select several repeated actions merely to restate the same mistake, and do not select every step in the remaining trajectory.
4. For EACH selected step, provide: (a) a short excerpt of what the failed agent could see before acting; (b) the comparable reference state, action, and observed outcome; (c) why the reference supports a different decision here; (d) one alternative action copied EXACTLY from this failed step's admissible_actions; and (e) one unique, short skill for acting in this kind of state.
5. A skill must describe a reusable decision rule, not a product ID or a replay of the reference trajectory. It may use hindsight to formulate the rule, but the agent must be able to follow it using information available BEFORE the selected action. Do not claim an attribute was confirmed on a search-results page if it was visible only on a later product page.
6. If references disagree with one another, appear inconsistent with the task, lack a comparable state, or provide no executable alternative at the failed step, omit that candidate. If no candidate remains, abstain. A successful outcome alone is not proof that every action in its trajectory was correct.

Return ONLY valid JSON. Do not output a separate reasoning monologue. Keep the analysis concise and evidence-based. Step indices start at 0.

{
  "status": "ok or abstain",
  "trajectory_analysis": "brief task requirements, failed trajectory, and outcome",
  "reference_assessment": "which references are usable for comparison and any important conflict",
  "first_failure_step": null,
  "step_skills": [
    {
      "step": 0,
      "reference_id": "id of the supporting successful trajectory",
      "reference_step": 0,
      "failed_state_evidence": "short excerpt visible before the failed action",
      "reference_evidence": "comparable reference observation, action, and relevant subsequent result",
      "why_selected": "why this is an actionable decision error rather than merely a different path",
      "expected_action": "EXACT action string from this step's admissible_actions",
      "skill": "When [observable condition], [decision rule]; avoid [specific failure mode]."
    }
  ]
}

When status is "abstain", set first_failure_step to null and step_skills to [].
When status is "ok", first_failure_step must equal the smallest selected step.
Do not copy or rewrite the successful trajectories' action sequences in the output.

The failed trajectory is JSON. Each of its steps has the full observation, admissible_actions, the think text, and the executed action. expected_action must be copied from that step's admissible_actions. Each successful trajectory is a chronological transcript in this order: Observation, Think, Action, Observation, Think, Action. Observations and think text are complete, not shortened. It does not list admissible actions.
"""


def _text(value: object) -> str:
    return str(value or "")


def _failure_steps(steps: list[dict]) -> list[dict]:
    formatted = []
    for index, step in enumerate(steps):
        formatted.append(
            {
                "step": int(step.get("step", index)),
                "observation": _text(step.get("observation")),
                "admissible_actions": step.get("admissible_actions") or [],
                "think": _text(step.get("think")),
                "action": _text(step.get("action")),
            }
        )
    return formatted


def format_success_transcript(steps: list[dict]) -> str:
    """Chronological observation / think / action transcript, without truncation."""
    lines = []
    for index, step in enumerate(steps):
        step_no = int(step.get("step", index))
        lines.append(f"Step {step_no}")
        lines.append(f"Observation: {_text(step.get('observation'))}")
        lines.append(f"Think: {_text(step.get('think'))}")
        lines.append(f"Action: {_text(step.get('action'))}")
    return "\n".join(lines)


def build_paired_episode(pair: dict) -> dict:
    failure = pair["failure"]
    success = pair["success"]
    return {
        "task": pair["task"],
        "failed_trajectory": {
            "id": str(failure.get("traj_uid") or pair.get("pair_id")),
            "final_outcome": "failure",
            "episode_reward": failure.get("episode_reward"),
            "steps": _failure_steps(failure.get("steps") or []),
        },
        "successful_trajectories": [
            {
                "id": str(success.get("traj_uid") or "success"),
                "final_outcome": "success",
                "episode_reward": success.get("episode_reward"),
                "trajectory": format_success_transcript(success.get("steps") or []),
            }
        ],
    }


def _think_text(response: str) -> str:
    match = re.search(r"<think>(.*?)</think>", response or "", flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def _full_steps(raw_steps: list[dict]) -> list[dict]:
    from script.build_intervention_pairs import extract_action, extract_admissible

    steps = []
    for index, step in enumerate(raw_steps):
        steps.append(
            {
                "step": int(step.get("turn", step.get("step", index))),
                "observation": str(step.get("observation") or ""),
                "admissible_actions": extract_admissible(step),
                "think": _think_text(str(step.get("response") or "")),
                "action": extract_action(step),
            }
        )
    return steps


def _load_training_episode(source: str, global_step: int, traj_uid: str) -> dict:
    path = os.path.join(
        "work/ckpt/rollouts",
        source,
        "episodes",
        f"step_{int(global_step):06d}_train",
        "episodes.jsonl",
    )
    with open(path) as handle:
        for line in handle:
            episode = json.loads(line)
            if episode.get("traj_uid") == traj_uid:
                return episode
    raise KeyError(f"missing trajectory {traj_uid} in {path}")


def _load_reject_episode(traj_uid: str) -> dict:
    task_id = str(traj_uid).split("reject-", 1)[-1]
    path = "work/reject_sample/webshop_grpo_3b_step150/success.jsonl"
    with open(path) as handle:
        for line in handle:
            episode = json.loads(line)
            if str(episode.get("task_id")) == task_id:
                return episode
    raise KeyError(f"missing reject trajectory {traj_uid}")


def with_full_context(pair: dict) -> dict:
    """Replace stored steps with the original observation and think text."""
    enriched = json.loads(json.dumps(pair))
    for side_name in ("failure", "success"):
        side = enriched[side_name]
        if side.get("source") == "grpo_step150_reject_sample":
            raw = _load_reject_episode(side.get("traj_uid"))
        else:
            raw = _load_training_episode(side["source"], side["global_step"], side["traj_uid"])
        side["steps"] = _full_steps(raw.get("steps") or [])
    return enriched


def build_analyzer_prompt(episode: dict) -> str:
    payload = json.dumps(episode, ensure_ascii=False, indent=2)
    return f"{ANALYZER_INSTRUCTIONS}\nInput:\n{payload}\n"
