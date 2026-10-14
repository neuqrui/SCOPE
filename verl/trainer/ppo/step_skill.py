"""Key-step skill analysis, parallel to ``seed_ana``.

The analyzer returns up to five step_skills. Steps listed in one target_steps
entry share that skill, and OPSD inserts it only on those steps. An empty list
means the episode has no clear error and no step is distilled. Failed episodes
also see same-group successful actions, as hints rather than a script to copy.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import torch

def extract_actions(steps: list[dict]) -> list[str]:
    actions = []
    for step in steps:
        response = str(step.get("response") or "")
        match = re.search(r"<action>(.*?)</action>", response, flags=re.IGNORECASE | re.DOTALL)
        if match:
            actions.append(match.group(1).strip())
        else:
            actions.append(response.strip())
    return actions


ANALYZER_SYSTEM = (
    "You analyze agent episodes and extract concise, policy-facing skills for "
    "specific error steps. Return ONLY valid JSON."
)

SUCCESS_RULES = (
    "Analyze the following SUCCESSFUL trajectory. It may still contain avoidable mistakes."
)

FAILURE_RULES = (
    "Analyze the following FAILED trajectory. Successful references are hints; "
    "differences alone do not prove mistakes."
)

STEP_SKILLS_SCHEMA = """Return up to 5 step_skills. For each:
1. Identify clear error steps in target_steps; steps requiring the same correction share one skill.
2. Write one short policy-facing skill that extracts the correction into a reusable workflow: the core decision rule and action ordering that would avoid the mistake.
3. Write a one-sentence error_reason and provide supporting evidence_steps, which need not include target_steps.

Important constraints:
- Judge errors using information available at each target step; do not label reasonable exploration.
- Write skill as one short, natural instruction, not a retrospective explanation or a list of labeled sections.
- Generalize beyond this task without names, IDs, exact values, or step numbers; keep the advice concrete and applicable at every target step.
- Use existing step indices, sorted and unique; sort entries by first target step.
- If no clear error is supported, return {"step_skills":[]}.
- Return only the top-level field step_skills.

Return format (replace string placeholders and 0 with actual text and existing integer step indices):
{"step_skills":[{"target_steps":[0],"error_reason":"string","skill":"string","evidence_steps":[0]}]}
"""


def prompt_trajectory(steps: list[dict]) -> str:
    """SEED-style step text: Observation, Response, and Action valid."""
    blocks = []
    for step in steps:
        lines = [
            f"Step {step['step_index']}",
            f"Observation: {step.get('observation', '')}",
            f"Response: {step.get('response', '')}",
        ]
        if "action_valid" in step:
            lines.append(f"Action valid: {json.dumps(bool(step['action_valid']))}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def format_success_references(reference_lists: list[list[str]]) -> str:
    if not reference_lists:
        return ""
    blocks = []
    for index, actions in enumerate(reference_lists, start=1):
        lines = "\n".join(f"Action {step_no}: {action}" for step_no, action in enumerate(actions))
        blocks.append(f"Successful reference {index}:\n{lines or '(empty)'}")
    return "Successful reference trajectories (actions only):\n" + "\n\n".join(blocks)


def _episode_context(*, task_description: str, outcome: str, steps: list[dict]) -> str:
    return (
        f"Episode context:\n- Task description: {task_description or '(not available)'}\n"
        f"- episode_success: {outcome}\n- Interaction trajectory:\n"
        f"{prompt_trajectory(steps)}"
    )


def build_success_analysis_prompt(*, steps: list[dict], task_description: str) -> str:
    context = _episode_context(task_description=task_description, outcome="success", steps=steps)
    return SUCCESS_RULES + "\n" + STEP_SKILLS_SCHEMA + "\n\n" + context


def build_failure_analysis_prompt(
    *,
    steps: list[dict],
    task_description: str,
    reference_lists: list[list[str]],
) -> str:
    context = _episode_context(task_description=task_description, outcome="failure", steps=steps)
    references = format_success_references(reference_lists)
    if references:
        context += "\n\n" + references
    return FAILURE_RULES + "\n" + STEP_SKILLS_SCHEMA + "\n\n" + context


def build_success_judge_prompt(
    *,
    steps: list[dict],
    task_description: str,
    analysis_text: str,
) -> str:
    return f"""You score the analyst, not the shopping agent. Return ONLY valid JSON.

The shopping episode succeeded, and it may still contain avoidable mistakes. The analyst returns up to 5 step_skills. Steps in one target_steps list share one skill. Training inserts that skill only at those steps. An empty step_skills list is correct when no clear error is supported.

- 0: not JSON, a target step is missing from the trajectory, or a listed skill is empty.
- 0.25: skills exist, but they narrate the episode, name this task's exact values, or group steps that need different corrections.
- 0.5: each skill is a short instruction shared by its target steps, but some targets are reasonable exploration rather than errors.
- 0.75: the targets are real mistakes and the skills apply at every listed step, with extra wording the agent does not need.
- 1: every group is a clear error, one shared instruction, and an empty list when the episode has no clear error.

The reason is one sentence, at most 30 words. Do not add any other text.

Return format:
{{"reason": "string", "score": 0}}

Task description: {task_description or "(not available)"}
Interaction trajectory:
{prompt_trajectory(steps)}
Analyst output:
{analysis_text}
"""


def build_failure_judge_prompt(
    *,
    steps: list[dict],
    task_description: str,
    reference_lists: list[list[str]],
    analysis_text: str,
) -> str:
    references = format_success_references(reference_lists) or "(no successful trajectory in this group)"
    return f"""You score the analyst, not the shopping agent. Return ONLY valid JSON.

The shopping episode failed. Successful references are hints; a difference alone is not a mistake. The analyst returns up to 5 step_skills. Steps in one target_steps list share one skill. Training inserts that skill only at those steps. An empty step_skills list is correct when no clear error is supported.

- 0: not JSON, a target step is missing from the trajectory, or a listed skill is empty.
- 0.25: skills exist, but they only restate how the success differs, narrate the episode, or group steps that need different corrections.
- 0.5: each skill is a short instruction shared by its target steps, but a target is not a clear error given the information at that step.
- 0.75: the targets are real mistakes and the skills apply at every listed step, with extra wording the agent does not need.
- 1: every group is a clear error, one shared instruction, and an empty list when the episode has no clear error.

The reason is one sentence, at most 30 words. Do not add any other text.

Return format:
{{"reason": "string", "score": 0}}

Task description: {task_description or "(not available)"}
{references}
Interaction trajectory:
{prompt_trajectory(steps)}
Analyst output:
{analysis_text}
"""


def _load_json_object(text: str) -> tuple[dict | None, str | None]:
    raw = str(text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.DOTALL)
    candidate = fenced.group(1) if fenced else raw
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end <= start:
        return None, "no json object"
    try:
        parsed = json.loads(candidate[start : end + 1])
    except json.JSONDecodeError as exc:
        return None, str(exc)
    if not isinstance(parsed, dict):
        return None, "json is not an object"
    return parsed, None


def _optional_step(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        step = int(value)
    except (TypeError, ValueError):
        return None
    if step < 0:
        return None
    return step


def parse_step_skill_analysis(text: str, valid_steps: set[int]) -> dict[str, Any]:
    """Keep groups whose target steps exist. Steps in one group share one skill."""
    parsed, error = _load_json_object(text)
    if parsed is None:
        return {"step_skills": [], "parse_error": error, "raw": text}
    raw_skills = parsed.get("step_skills")
    if raw_skills is None:
        return {"step_skills": [], "parse_error": error or "missing step_skills", "raw": text}
    if not isinstance(raw_skills, list):
        return {"step_skills": [], "parse_error": "step_skills must be a list", "raw": text}
    cleaned = []
    used: set[int] = set()
    for item in raw_skills[:5]:
        if not isinstance(item, dict):
            continue
        targets = item.get("target_steps")
        if not isinstance(targets, list):
            continue
        target_steps = []
        for value in targets:
            step = _optional_step(value)
            if step is None or step not in valid_steps or step in target_steps or step in used:
                continue
            target_steps.append(step)
        skill = str(item.get("skill") or "").strip()
        if not target_steps or not skill:
            continue
        evidence = []
        for value in item.get("evidence_steps") or []:
            step = _optional_step(value)
            if step is not None and step not in evidence:
                evidence.append(step)
        target_steps = sorted(target_steps)
        used.update(target_steps)
        cleaned.append(
            {
                "target_steps": target_steps,
                "error_reason": str(item.get("error_reason") or "").strip(),
                "skill": skill,
                "evidence_steps": evidence,
            }
        )
    cleaned.sort(key=lambda item: item["target_steps"][0])
    if error is None and raw_skills and not cleaned:
        error = "no valid step skill"
    return {"step_skills": cleaned, "parse_error": error, "raw": text}


def skill_for_step(analysis: dict, step_index: int) -> str:
    for item in analysis.get("step_skills") or []:
        if step_index in set(item.get("target_steps") or []):
            return str(item.get("skill") or "").strip()
    return ""


def parse_judge_score(text: str) -> tuple[float, str, str | None]:
    parsed, error = _load_json_object(text)
    if parsed is None:
        return 0.0, "", error
    reason = str(parsed.get("reason") or "").strip()
    try:
        score = float(parsed.get("score"))
    except (TypeError, ValueError):
        return 0.0, reason, "missing score"
    score = max(0.0, min(1.0, score))
    levels = (0.0, 0.25, 0.5, 0.75, 1.0)
    score = min(levels, key=lambda level: abs(level - score))
    return score, reason, None


def complete_openai_compatible(
    prompts: list[str],
    *,
    model: str | None,
    base_url: str | None,
    api_key: str | None,
    max_tokens: int,
    temperature: float = 0.0,
    top_p: float | None = None,
    system: str | None = None,
) -> list[str]:
    """Call an OpenAI-compatible chat endpoint, including a local vLLM server."""
    if not prompts:
        return []
    from openai import OpenAI

    resolved_key = api_key or os.environ.get("JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY") or "EMPTY"
    resolved_url = base_url or os.environ.get("JUDGE_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    resolved_model = model or os.environ.get("JUDGE_MODEL") or os.environ.get("OPENAI_MODEL")
    if not resolved_url:
        raise ValueError("step_skill judge requires judge_base_url or JUDGE_BASE_URL")
    if not resolved_model:
        raise ValueError("step_skill judge requires judge_model or JUDGE_MODEL")
    client = OpenAI(api_key=resolved_key, base_url=resolved_url)
    texts = []
    for prompt in prompts:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        request = {
            "model": resolved_model,
            "temperature": temperature,
            "max_tokens": int(max_tokens),
            "messages": messages,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        }
        if top_p is not None:
            request["top_p"] = top_p
        response = client.chat.completions.create(**request)
        texts.append(response.choices[0].message.content or "")
    return texts


def category_advantages(labels: list[str], group_ids: list[Any], scores: list[float]) -> list[float]:
    """GRPO advantages inside success and inside failure.

    A category with a single trajectory uses the whole group's mean and std
    instead of that category's own mean.
    """
    if not (len(labels) == len(group_ids) == len(scores)):
        raise ValueError("labels, group_ids, and scores must have the same length")
    advantages = [0.0] * len(scores)
    groups: dict[Any, list[int]] = {}
    for index, group_id in enumerate(group_ids):
        groups.setdefault(group_id, []).append(index)
    for indexes in groups.values():
        group_scores = [scores[index] for index in indexes]
        group_mean, group_std = _mean_std(group_scores)
        for label in ("success", "failure"):
            category = [index for index in indexes if labels[index] == label]
            if not category:
                continue
            if len(category) == 1:
                index = category[0]
                if len(indexes) == 1:
                    advantages[index] = 0.0
                    continue
                advantages[index] = _scaled_advantage(scores[index], group_mean, group_std)
                continue
            cat_scores = [scores[index] for index in category]
            cat_mean, cat_std = _mean_std(cat_scores)
            for index in category:
                advantages[index] = _scaled_advantage(scores[index], cat_mean, cat_std)
    return advantages


def _mean_std(values: list[float]) -> tuple[float, float]:
    tensor = torch.tensor(values, dtype=torch.float32)
    if tensor.numel() <= 1:
        return float(tensor.mean()) if tensor.numel() else 0.0, 1.0
    return float(tensor.mean()), float(tensor.std(unbiased=True))


def _scaled_advantage(score: float, mean: float, std: float) -> float:
    if std < 1e-6:
        return 0.0
    return (score - mean) / (std + 1e-6)


def insert_privilege_text(*, observation: str, privilege: str) -> str:
    """Insert privileged text after the task line, matching the SEED placement."""
    body = str(privilege or "").strip()
    prompt_text = str(observation or "")
    if not body:
        return prompt_text
    task_match = re.search(r"^.*\bYour task is to:\s*.*(?:\n|$)", prompt_text, flags=re.MULTILINE)
    if task_match is not None:
        prefix = prompt_text[: task_match.end()].rstrip()
        suffix = prompt_text[task_match.end() :].lstrip()
        if prefix and suffix:
            return f"{prefix}\n\n{body}\n\n{suffix}"
        if prefix:
            return f"{prefix}\n\n{body}"
        return f"{body}\n\n{suffix}"
    anchor = "Now it's your turn to"
    anchor_idx = prompt_text.find(anchor)
    if anchor_idx == -1:
        anchor = "Now it's your turn"
        anchor_idx = prompt_text.find(anchor)
    if anchor_idx == -1:
        return f"{prompt_text.rstrip()}\n\n{body}" if prompt_text else body
    prefix = prompt_text[:anchor_idx].rstrip()
    suffix = prompt_text[anchor_idx:].lstrip()
    if prefix and suffix:
        return f"{prefix}\n\n{body}\n\n{suffix}"
    if prefix:
        return f"{prefix}\n\n{body}"
    return f"{body}\n\n{suffix}"


def privilege_for_step(analysis: dict, step_index: int) -> str:
    """OPSD privilege for one decision. Steps in the same target list share it."""
    skill = skill_for_step(analysis, step_index)
    if not skill:
        return ""
    return (
        "**Step Skill**\n"
        "Use this skill for the current decision. Other steps in the same "
        "correction share this instruction:\n"
        f"[{skill}]"
    )
