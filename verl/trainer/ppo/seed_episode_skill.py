"""SEED episode-only skill generation for the SDAR teacher.

The analysis prompt matches ``SEEDEpisodeAnalyzer`` with
``skill_mode=episode_only`` and ``include_episode_summary=True``.
The teacher observation is the SEED augmented prompt: the episode skill is
inserted after the task line and before "Now it's your turn".
"""

from __future__ import annotations

import json
import re
from typing import Any


def format_episode_steps(steps: list[dict]) -> str:
    lines = []
    for step in steps:
        action_valid = ""
        if "action_valid" in step:
            action_valid = f"\nAction valid: {str(bool(step['action_valid'])).lower()}"
        lines.append(
            f"Step {step['step_index']}\n"
            f"Observation: {step['observation']}\n"
            f"Response: {step['response']}{action_valid}\n"
        )
    return "".join(lines)


def build_episode_analysis_prompt(
    *,
    steps: list[dict],
    task_description: str,
    episode_success: float | None,
) -> str:
    if episode_success is None:
        outcome_label = "unknown"
    else:
        outcome_label = "success" if float(episode_success) >= 1.0 else "failure"

    if outcome_label == "success":
        episode_skill_instruction = (
            "Write one episode_skill that extracts the successful trajectory into workflow: "
            "the core decision rule and action ordering that made this trajectory work. "
        )
    elif outcome_label == "failure":
        episode_skill_instruction = (
            "Write one episode_skill that extracts the failed trajectory into avoidance rules: "
            "the core mistake and warning signs that agent should avoid. "
        )
    else:
        episode_skill_instruction = (
            "Write one episode_skill distilled from the trajectory. If it succeeded, extract "
            "the successful workflow; if it failed, extract the avoidance rules. "
        )

    return f"""Analyze the following agent episode and return ONLY valid JSON.

You need to complete two fields:
1. Write a concise episode_summary.
2. {episode_skill_instruction}

Important constraints:
- Write episode_skill as one short policy-facing skill, not as retrospective explanation of the trajectory.
- Return only these top-level fields: episode_summary, episode_skill.

Return format:
{{
  "episode_summary": "string",
  "episode_skill": "string"
}}

Episode context:
- Task description: {task_description or "(not available)"}
- episode_success: {outcome_label}
- Interaction trajectory: {format_episode_steps(steps)}
"""


def parse_analysis_json(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.DOTALL)
    candidate = fenced.group(1) if fenced else raw
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end <= start:
        return {"episode_summary": "", "episode_skill": "", "parse_error": "no json object", "raw": raw}
    try:
        parsed = json.loads(candidate[start : end + 1])
    except json.JSONDecodeError as exc:
        return {"episode_summary": "", "episode_skill": "", "parse_error": str(exc), "raw": raw}
    summary = str(parsed.get("episode_summary", ""))
    skill = str(parsed.get("episode_skill", "")).strip()
    error = None if skill else "empty episode_skill"
    return {
        "episode_summary": summary,
        "episode_skill": skill,
        "parse_error": error,
        "raw": raw,
    }


def extract_task_description(observation: str) -> str:
    match = re.search(r"Your task is to:\s*(.+)", observation or "")
    if match is None:
        return ""
    return match.group(1).strip()


def build_augmented_observation_text(*, observation: str, episode_skill: str) -> str:
    """Insert the episode skill the way ``seed.prompting.build_augmented_observation_text`` does."""
    body = str(episode_skill or "").strip()
    prompt_text = str(observation or "")
    if not body:
        return prompt_text
    section = (
        "**Episode-Level Skill**\n"
        "Refer to this episode-level skill when deciding what action "
        "to take in the current episode:\n"
        f"[{body}]"
    )
    task_match = re.search(r"^.*\bYour task is to:\s*.*(?:\n|$)", prompt_text, flags=re.MULTILINE)
    if task_match is not None:
        prefix = prompt_text[: task_match.end()].rstrip()
        suffix = prompt_text[task_match.end() :].lstrip()
        if prefix and suffix:
            return f"{prefix}\n\n{section}\n\n{suffix}"
        if prefix:
            return f"{prefix}\n\n{section}"
        return f"{section}\n\n{suffix}"
    anchor = "Now it's your turn to"
    anchor_idx = prompt_text.find(anchor)
    if anchor_idx == -1:
        anchor = "Now it's your turn"
        anchor_idx = prompt_text.find(anchor)
    if anchor_idx == -1:
        return f"{prompt_text.rstrip()}\n\n{section}" if prompt_text else section
    prefix = prompt_text[:anchor_idx].rstrip()
    suffix = prompt_text[anchor_idx:].lstrip()
    if prefix and suffix:
        return f"{prefix}\n\n{section}\n\n{suffix}"
    if prefix:
        return f"{prefix}\n\n{section}"
    return f"{section}\n\n{suffix}"
