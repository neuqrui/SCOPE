"""SEED-format analysis prompt and skill injection.

The episode prompt matches ``SEEDEpisodeAnalyzer._build_default_episode_analysis_prompt``
for ``skill_mode=episode_only`` with ``include_episode_summary=True``.
Skill injection reuses ``SEED/seed/prompting.py`` when that file is present.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
from typing import Any


_SEED_ROOT = os.environ.get("SEED_ROOT", "/home/liuqirui/SEED")


def format_episode_steps(steps: list[dict]) -> str:
    """Same layout as ``SEEDEpisodeAnalyzer._format_episode_steps``."""
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
    if start == -1 or end == -1 or end <= start:
        return {"episode_summary": "", "episode_skill": "", "parse_error": "no json object", "raw": raw}
    try:
        parsed = json.loads(candidate[start : end + 1])
    except json.JSONDecodeError as exc:
        return {"episode_summary": "", "episode_skill": "", "parse_error": str(exc), "raw": raw}
    return {
        "episode_summary": str(parsed.get("episode_summary", "")),
        "episode_skill": str(parsed.get("episode_skill", "")),
        "raw": raw,
    }


def _load_seed_prompting():
    path = os.path.join(_SEED_ROOT, "seed", "prompting.py")
    if not os.path.isfile(path):
        return None
    spec = importlib.util.spec_from_file_location("seed_prompting_standalone", path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_SEED_PROMPTING = _load_seed_prompting()


def inject_episode_skill(observation: str, episode_skill: str) -> str:
    """Insert hindsight skill the way SEED builds the teacher observation."""
    if _SEED_PROMPTING is not None:
        return _SEED_PROMPTING.build_augmented_observation_text(
            observation=observation,
            episode_skill=episode_skill,
        )
    body = str(episode_skill or "").strip()
    if not body:
        return observation
    section = (
        "**Episode-Level Skill**\n"
        "Refer to this episode-level skill when deciding what action to take in the current episode:\n"
        f"[{body}]"
    )
    anchor = "Now it's your turn to"
    idx = observation.find(anchor)
    if idx == -1:
        return f"{observation.rstrip()}\n\n{section}"
    return f"{observation[:idx].rstrip()}\n\n{section}\n\n{observation[idx:].lstrip()}"
