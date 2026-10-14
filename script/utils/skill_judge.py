"""Keyword candidates, then a model marks steps that violate those skills.

Keyword matching only builds the candidate set. The judge may cite a skill on a
step only when that step's action violates it. Uncited steps are not distilled.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from script.utils.seed_prompt import format_episode_steps
from script.utils.skill_match import resolve_step


def build_judge_prompt(*, steps: list[dict], task_description: str, skill_blocks: list[tuple[str, str]]) -> str:
    catalog = "\n\n".join(f"### {name}\n{text}" for name, text in skill_blocks)
    names = [name for name, _ in skill_blocks]
    return f"""You judge an agent trajectory against a fixed skill list. Return ONLY valid JSON.

A step violates a skill only when its action breaks that skill. Relevance is not a violation.
If the step already follows the skill, or no candidate skill applies, omit the step.
You may cite only skill names from the candidate list. Do not invent skills.

Return format:
{{
  "violations": [
    {{"step": 0, "skills": ["general_skills"], "reason": "one short sentence"}}
  ]
}}

general_skills is page-scoped. Cite it only when the action breaks a rule for that page:
- search page: query formulation (rules 1, 2, 14)
- results page: scan/click candidate (rules 3, 9, 12)
- product page: variant/price/attribute verification (rules 4, 5, 6, 7, 8, 13)
The domain skill was chosen once from the task description. Do not cite any other category.

Candidate skill names: {names}

Candidate skill texts:
{catalog}

Task description: {task_description or "(not available)"}
Interaction trajectory:
{format_episode_steps(steps)}
"""


def parse_violations(text: str, *, allowed_skills: set[str], n_steps: int) -> dict[str, Any]:
    raw = str(text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.DOTALL)
    candidate = fenced.group(1) if fenced else raw
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end <= start:
        return {"by_step": {}, "parse_error": "no json object", "raw": raw}
    try:
        parsed = json.loads(candidate[start : end + 1])
    except json.JSONDecodeError as exc:
        return {"by_step": {}, "parse_error": str(exc), "raw": raw}

    by_step: dict[int, dict[str, Any]] = {}
    for item in parsed.get("violations") or []:
        if not isinstance(item, dict):
            continue
        try:
            step_idx = int(item.get("step"))
        except (TypeError, ValueError):
            continue
        if step_idx < 0 or step_idx >= n_steps:
            continue
        skills = []
        for name in item.get("skills") or []:
            skill_name = str(name)
            if skill_name in allowed_skills and skill_name not in skills:
                skills.append(skill_name)
        if not skills:
            continue
        slot = by_step.setdefault(step_idx, {"skills": [], "reason": ""})
        for skill_name in skills:
            if skill_name not in slot["skills"]:
                slot["skills"].append(skill_name)
        reason = str(item.get("reason") or "").strip()
        if reason:
            slot["reason"] = reason if not slot["reason"] else f"{slot['reason']} {reason}"
    return {"by_step": by_step, "parse_error": None, "raw": raw}


def complete_text(policy, prompt: str, *, backend: str, model: str | None, max_prompt_length: int) -> str:
    if backend == "hf":
        return policy.generate(prompt, temperature=0.0, max_prompt_length=max_prompt_length)["response"]
    if backend != "openai":
        raise ValueError(f"Unsupported ana backend: {backend}")
    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )
    response = client.chat.completions.create(
        model=model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content or ""


class SkillJudgeTeacher:
    """Keyword filter, then distill only steps the judge says violate a candidate skill."""

    name = "skill_judge"
    online = False

    def __init__(
        self,
        policy,
        skills_dir: str,
        *,
        skill_all: bool = False,
        temperature: float = 0.0,
        ana_backend: str = "hf",
        ana_model: str | None = None,
        ana_max_prompt_length: int = 32768,
    ):
        from verl.trainer.ppo.rlsd_utils import SkillProvider

        self.policy = policy
        self.temperature = float(temperature)
        self.provider = SkillProvider(skills_dir, skill_all=skill_all)
        self.ana_backend = ana_backend
        self.ana_model = ana_model
        self.ana_max_prompt_length = int(ana_max_prompt_length)

    def act(self, student_prompt: str, *, task: str = "", observation: str = "") -> dict[str, Any]:
        return {
            "selected_skills": None,
            "matched_tasks": None,
            "skill_text": None,
            "teacher_prompt": None,
            "teacher_response": None,
            "teacher_response_ids": None,
            "teacher_prompt_tokens": None,
            "distill": False,
        }

    def finalize(self, episode: dict) -> None:
        steps = episode["steps"]
        task_text = str(episode.get("task") or "")
        routed = [
            resolve_step(
                self.provider,
                task_text=task_text,
                observation=step.get("anchor_obs") or "",
                skill_all=self.provider.skill_all,
            )
            for step in steps
        ]
        candidate_names: list[str] = []
        for step, resolved in zip(steps, routed):
            step["keyword_skills"] = list(resolved["keyword_skills"])
            step["matched_tasks"] = list(resolved["matched_tasks"])
            step["domain_skill"] = resolved["domain_skill"]
            step["domain_keywords"] = list(resolved["domain_keywords"])
            step["page_type"] = resolved["page_type"]
            step["page_route"] = resolved["page_route"]
            step["page_rules"] = list(resolved["page_rules"])
            step["general_text"] = resolved.get("general_text") or ""
            step["routed_skill_text"] = resolved["skill_text"]
            for name in resolved["selected_skills"]:
                if name not in candidate_names and name in self.provider.skill_contents:
                    candidate_names.append(name)
        episode["domain_skill"] = routed[0]["domain_skill"] if routed else None
        episode["domain_keywords"] = list(routed[0]["domain_keywords"]) if routed else []

        skill_blocks = [(name, self.provider.skill_contents[name]) for name in candidate_names]
        analysis_steps = [
            {
                "step_index": int(step["turn"]),
                "observation": step["anchor_obs"],
                "response": step["student_response"],
                "action_valid": bool(step["is_action_valid"]),
            }
            for step in steps
        ]
        prompt = build_judge_prompt(
            steps=analysis_steps,
            task_description=str(episode.get("task") or ""),
            skill_blocks=skill_blocks,
        )
        raw = complete_text(
            self.policy,
            prompt,
            backend=self.ana_backend,
            model=self.ana_model,
            max_prompt_length=self.ana_max_prompt_length,
        )
        parsed = parse_violations(raw, allowed_skills=set(candidate_names), n_steps=len(steps))
        episode["candidate_skills"] = candidate_names
        episode["judge_prompt"] = prompt
        episode["judge_raw"] = raw
        episode["judge_parse_error"] = parsed["parse_error"]
        by_step = parsed["by_step"]

        for step in steps:
            verdict = by_step.get(int(step["turn"]))
            if not verdict:
                step["distill"] = False
                step["selected_skills"] = []
                step["skill_text"] = ""
                step["violation_reason"] = ""
                step["teacher_prompt"] = None
                step["teacher_response"] = None
                step["teacher_response_ids"] = None
                continue
            names = verdict["skills"]
            blocks = []
            for name in names:
                if name == "general_skills" and step.get("general_text"):
                    blocks.append(step["general_text"])
                elif name in self.provider.skill_contents:
                    blocks.append(self.provider.skill_contents[name])
            skill_text = "\n\n".join(blocks)
            teacher_user = f"[Privileged Skill Information]\n{skill_text}\n\n{step['student_prompt']}"
            generated = self.policy.generate(teacher_user, temperature=self.temperature)
            step["distill"] = True
            step["selected_skills"] = names
            step["skill_text"] = skill_text
            step["violation_reason"] = verdict["reason"]
            step["teacher_prompt"] = teacher_user
            step["teacher_response"] = generated["response"]
            step["teacher_response_ids"] = generated["response_ids"]
            step["teacher_prompt_tokens"] = generated["prompt_tokens"]
            step["teacher_action"] = ""
