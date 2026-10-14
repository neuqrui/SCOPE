"""Two teacher modes for the debug loop.

``skill_bank`` resolves a static skill at each step and generates a teacher action
on the skill-prefixed prompt (same prefix as ``build_teacher_batch``).

``seed_ana`` lets the student finish the episode, then builds a SEED analysis
prompt from raw per-step observations (not the 2-step history window) and asks
an analyzer for ``episode_skill``. Teacher actions are then generated on each
saved student prompt with that skill inserted.
"""

from __future__ import annotations

import os
from typing import Any

from verl.trainer.ppo.rlsd_utils import SkillProvider

from script.utils.skill_match import resolve_step
from script.utils.seed_prompt import (
    build_episode_analysis_prompt,
    inject_episode_skill,
    parse_analysis_json,
)


class SkillBankTeacher:
    name = "skill_bank"
    online = True

    def __init__(self, policy, skills_dir: str, skill_all: bool = False, temperature: float = 0.0):
        self.policy = policy
        self.temperature = float(temperature)
        self.provider = SkillProvider(skills_dir, skill_all=skill_all)

    def act(self, student_prompt: str, *, task: str = "", observation: str = "") -> dict[str, Any]:
        resolved = resolve_step(
            self.provider,
            task_text=task,
            observation=observation,
            skill_all=self.provider.skill_all,
        )
        skill_text = resolved["skill_text"]
        teacher_user = f"[Privileged Skill Information]\n{skill_text}\n\n{student_prompt}"
        generated = self.policy.generate(teacher_user, temperature=self.temperature)
        return {
            "selected_skills": list(resolved["selected_skills"]),
            "matched_tasks": list(resolved["matched_tasks"]),
            "skill_text": skill_text,
            "keyword_skills": list(resolved["keyword_skills"]),
            "domain_skill": resolved["domain_skill"],
            "domain_keywords": list(resolved["domain_keywords"]),
            "page_type": resolved["page_type"],
            "page_route": resolved["page_route"],
            "page_rules": list(resolved["page_rules"]),
            "teacher_prompt": teacher_user,
            "teacher_response": generated["response"],
            "teacher_response_ids": generated["response_ids"],
            "teacher_prompt_tokens": generated["prompt_tokens"],
        }

    def finalize(self, episode: dict) -> None:
        return None


class SeedAnalysisTeacher:
    name = "seed_ana"
    online = False

    def __init__(
        self,
        policy,
        *,
        ana_backend: str = "hf",
        ana_model: str | None = None,
        ana_policy=None,
        temperature: float = 0.0,
        ana_max_prompt_length: int = 32768,
    ):
        self.policy = policy
        self.temperature = float(temperature)
        self.ana_max_prompt_length = int(ana_max_prompt_length)
        self.ana_backend = ana_backend
        self.ana_model = ana_model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self.ana_policy = ana_policy if ana_policy is not None else policy

    def act(self, student_prompt: str, *, task: str = "", observation: str = "") -> dict[str, Any]:
        return {
            "selected_skills": None,
            "matched_tasks": None,
            "skill_text": None,
            "teacher_prompt": None,
            "teacher_response": None,
            "teacher_response_ids": None,
            "teacher_prompt_tokens": None,
        }

    def finalize(self, episode: dict) -> None:
        steps = episode["steps"]
        analysis_steps = []
        for step in steps:
            analysis_steps.append(
                {
                    "step_index": int(step["turn"]),
                    "observation": step["anchor_obs"],
                    "response": step["student_response"],
                    "action_valid": bool(step["is_action_valid"]),
                    "step_reward": float(step["reward"]),
                }
            )
        won = bool(episode.get("won"))
        prompt = build_episode_analysis_prompt(
            steps=analysis_steps,
            task_description=str(episode.get("task") or ""),
            episode_success=1.0 if won else 0.0,
        )
        raw = self._complete_analysis(prompt)
        parsed = parse_analysis_json(raw)
        episode["analysis_prompt"] = prompt
        episode["analysis_raw"] = raw
        episode["episode_summary"] = parsed.get("episode_summary", "")
        episode["episode_skill"] = parsed.get("episode_skill", "")
        episode["analysis_parse_error"] = parsed.get("parse_error")
        skill = episode["episode_skill"]
        for step in steps:
            teacher_user = inject_episode_skill(step["student_prompt"], skill)
            generated = self.policy.generate(teacher_user, temperature=self.temperature)
            step["skill_text"] = skill
            step["selected_skills"] = ["episode_skill"] if skill else []
            step["matched_tasks"] = ["hindsight"]
            step["teacher_prompt"] = teacher_user
            step["teacher_response"] = generated["response"]
            step["teacher_response_ids"] = generated["response_ids"]
            step["teacher_prompt_tokens"] = generated["prompt_tokens"]

    def _complete_analysis(self, prompt: str) -> str:
        if self.ana_backend == "hf":
            return self.ana_policy.generate(
                prompt,
                temperature=0.0,
                max_prompt_length=self.ana_max_prompt_length,
            )["response"]
        if self.ana_backend != "openai":
            raise ValueError(f"Unsupported ana backend: {self.ana_backend}")
        from openai import OpenAI

        client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("OPENAI_BASE_URL"),
        )
        response = client.chat.completions.create(
            model=self.ana_model,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""
