"""Debug one WebShop episode with a student policy and a side-by-side teacher.

Does not modify or enter the training loop. Imports the existing WebShop manager,
``SkillProvider``, and (for SEED mode) the SEED observation-skill format.

Examples:
  python -m script.run_webshop_teacher_debug --teacher-mode skill_bank --episodes 1
  python -m script.run_webshop_teacher_debug --teacher-mode seed_ana --ana-backend hf --episodes 1
  python -m script.run_webshop_teacher_debug --teacher-mode seed_ana --ana-backend openai --ana-model gpt-4o-mini
"""

from __future__ import annotations

import argparse
import os

from script.utils.policy import LocalChatPolicy
from script.utils.report import write_episode
from script.utils.rollout import make_webshop_env, run_episode
from script.utils.scoring import annotate_episode
from script.utils.skill_judge import SkillJudgeTeacher
from script.utils.teachers import SeedAnalysisTeacher, SkillBankTeacher


def _parse_args():
    parser = argparse.ArgumentParser(description="WebShop student/teacher debug rollout")
    parser.add_argument(
        "--teacher-mode",
        choices=["skill_bank", "seed_ana", "skill_judge"],
        required=True,
    )
    parser.add_argument("--model-path", default=os.environ.get("MODEL_PATH", ""))
    parser.add_argument("--skills-dir", default="skills/webshop")
    parser.add_argument("--skill-all", action="store_true")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--history-length", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--goal-idx", type=int, default=None)
    parser.add_argument("--use-small", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-prompt-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--student-temperature", type=float, default=1.0)
    parser.add_argument("--teacher-temperature", type=float, default=0.0)
    parser.add_argument("--truncation", choices=["error", "left"], default="error")
    parser.add_argument("--logit-temperature", type=float, default=1.0)
    parser.add_argument("--opsd-coef", type=float, default=0.01)
    parser.add_argument("--ana-max-prompt-length", type=int, default=32768)
    parser.add_argument("--ana-backend", choices=["hf", "openai"], default="hf")
    parser.add_argument("--ana-model", default=os.environ.get("OPENAI_MODEL"))
    parser.add_argument(
        "--out-dir",
        default="script/outputs/webshop_teacher_debug",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    if not args.model_path:
        raise SystemExit("Pass --model-path or set MODEL_PATH")

    policy = LocalChatPolicy(
        args.model_path,
        max_prompt_length=args.max_prompt_length,
        max_new_tokens=args.max_new_tokens,
        temperature=args.student_temperature,
        truncation=args.truncation,
    )
    if args.teacher_mode == "skill_bank":
        teacher = SkillBankTeacher(
            policy,
            args.skills_dir,
            skill_all=args.skill_all,
            temperature=args.teacher_temperature,
        )
    elif args.teacher_mode == "skill_judge":
        teacher = SkillJudgeTeacher(
            policy,
            args.skills_dir,
            skill_all=args.skill_all,
            temperature=args.teacher_temperature,
            ana_backend=args.ana_backend,
            ana_model=args.ana_model,
            ana_max_prompt_length=args.ana_max_prompt_length,
        )
    else:
        teacher = SeedAnalysisTeacher(
            policy,
            ana_backend=args.ana_backend,
            ana_model=args.ana_model,
            temperature=args.teacher_temperature,
            ana_max_prompt_length=args.ana_max_prompt_length,
        )

    env = make_webshop_env(
        seed=args.seed,
        max_steps=args.max_steps,
        history_length=args.history_length,
        use_small=args.use_small,
        goal_idx=args.goal_idx,
    )
    out_dir = os.path.join(args.out_dir, args.teacher_mode)
    try:
        for episode_idx in range(args.episodes):
            episode = run_episode(
                env,
                policy,
                teacher,
                max_steps=args.max_steps,
                episode_idx=episode_idx,
            )
            annotate_episode(
                policy,
                episode,
                opsd_coef=args.opsd_coef,
                logit_temperature=args.logit_temperature,
            )
            json_path, html_path = write_episode(episode, out_dir)
            print(
                f"episode={episode_idx} won={episode['won']} "
                f"reward={episode['episode_reward']} html={html_path} json={json_path}"
            )
    finally:
        env.envs.close()


if __name__ == "__main__":
    main()
