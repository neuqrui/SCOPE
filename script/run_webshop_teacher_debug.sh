#!/usr/bin/env bash
# WebShop student/teacher debug loop.
# Edit the block below, then: bash script/run_webshop_teacher_debug.sh

set -euo pipefail

# skill_bank: keyword skill on every step.
# seed_ana: SEED-format hindsight skill.
# skill_judge: keyword candidates, distill only steps that violate one of them.
TEACHER_MODE=skill_bank
CUDA_VISIBLE_DEVICES=0
# Local HF directory. .env's MODEL_PATH is applied first, then overridden below.
MODEL_PATH=/data/qirui/SDAR/models/Qwen2.5-3B-Instruct
# SEED hindsight-skill SFT currently on disk (ALFWorld, not WebShop):
# MODEL_PATH=/data/qirui/SEED/models/SEED-ALFWorld-3B-SFT

EPISODES=1
# Train rollout uses actor_rollout_ref.rollout.temperature=1.0.
# The webshop script only overrides validation sampling to 0.4.
STUDENT_TEMPERATURE=1.0
TEACHER_TEMPERATURE=1.0
# Log-prob / entropy temperature. Training divides logits by rollout.temperature (1.0).
LOGIT_TEMPERATURE=1.0
# Sampled-token OPSD, same coef as examples/opsd_kl_trainer/run_webshop_3b.sh
# L = opsd_coef * mean(-stopgrad(logπ_T - logπ_S) * logπ_S)
OPSD_COEF=1
MAX_NEW_TOKENS=512
MAX_PROMPT_LENGTH=4096
TRUNCATION=error          # error | left

MAX_STEPS=15
HISTORY_LENGTH=2
SEED=0
GOAL_IDX=                 # empty = random train goal each episode
USE_SMALL=true            # true = 1000-item catalog

SKILLS_DIR=skills/webshop
SKILL_ALL=false           # true = prepend every skill

# seed_ana only. hf uses the same local model; openai needs OPENAI_API_KEY.
# Analysis sees the full trajectory, so it is not capped by MAX_PROMPT_LENGTH.
ANA_BACKEND=hf            # hf | openai
ANA_MODEL=
ANA_MAX_PROMPT_LENGTH=32768

OUT_DIR=script/outputs/webshop_teacher_debug

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TASK_KIND=WEBSHOP
# .env overwrites these when sourced. Restore the values chosen above.
chosen_cuda="$CUDA_VISIBLE_DEVICES"
chosen_model="$MODEL_PATH"
source "$PROJECT_ROOT/examples/_sdar_env.sh"
export CUDA_VISIBLE_DEVICES="$chosen_cuda"
export MODEL_PATH="$chosen_model"

cd "$PROJECT_ROOT"

args=(
    --teacher-mode "$TEACHER_MODE"
    --model-path "$MODEL_PATH"
    --episodes "$EPISODES"
    --student-temperature "$STUDENT_TEMPERATURE"
    --teacher-temperature "$TEACHER_TEMPERATURE"
    --logit-temperature "$LOGIT_TEMPERATURE"
    --opsd-coef "$OPSD_COEF"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --max-prompt-length "$MAX_PROMPT_LENGTH"
    --truncation "$TRUNCATION"
    --max-steps "$MAX_STEPS"
    --history-length "$HISTORY_LENGTH"
    --seed "$SEED"
    --skills-dir "$SKILLS_DIR"
    --ana-backend "$ANA_BACKEND"
    --ana-max-prompt-length "$ANA_MAX_PROMPT_LENGTH"
    --out-dir "$OUT_DIR"
)

if [[ "$USE_SMALL" == "true" ]]; then
    args+=(--use-small)
else
    args+=(--no-use-small)
fi

if [[ "$SKILL_ALL" == "true" ]]; then
    args+=(--skill-all)
fi

if [[ -n "$GOAL_IDX" ]]; then
    args+=(--goal-idx "$GOAL_IDX")
fi

if [[ -n "$ANA_MODEL" ]]; then
    args+=(--ana-model "$ANA_MODEL")
fi

python -m script.run_webshop_teacher_debug "${args[@]}"
