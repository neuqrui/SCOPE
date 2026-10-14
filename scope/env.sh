#!/usr/bin/env bash
# SCOPE bootstrap. Sources the SDAR env, then the SCOPE model paths.
# Callers set TASK_KIND and PROJECT_ROOT before sourcing this file.

SCOPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCOPE_DIR/.." && pwd)}"
# shellcheck disable=SC1091
source "$PROJECT_ROOT/examples/_sdar_env.sh"

if [[ -f "$SCOPE_DIR/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$SCOPE_DIR/.env"
    set +a
fi

# Defaults live here so a missing scope/.env still resolves the three base models.
# scope/.env can override any of these. Repo .gitignore ignores *.env files.
: "${SCOPE_BASE_3B:=/data/qirui/SDAR/models/Qwen2.5-3B-Instruct}"
: "${SCOPE_BASE_7B:=Qwen/Qwen2.5-7B-Instruct}"
: "${SCOPE_BASE_QWEN3:=/data/qirui/SDAR/models/Qwen3-1.7B}"
: "${SCOPE_SFT_ROOT:=/data/qirui/SDAR/ckpt/sft/scope}"
: "${SCOPE_SFT_EPOCHS:=3}"
: "${SCOPE_SAMPLE_ROOT:=/data/qirui/SDAR/ckpt/rollouts_val/scope}"
export SCOPE_BASE_3B SCOPE_BASE_7B SCOPE_BASE_QWEN3 SCOPE_SFT_ROOT SCOPE_SFT_EPOCHS SCOPE_SAMPLE_ROOT

scope_latest_sft() {
    local save_dir="$1"
    if [[ -n "${SCOPE_RL_MODEL:-}" ]]; then
        if [[ ! -f "${SCOPE_RL_MODEL}/config.json" ]]; then
            echo "SCOPE_RL_MODEL has no config.json: ${SCOPE_RL_MODEL}" >&2
            return 1
        fi
        echo "$SCOPE_RL_MODEL"
        return 0
    fi
    if [[ ! -d "$save_dir" ]]; then
        echo "SFT checkpoint directory is missing: $save_dir" >&2
        echo "Train scope/sft for this task and model first. SFT and RL stay in separate scripts." >&2
        return 1
    fi
    local step
    step="$(find "$save_dir" -mindepth 1 -maxdepth 1 -type d -name 'global_step_*' -printf '%f\n' | sed 's/global_step_//' | sort -n | tail -1)"
    if [[ -z "$step" || ! -f "$save_dir/global_step_${step}/config.json" ]]; then
        echo "No global_step checkpoint with config.json under $save_dir" >&2
        return 1
    fi
    if [[ "$step" -le 200 ]]; then
        echo "Latest SFT checkpoint must be past step 200, got global_step_${step} under $save_dir" >&2
        return 1
    fi
    echo "$save_dir/global_step_${step}"
}

scope_init_method() {
    # Three method settings: step_skill, OPSD gate, length reward.
    LENGTH_REWARD="${LENGTH_REWARD:-pairwise}"
    case "$LENGTH_REWARD" in
        none)
            success_length_adv_coef=0
            success_length_bonus_coef=0
            success_length_pairwise_tau=0
            ;;
        adv)
            success_length_adv_coef=1.0
            success_length_bonus_coef=0
            success_length_pairwise_tau=0
            ;;
        bonus)
            success_length_adv_coef=0
            success_length_bonus_coef=5.0
            success_length_pairwise_tau=0
            ;;
        pairwise)
            success_length_adv_coef=0
            success_length_bonus_coef=0
            success_length_pairwise_tau=0.1
            ;;
        *)
            echo "LENGTH_REWARD must be none, adv, bonus, or pairwise, got ${LENGTH_REWARD}" >&2
            return 1
            ;;
    esac

    opsd_coef=0.01
    opsd_objective=sampled_kl
    opsd_gate="${OPSD_GATE:-sdar}"
    opsd_gate_beta="${OPSD_GATE_BETA:-5.0}"
    opsd_adv_clip="${OPSD_ADV_CLIP:-10}"
    skill_source=step_skill
    ana_loss_mode=none
    ana_backend=policy_vllm
    case "$opsd_gate" in
        sdar|sdar-gate|sdar_gate)
            opsd_gate_tag="gatesdar_beta${opsd_gate_beta}"
            ;;
        clip)
            opsd_gate_tag="gateclip_c${opsd_adv_clip}"
            ;;
        tanh)
            opsd_gate_tag="gatetanh_c${opsd_adv_clip}"
            ;;
        *)
            echo "OPSD_GATE must be sdar-gate, clip, or tanh (got ${opsd_gate})" >&2
            return 1
            ;;
    esac
    experiment_name="scope_${SCOPE_TASK}_${SCOPE_MODEL_TAG}_${opsd_gate_tag}_length_${LENGTH_REWARD}_step_skill"
}
