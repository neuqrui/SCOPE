#!/usr/bin/env bash
# 180 train goals split across 5 models (36 each). Each model is data-parallel
# on GPUs 4,5,6,7: 9 goals per card, 8 rollouts, temperature 0.7.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TASK_KIND=WEBSHOP
source "$PROJECT_ROOT/examples/_sdar_env.sh"

export SAMPLE_GPUS="${SAMPLE_GPUS:-4,5,6,7}"
unset RAY_ADDRESS
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_GPU_MEMORY_UTILIZATION=0.6
export JAVA_HOME="${CONDA_PREFIX:-$ENV_PREFIX}/lib/jvm"
export JVM_PATH="$JAVA_HOME/lib/server/libjvm.so"
export PATH="$JAVA_HOME/bin:$PATH"
export _JAVA_OPTIONS="-XX:+UseSerialGC -Xss512k -Xmx1g"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

IFS=',' read -r -a GPU_LIST <<< "$SAMPLE_GPUS"
NUM_SHARDS=${#GPU_LIST[@]}
export OUT="${OUT:-work/ckpt/rollouts_val/webshop_train180_t0.7}"
LOG_DIR="${LOG_DIR:-work/logs/webshop_train180_dp}"
mkdir -p "$LOG_DIR" "$OUT/shards"
cd "$PROJECT_ROOT"

run_model() {
    local model_index="$1"
    local shard gpu
    local pids=()
    for shard in "${!GPU_LIST[@]}"; do
        gpu="${GPU_LIST[$shard]}"
        mkdir -p "work/tmp/ray_webshop_dp_${model_index}_${shard}"
        CUDA_VISIBLE_DEVICES="$gpu" \
        RAY_TMPDIR="work/tmp/ray_webshop_dp_${model_index}_${shard}" \
        python3 -m script.eval_webshop_train_sample \
            --output-dir "$OUT" \
            --model-index "$model_index" \
            --shard-index "$shard" \
            --num-shards "$NUM_SHARDS" \
            --tasks-per-model 36 \
            > "$LOG_DIR/model${model_index}_shard${shard}.log" 2>&1 &
        pids+=("$!")
        echo "model $model_index shard $shard -> GPU $gpu pid ${pids[-1]}"
    done
    local failed=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            failed=1
        fi
    done
    if [[ "$failed" -ne 0 ]]; then
        echo "model $model_index had a failed shard" >&2
        exit 1
    fi
    python3 -m script.eval_webshop_train_sample \
        --output-dir "$OUT" \
        --model-index "$model_index" \
        --num-shards "$NUM_SHARDS" \
        --tasks-per-model 36 \
        --merge-shards
}

for model_index in 0 1 2 3 4; do
    echo "===== model $model_index ====="
    run_model "$model_index"
done
python3 - << 'PY'
import json
from pathlib import Path
import os
root = Path(os.environ["OUT"])
goals = json.loads((root / "goals.json").read_text())
order = [int(i) for i in goals["config"]["goal_indices"]]
names = ["base", "global_step_50", "global_step_100", "global_step_150", "global_step_180"]
by_goal = {}
models = []
for name in names:
    data = json.loads((root / f"{name}.json").read_text())
    models.append({
        "model_name": data["model_name"],
        "model_path": data["model_path"],
        "checkpoint_step": data["checkpoint_step"],
        "summary": data["summary"],
        "prompt_truncations": data["prompt_truncations"],
        "goal_indices": [task["goal_index"] for task in data["tasks"]],
    })
    for task in data["tasks"]:
        task = dict(task)
        task["model_name"] = data["model_name"]
        task["checkpoint_step"] = data["checkpoint_step"]
        task["model_path"] = data["model_path"]
        by_goal[task["goal_index"]] = task
config = dict(goals["config"])
config["n_tasks"] = len(order)
config["tasks_per_model"] = 36
config["trajectory_format"] = "SEED analyzer: Step / Observation / Response / Action valid"
report = {
    "config": config,
    "models": models,
    "tasks": [by_goal[i] for i in order],
}
out = root / "all.json"
out.write_text(json.dumps(report))
print(f"wrote {out} tasks={len(report['tasks'])}")
PY
echo "done -> $OUT"
