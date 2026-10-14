#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# alfworld: 3B, then 7B, then Qwen3-1.7B. Each run loads its own SFT checkpoint.
echo "========== SCOPE alfworld 3b =========="
bash "$SCRIPT_DIR/run_alfworld_3b_step_skill.sh"
echo "========== SCOPE alfworld 7b =========="
bash "$SCRIPT_DIR/run_alfworld_7b_step_skill.sh"
echo "========== SCOPE alfworld qwen3_1.7b =========="
bash "$SCRIPT_DIR/run_alfworld_qwen3_1.7b_step_skill.sh"
echo "SCOPE alfworld serial run finished."
