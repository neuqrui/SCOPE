#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# webshop: 3B, then 7B, then Qwen3-1.7B. Each run loads its own SFT checkpoint.
echo "========== SCOPE webshop 3b =========="
bash "$SCRIPT_DIR/run_webshop_3b_step_skill.sh"
echo "========== SCOPE webshop 7b =========="
bash "$SCRIPT_DIR/run_webshop_7b_step_skill.sh"
echo "========== SCOPE webshop qwen3_1.7b =========="
bash "$SCRIPT_DIR/run_webshop_qwen3_1.7b_step_skill.sh"
echo "SCOPE webshop serial run finished."
