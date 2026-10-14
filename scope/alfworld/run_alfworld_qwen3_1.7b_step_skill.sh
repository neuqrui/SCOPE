#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SCOPE_TASK=alfworld
export SCOPE_MODEL_TAG=qwen3_1.7b
export SCOPE_DISABLE_THINKING=1
exec "$SCRIPT_DIR/_run.sh" "$@"
