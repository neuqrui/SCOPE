#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SCOPE_TASK=search
export SCOPE_MODEL_TAG=qwen2.5_7b
export SCOPE_DISABLE_THINKING=0
exec "$SCRIPT_DIR/_run.sh" "$@"
