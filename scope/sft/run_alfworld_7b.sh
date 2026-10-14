#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TASK_KIND=ALFWORLD
source "$PROJECT_ROOT/scope/env.sh"
export SCOPE_TASK=alfworld
export SCOPE_MODEL_TAG=qwen2.5_7b
export SCOPE_BASE_MODEL="$SCOPE_BASE_7B"
exec "$SCRIPT_DIR/_run.sh" "$@"
