#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TASK_KIND=WEBSHOP
source "$PROJECT_ROOT/scope/env.sh"
export SCOPE_TASK=webshop
export SCOPE_MODEL_TAG=qwen2.5_3b
export SCOPE_BASE_MODEL="$SCOPE_BASE_3B"
exec "$SCRIPT_DIR/_run.sh" "$@"
