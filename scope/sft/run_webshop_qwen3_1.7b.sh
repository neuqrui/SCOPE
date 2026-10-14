#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TASK_KIND=WEBSHOP
source "$PROJECT_ROOT/scope/env.sh"
export SCOPE_TASK=webshop
export SCOPE_MODEL_TAG=qwen3_1.7b
export SCOPE_BASE_MODEL="$SCOPE_BASE_QWEN3"
exec "$SCRIPT_DIR/_run.sh" "$@"
