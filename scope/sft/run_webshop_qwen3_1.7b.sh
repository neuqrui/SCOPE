#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TASK_KIND=WEBSHOP
# Keep an epoch count passed in by the caller. Otherwise train for 2 epochs.
_scope_sft_epochs_from_user="${SCOPE_SFT_EPOCHS-}"
source "$PROJECT_ROOT/scope/env.sh"
if [[ -n "$_scope_sft_epochs_from_user" ]]; then
    export SCOPE_SFT_EPOCHS="$_scope_sft_epochs_from_user"
else
    export SCOPE_SFT_EPOCHS=2
fi
export SCOPE_TASK=webshop
export SCOPE_MODEL_TAG=qwen3_1.7b
export SCOPE_BASE_MODEL="$SCOPE_BASE_QWEN3"
exec "$SCRIPT_DIR/_run.sh" "$@"
