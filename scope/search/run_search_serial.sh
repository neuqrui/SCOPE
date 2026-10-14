#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# search: 3B, then 7B, then Qwen3-1.7B. Each run loads its own SFT checkpoint.
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TASK_KIND=SEARCH
source "$PROJECT_ROOT/scope/env.sh"
LOG="$TMPDIR/scope_retrieval_server.log"
RETRIEVER_PID=
if ! curl -sf -o /dev/null -X POST "$SEARCH_URL" -H "Content-Type: application/json" -d '{"query": "test", "topk": 1}' --max-time 5; then
    echo "Starting retrieval server"
    bash "$PROJECT_ROOT/examples/search/retriever/retrieval_launch.sh" > "$LOG" 2>&1 &
    RETRIEVER_PID=$!
    ready=0
    for _ in $(seq 1 400); do
        if curl -sf -o /dev/null -X POST "$SEARCH_URL" -H "Content-Type: application/json" -d '{"query": "test", "topk": 1}' --max-time 5; then
            echo "Retrieval server is ready"
            ready=1
            break
        fi
        if ! kill -0 "$RETRIEVER_PID" 2>/dev/null; then
            echo "Retrieval server exited. See $LOG" >&2
            exit 1
        fi
        sleep 5
    done
    if [[ "$ready" != 1 ]]; then
        echo "Retrieval server did not become ready. See $LOG" >&2
        exit 1
    fi
fi
cleanup() { [[ -n "$RETRIEVER_PID" ]] && kill "$RETRIEVER_PID" 2>/dev/null || true; }
trap cleanup EXIT
echo "========== SCOPE search 3b =========="
bash "$SCRIPT_DIR/run_search_3b_step_skill.sh"
echo "========== SCOPE search 7b =========="
bash "$SCRIPT_DIR/run_search_7b_step_skill.sh"
echo "========== SCOPE search qwen3_1.7b =========="
bash "$SCRIPT_DIR/run_search_qwen3_1.7b_step_skill.sh"
echo "SCOPE search serial run finished."
