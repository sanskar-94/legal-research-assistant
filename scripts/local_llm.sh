#!/usr/bin/env bash
# Start or stop the local answer-generation model (LFM3-1.7B via llama.cpp).
#
# llama-server exposes an OpenAI-compatible API, so the assistant uses it
# through LLM_BASE_URL / LLM_MODEL in .env with no other changes.
#
# Usage:
#   scripts/local_llm.sh start     # load the model in the background
#   scripts/local_llm.sh stop      # free the memory again
#   scripts/local_llm.sh status
#
# Requires llama.cpp 0.8.0 or newer (brew install llama.cpp) and the GGUF file
# from https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF. Override the defaults
# with LOCAL_LLM_MODEL_PATH and LOCAL_LLM_PORT.

set -euo pipefail

MODEL_PATH="${LOCAL_LLM_MODEL_PATH:-$HOME/models/LFM3-1.7B-Q4_K_M.gguf}"
PORT="${LOCAL_LLM_PORT:-8080}"
ALIAS="LFM3-1.7B"
STATE_DIR="${TMPDIR:-/tmp}/legal-research-assistant"
PID_FILE="$STATE_DIR/llama-server.pid"
LOG_FILE="$STATE_DIR/llama-server.log"

is_running() {
    [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null
}

is_ready() {
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1
}

start() {
    if is_running; then
        echo "Local model is already running on port $PORT."
        return
    fi
    if ! command -v llama-server >/dev/null; then
        echo "llama-server not found. Install it with: brew install llama.cpp" >&2
        exit 1
    fi
    if [[ ! -f "$MODEL_PATH" ]]; then
        echo "Model file not found: $MODEL_PATH" >&2
        exit 1
    fi

    mkdir -p "$STATE_DIR"
    # Settings are kept small for an 8 GB laptop: one request slot and an
    # 8k context (the prompt is about 3k tokens plus 1k for the answer).
    # Thinking is off so the whole token budget goes to the cited answer.
    # Sampling follows the model card: top-k 50, repeat penalty 1.1.
    nohup llama-server \
        -m "$MODEL_PATH" \
        --alias "$ALIAS" \
        --host 127.0.0.1 --port "$PORT" \
        -c 8192 -np 1 -ngl 99 \
        --reasoning off \
        --top-k 50 --repeat-penalty 1.1 \
        --no-webui \
        >"$LOG_FILE" 2>&1 &
    echo $! >"$PID_FILE"

    echo -n "Loading $ALIAS"
    for _ in $(seq 1 60); do
        if is_ready; then
            echo " ready on http://127.0.0.1:$PORT/v1"
            return
        fi
        if ! is_running; then
            echo " failed. See $LOG_FILE" >&2
            exit 1
        fi
        echo -n "."
        sleep 1
    done
    echo " still loading after 60 s. See $LOG_FILE" >&2
    exit 1
}

stop() {
    if is_running; then
        kill "$(cat "$PID_FILE")"
        echo "Local model stopped."
    else
        echo "Local model is not running."
    fi
    rm -f "$PID_FILE"
}

status() {
    if is_ready; then
        echo "Running and ready on http://127.0.0.1:$PORT/v1"
    elif is_running; then
        echo "Running but still loading. See $LOG_FILE"
    else
        echo "Not running."
    fi
}

case "${1:-}" in
    start) start ;;
    stop) stop ;;
    status) status ;;
    *) echo "Usage: $0 {start|stop|status}" >&2; exit 2 ;;
esac
