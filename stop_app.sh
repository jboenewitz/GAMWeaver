#!/bin/bash

set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

kill_port() {
    local port="$1"
    local label="$2"
    local pids=""

    if command -v lsof >/dev/null 2>&1; then
        pids="$(lsof -ti tcp:"$port" 2>/dev/null | sort -u)"
    elif command -v fuser >/dev/null 2>&1; then
        pids="$(fuser "$port"/tcp 2>/dev/null | tr ' ' '\n' | sed '/^$/d' | sort -u)"
    else
        echo "⚠️  Neither lsof nor fuser is available, so $label cannot be stopped automatically."
        return
    fi

    if [ -z "$pids" ]; then
        echo "ℹ️  No $label process is using port $port."
        return
    fi

    echo "🛑 Stopping $label on port $port: $pids"
    kill $pids 2>/dev/null || true
    sleep 1

    if command -v lsof >/dev/null 2>&1; then
        pids="$(lsof -ti tcp:"$port" 2>/dev/null | sort -u)"
    else
        pids="$(fuser "$port"/tcp 2>/dev/null | tr ' ' '\n' | sed '/^$/d' | sort -u)"
    fi

    if [ -n "$pids" ]; then
        echo "⚠️  Forcing remaining $label processes to exit: $pids"
        kill -9 $pids 2>/dev/null || true
    fi
}

kill_matching_processes() {
    local pattern="$1"
    local label="$2"
    local pids=""

    if ! command -v pgrep >/dev/null 2>&1; then
        return
    fi

    pids="$(pgrep -f "$pattern" || true)"

    if [ -z "$pids" ]; then
        return
    fi

    echo "🧹 Cleaning up leftover $label processes: $pids"
    kill $pids 2>/dev/null || true
}

echo "🛑 Stopping Bike Rental Prediction app..."

kill_port 8000 "backend"
kill_port 3000 "frontend"

# Uvicorn with --reload can leave a parent watcher process behind.
kill_matching_processes "$SCRIPT_DIR/backend/venv/bin/python -m uvicorn app.main:app --reload" "backend watcher"
kill_matching_processes "$SCRIPT_DIR/frontend/node_modules/.bin/vite" "frontend watcher"

echo "✅ Stop command finished."
