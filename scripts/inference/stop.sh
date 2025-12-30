#!/bin/bash
# Stop WAG API Server

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/api_server.pid"
PORT="${PORT:-5000}"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "Stopping server (PID: $PID)..."
        kill "$PID"
        sleep 2
        # Force kill if still running
        if ps -p "$PID" > /dev/null 2>&1; then
            kill -9 "$PID"
        fi
        rm -f "$PID_FILE"
        echo "Server stopped."
    else
        echo "Server not running (stale PID file)."
        rm -f "$PID_FILE"
    fi
else
    # Try to find and kill by port
    PID=$(lsof -t -i:$PORT 2>/dev/null)
    if [ -n "$PID" ]; then
        echo "Found server on port $PORT (PID: $PID). Stopping..."
        kill "$PID" 2>/dev/null || sudo kill "$PID"
        echo "Server stopped."
    else
        echo "Server not running."
    fi
fi
