#!/bin/bash
# Start WAG API Server

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="/mnt/data/sri/wag/scripts/venv"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-5000}"
LOG_FILE="$SCRIPT_DIR/api_server.log"
PID_FILE="$SCRIPT_DIR/api_server.pid"

# Check if already running
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "Server already running (PID: $PID)"
        exit 1
    else
        rm -f "$PID_FILE"
    fi
fi

# Activate virtual environment and start server
echo "Starting WAG API Server on $HOST:$PORT..."
source "$VENV_PATH/bin/activate"
cd "$SCRIPT_DIR"
nohup python api_server.py --host "$HOST" --port "$PORT" > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"

sleep 2

# Verify it started
if [ -f "$PID_FILE" ] && ps -p $(cat "$PID_FILE") > /dev/null 2>&1; then
    echo "Server started successfully (PID: $(cat $PID_FILE))"
    echo "Logs: $LOG_FILE"
    echo "URL: http://$HOST:$PORT/"
else
    echo "Failed to start server. Check $LOG_FILE for errors."
    exit 1
fi
