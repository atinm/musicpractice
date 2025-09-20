#!/bin/bash
# MusicPractice Launcher Script

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"

# Change to project directory
cd "$PROJECT_DIR"

# Use the Python from the virtual environment directly
if [ -f ".venv/bin/python" ]; then
    echo "Using virtual environment Python..."
    PYTHON_CMD=".venv/bin/python"
else
    echo "Warning: .venv not found, using system Python"
    PYTHON_CMD="python3"
fi

# Run the app with Python
"$PYTHON_CMD" app.py "$@"
