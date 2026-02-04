#!/bin/bash
# Quick activate script for deep-research-agent virtual environment

# Change to the project directory
cd "$(dirname "$0")"

# Activate the virtual environment
source venv/bin/activate

echo "✅ Virtual environment activated!"
echo "📁 Current directory: $(pwd)"
echo "🐍 Python: $(which python)"
echo ""
echo "To deactivate, run: deactivate"
