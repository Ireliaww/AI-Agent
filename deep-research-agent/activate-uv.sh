#!/bin/bash
# Quick activate script using uv for deep-research-agent

# Change to the project directory
cd "$(dirname "$0")"

# Activate the virtual environment
source venv/bin/activate

echo "✅ Virtual environment activated (managed by uv)!"
echo "📁 Current directory: $(pwd)"
echo "🐍 Python: $(which python)"
echo "⚡ uv version: $(~/.local/bin/uv --version)"
echo ""
echo "💡 Quick commands:"
echo "   - Install packages: uv pip install <package>"
echo "   - Install from requirements: uv pip install -r requirements.txt"
echo "   - To deactivate: deactivate"
