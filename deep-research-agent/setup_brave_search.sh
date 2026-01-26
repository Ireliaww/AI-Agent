#!/bin/bash
# Setup script for Brave Search MCP Server
# This script helps configure the Deep Research Agent to use real web search

set -e

echo "============================================"
echo "  Brave Search MCP Setup for Deep Research Agent"
echo "============================================"
echo ""

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "Error: Node.js is not installed."
    echo ""
    echo "Please install Node.js first:"
    echo "  macOS:   brew install node"
    echo "  Ubuntu:  sudo apt install nodejs npm"
    echo "  Windows: https://nodejs.org/en/download/"
    exit 1
fi

echo "✓ Node.js found: $(node --version)"
echo "✓ npm found: $(npm --version)"
echo ""

# Check if npx is available
if ! command -v npx &> /dev/null; then
    echo "Error: npx not found. Please update npm:"
    echo "  npm install -g npm"
    exit 1
fi

echo "✓ npx found"
echo ""

# Test if the Brave Search MCP server can be downloaded
echo "Testing Brave Search MCP server availability..."
if npx -y @anthropic/mcp-server-brave-search --help &> /dev/null; then
    echo "✓ @anthropic/mcp-server-brave-search is available"
    MCP_CMD="npx -y @anthropic/mcp-server-brave-search"
else
    echo "Trying alternative package..."
    if npx -y @modelcontextprotocol/server-brave-search --help &> /dev/null; then
        echo "✓ @modelcontextprotocol/server-brave-search is available"
        MCP_CMD="npx -y @modelcontextprotocol/server-brave-search"
    else
        echo "Error: Could not find Brave Search MCP server package"
        echo "Please check your internet connection or npm registry access"
        exit 1
    fi
fi

echo ""
echo "============================================"
echo "  Configuration"
echo "============================================"
echo ""

# Check for existing API key
if [ -f .env ]; then
    EXISTING_KEY=$(grep "^BRAVE_API_KEY=" .env 2>/dev/null | cut -d'=' -f2)
    if [ -n "$EXISTING_KEY" ] && [ "$EXISTING_KEY" != "your_brave_api_key_here" ]; then
        echo "Found existing BRAVE_API_KEY in .env"
        read -p "Do you want to keep it? (y/n): " KEEP_KEY
        if [ "$KEEP_KEY" = "y" ] || [ "$KEEP_KEY" = "Y" ]; then
            BRAVE_API_KEY="$EXISTING_KEY"
        fi
    fi
fi

# Prompt for API key if not set
if [ -z "$BRAVE_API_KEY" ]; then
    echo ""
    echo "You need a Brave Search API key."
    echo "Get one free at: https://brave.com/search/api/"
    echo "(Free tier: 2,000 queries/month)"
    echo ""
    read -p "Enter your Brave Search API Key: " BRAVE_API_KEY

    if [ -z "$BRAVE_API_KEY" ]; then
        echo "Error: API key is required"
        exit 1
    fi
fi

# Update .env file
echo ""
echo "Updating .env file..."

if [ -f .env ]; then
    # Update existing .env
    if grep -q "^BRAVE_API_KEY=" .env; then
        sed -i.bak "s|^BRAVE_API_KEY=.*|BRAVE_API_KEY=$BRAVE_API_KEY|" .env
    else
        echo "BRAVE_API_KEY=$BRAVE_API_KEY" >> .env
    fi

    if grep -q "^MCP_GOOGLE_SEARCH_CMD=" .env; then
        sed -i.bak "s|^MCP_GOOGLE_SEARCH_CMD=.*|MCP_GOOGLE_SEARCH_CMD=$MCP_CMD|" .env
    else
        echo "MCP_GOOGLE_SEARCH_CMD=$MCP_CMD" >> .env
    fi

    # Clean up backup file
    rm -f .env.bak
else
    # Create new .env
    cat > .env << EOF
GOOGLE_API_KEY=your_google_api_key_here
MCP_GOOGLE_SEARCH_CMD=$MCP_CMD
BRAVE_API_KEY=$BRAVE_API_KEY
GEMINI_MODEL=gemini-2.5-flash
DEBUG=false
EOF
fi

echo "✓ .env file updated"
echo ""
echo "============================================"
echo "  Setup Complete!"
echo "============================================"
echo ""
echo "Your configuration:"
echo "  MCP Server: $MCP_CMD"
echo "  API Key: ${BRAVE_API_KEY:0:10}..."
echo ""
echo "Test the setup with:"
echo "  source venv/bin/activate"
echo "  python main.py -q \"What is AI?\""
echo ""
echo "Note: Make sure your GOOGLE_API_KEY is also configured in .env"
echo ""
