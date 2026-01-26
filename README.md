# Multi-Mode AI Assistant

A powerful AI assistant powered by Gemini with dual capabilities: **Deep Research** for comprehensive information gathering and **Auto Coding** for solving programming problems with self-healing debug loops.

## Features

### Smart Router
- **Automatic Intent Classification:** Intelligently routes queries to the appropriate agent based on content analysis
- **Keyword Detection:** Fast routing for obvious cases (LeetCode, code, research, explain, etc.)
- **AI-Powered Classification:** Uses Gemini for ambiguous queries with confidence scoring

### Deep Research Mode
- **Iterative Research:** Multi-step workflow that progressively gathers information
- **Parallel Search Execution:** Runs multiple search queries simultaneously for efficiency
- **Comprehensive Reports:** Generates well-structured Markdown reports with sources
- **MCP Protocol:** Uses Model Context Protocol for robust tool communication

### Auto Coding Mode
- **Self-Healing Debug Loop:** Automatically fixes code errors (up to 3 attempts)
- **LeetCode Solver:** Optimized for algorithm and data structure problems
- **Test Case Generation:** Includes comprehensive test cases with edge cases
- **Solution Persistence:** Saves solutions to files for later reference

## How It Works

### Research Workflow (Plan → Act → Reason → Report)

1. **Plan:** Generates diverse search queries to explore the topic from multiple angles
2. **Act:** Executes searches in parallel using MCP-based search tools
3. **Reason:** Evaluates if information is sufficient; generates targeted follow-up queries if needed
4. **Report:** Produces a comprehensive Markdown report with executive summary and sources

### Coding Workflow (Code → Run → Fix)

1. **Generate:** Creates Python solution with test cases based on problem description
2. **Execute:** Runs the code and captures output/errors
3. **Debug:** If tests fail, analyzes errors and generates corrected code
4. **Iterate:** Repeats until all tests pass or max attempts reached

## Project Structure

```
deep-research-agent/
├── main.py                 # Unified entry point
├── src/
│   ├── router.py          # Intent classification
│   ├── client.py          # Gemini API client
│   ├── agents/
│   │   ├── researcher.py  # Research agent
│   │   └── coder.py       # Coding agent
│   └── tools/
│       ├── search_tools.py
│       └── file_tools.py
├── solutions/             # Generated code solutions
└── requirements.txt
```

## Quick Start

1. **Navigate to the project directory:**
    ```bash
    cd deep-research-agent
    ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Configure API Key:**
    Create a `.env` file with your Google API key:
    ```
    GOOGLE_API_KEY=your_key_here
    ```
    Get your API key from [Google AI Studio](https://aistudio.google.com/apikey).

4. **Run the assistant:**
    ```bash
    python main.py
    ```

## Usage

### Interactive Mode
```bash
python main.py                    # Auto-detect mode
python main.py --mode coding      # Force coding mode
python main.py --mode research    # Force research mode
```

### Direct Query Mode
```bash
python main.py -q "Write a function to find two numbers that add up to a target"
python main.py -q "What are the latest developments in quantum computing?"
```

### Testing (Mock Mode)
```bash
python main.py --mock             # Test without API calls
```

### In-App Commands
- `/help` - Show help message
- `/mode <coding|research|auto>` - Switch modes
- `/clear` - Clear screen
- `/quit` - Exit

## Examples

**Coding Query:**
```
You: Write a function to solve LeetCode Two Sum problem
→ Routes to Coding Agent → Generates code → Runs tests → Auto-fixes if needed
```

**Research Query:**
```
You: What are the latest AI trends in 2024?
→ Routes to Research Agent → Searches → Analyzes → Generates report
```

## Dependencies

- `google-genai` - Google Gemini AI SDK
- `mcp` - Model Context Protocol SDK
- `python-dotenv` - Environment variable management
- `nest_asyncio` - Async event loop handling
- `httpx` - HTTP client
- `rich` - Beautiful console output
- `questionary` - Interactive CLI prompts

## Agent Demos

This repository also includes a collection of other AI agent examples in the `Agent_demos` directory. These demos showcase various agent implementations using different frameworks and models, such as AutoGen, LangChain, and the Anthropic API. You can explore the `Agent_demos` directory to see these other examples.