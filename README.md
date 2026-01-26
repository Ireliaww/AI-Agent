# Deep Research Agent

This repository contains the Deep Research Agent, a multi-step research agent that uses Gemini AI and the MCP protocol to conduct thorough research on any topic.

## Features

-   **Interactive Mode:** Engage in a conversation with the agent to refine your research questions.
-   **Direct Question Mode:** Get a quick answer to a specific question.
-   **Powered by Gemini AI:** Leverages the power of Google's Gemini models for research and analysis.
-   **MCP Protocol:** Implements the Model Context Protocol for robust and structured communication.
-   **Mock Search:** Includes a mock search mode for testing the agent without making real API calls.
-   **Rich Console Output:** Provides a user-friendly and easy-to-read console experience.

## How it Works

The Deep Research Agent follows a multi-step, iterative workflow to gather and synthesize information:

1.  **Plan:** Based on the initial research question, the agent uses Gemini to brainstorm a set of diverse search queries. This ensures that the topic is explored from multiple angles.

2.  **Act:** The agent executes these search queries in parallel. It uses the Model Context Protocol (MCP) to call a search tool, which can be a real search engine or a mock server for testing.

3.  **Reason:** The agent analyzes the collected search results. It uses Gemini to evaluate whether it has enough information to answer the user's question comprehensively. If the information is insufficient and the agent has not reached its maximum iteration limit, it will generate additional, more targeted search queries to fill in the gaps.

4.  **Iterate:** The agent loops through the "Act" and "Reason" steps, progressively building up its knowledge base until it determines the information is sufficient.

5.  **Report:** Once the research is complete, the agent generates a comprehensive, well-structured report in Markdown format. The report includes an executive summary, logical sections for the gathered information, and a list of the sources used.

## Usage

1.  **Navigate to the agent's directory:**
    ```bash
    cd deep-research-agent
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure API Key:**
    Create a `.env` file in the `deep-research-agent` directory with your Google API key:
    ```
    GOOGLE_API_KEY=your_key_here
    ```
    You can get your API key from [Google AI Studio](https://aistudio.google.com/apikey).

4.  **Run the agent:**
    -   **Interactive mode:**
        ```bash
        python main.py
        ```
    -   **Direct question mode:**
        ```bash
        python main.py --question "What is quantum computing?"
        ```
    -   **Mock mode (for testing without an API key):**
        ```bash
        python main.py --mock
        ```

## Dependencies

The `deep-research-agent` requires the following Python packages:

-   `google-genai`
-   `mcp`
-   `python-dotenv`
-   `nest_asyncio`
-   `httpx`
-   `rich`

## Agent Demos

This repository also includes a collection of other AI agent examples in the `Agent_demos` directory. These demos showcase various agent implementations using different frameworks and models, such as AutoGen, LangChain, and the Anthropic API. You can explore the `Agent_demos` directory to see these other examples.