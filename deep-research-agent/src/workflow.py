"""
Deep Research Workflow Module - Multi-step research agent logic.

This module implements the core research workflow:
1. Plan: Analyze question and generate search queries
2. Act: Execute searches in parallel
3. Reason: Analyze results and determine if more info needed
4. Report: Generate comprehensive research report
"""

import asyncio
import json
from typing import Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum

from .client import GeminiClient
from .mcp_handler import MCPHandler, SearchResponse, create_mcp_handler


class WorkflowState(Enum):
    """Enum representing workflow states."""
    IDLE = "idle"
    PLANNING = "planning"
    SEARCHING = "searching"
    REASONING = "reasoning"
    REPORTING = "reporting"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class ResearchContext:
    """Context object tracking research progress."""

    original_question: str
    search_queries: list[str] = field(default_factory=list)
    search_results: list[SearchResponse] = field(default_factory=list)
    reasoning_notes: list[str] = field(default_factory=list)
    final_report: str = ""
    state: WorkflowState = WorkflowState.IDLE
    iteration: int = 0
    max_iterations: int = 3
    error: Optional[str] = None

    def add_search_results(self, results: list[SearchResponse]) -> None:
        """Add search results to context."""
        self.search_results.extend(results)

    def get_all_results_markdown(self) -> str:
        """Get all search results as markdown."""
        if not self.search_results:
            return "No search results available."

        sections = []
        for response in self.search_results:
            sections.append(response.to_markdown())

        return "\n\n".join(sections)

    def is_sufficient(self) -> bool:
        """Check if we have enough information."""
        # Simple heuristic: at least 5 results total
        total_results = sum(len(r.results) for r in self.search_results)
        return total_results >= 5


# Prompts for each workflow stage
PLAN_PROMPT = """Based on the user's research question, generate 3 diverse search queries that would help gather comprehensive information.

User's Question: {question}

Requirements:
1. Generate exactly 3 search queries
2. Each query should focus on a different aspect of the topic
3. Include a mix of broad and specific queries
4. Consider different perspectives or angles

Respond ONLY with a JSON object in this exact format:
{{
    "queries": [
        "first search query",
        "second search query",
        "third search query"
    ],
    "reasoning": "Brief explanation of why these queries were chosen"
}}
"""

REASON_PROMPT = """Analyze the following search results and determine if we have sufficient information to answer the user's question.

Original Question: {question}

Search Results:
{results}

Previous Analysis Notes:
{notes}

Current Iteration: {iteration} of {max_iterations}

Tasks:
1. Evaluate if the search results provide enough information to comprehensively answer the question
2. Identify any gaps in the information
3. If more searches are needed AND we haven't reached max iterations, suggest 2 additional queries

Respond ONLY with a JSON object in this exact format:
{{
    "sufficient": true/false,
    "confidence": 0.0-1.0,
    "analysis": "Your analysis of the available information",
    "gaps": ["list of information gaps if any"],
    "additional_queries": ["optional: additional search queries if needed"]
}}
"""

REPORT_PROMPT = """Generate a comprehensive research report based on the following information.

Original Question: {question}

Search Results:
{results}

Analysis Notes:
{notes}

Requirements:
1. Write a well-structured report in Markdown format
2. Include an executive summary at the beginning
3. Organize information into logical sections
4. Cite sources where applicable (use URLs from search results)
5. Include a "Sources" section at the end
6. Be objective and present multiple perspectives if relevant
7. Acknowledge any limitations or gaps in the available information

The report should be comprehensive but concise, suitable for someone seeking a thorough understanding of the topic.
"""


class DeepResearchWorkflow:
    """
    Implements the Deep Research workflow using Gemini and MCP.

    The workflow follows these steps:
    1. PLAN: Analyze the question and generate search queries
    2. ACT: Execute searches using MCP
    3. REASON: Evaluate if more information is needed
    4. REPORT: Generate final research report
    """

    def __init__(
        self,
        gemini_client: Optional[GeminiClient] = None,
        mcp_handler: Optional[MCPHandler] = None,
        use_mock: bool = False,
        on_state_change: Optional[Callable[[WorkflowState, str], None]] = None,
    ):
        """
        Initialize the workflow.

        Args:
            gemini_client: Optional pre-configured Gemini client.
            mcp_handler: Optional pre-configured MCP handler.
            use_mock: If True, use mock MCP handler for testing.
            on_state_change: Optional callback for state changes.
        """
        self.gemini = gemini_client or GeminiClient()
        self.mcp = mcp_handler or create_mcp_handler(use_mock=use_mock)
        self.on_state_change = on_state_change

    def _notify_state_change(self, state: WorkflowState, message: str) -> None:
        """Notify observers of state change."""
        if self.on_state_change:
            self.on_state_change(state, message)

    async def _plan(self, context: ResearchContext) -> list[str]:
        """
        Plan phase: Generate search queries based on the question.

        Args:
            context: The research context.

        Returns:
            List of search queries.
        """
        self._notify_state_change(
            WorkflowState.PLANNING,
            f"Analyzing question and generating search queries..."
        )

        prompt = PLAN_PROMPT.format(question=context.original_question)

        response = await self.gemini.generate_content(
            contents=prompt,
            temperature=0.7,
        )

        # Parse the JSON response
        try:
            # Extract JSON from response
            text = response.text.strip()
            # Handle potential markdown code blocks
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()

            data = json.loads(text)
            queries = data.get("queries", [])

            if queries:
                self._notify_state_change(
                    WorkflowState.PLANNING,
                    f"Generated {len(queries)} search queries: {queries}"
                )
                return queries

        except json.JSONDecodeError as e:
            self._notify_state_change(
                WorkflowState.PLANNING,
                f"Warning: Could not parse response as JSON, extracting queries manually"
            )

        # Fallback: extract queries from text
        lines = response.text.strip().split("\n")
        queries = [
            line.strip().strip("-").strip("*").strip('"').strip("'")
            for line in lines
            if line.strip() and not line.startswith("{") and not line.startswith("}")
        ][:3]

        return queries if queries else [context.original_question]

    async def _act(self, context: ResearchContext, queries: list[str]) -> list[SearchResponse]:
        """
        Act phase: Execute searches in parallel.

        Args:
            context: The research context.
            queries: List of search queries.

        Returns:
            List of search responses.
        """
        self._notify_state_change(
            WorkflowState.SEARCHING,
            f"Executing {len(queries)} searches in parallel..."
        )

        async with self.mcp.connect():
            # Execute all searches in parallel
            tasks = [self.mcp.call_search_tool(query) for query in queries]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions and convert to SearchResponse
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self._notify_state_change(
                    WorkflowState.SEARCHING,
                    f"Search failed for '{queries[i]}': {result}"
                )
                valid_results.append(SearchResponse(
                    query=queries[i],
                    error=str(result)
                ))
            else:
                valid_results.append(result)
                self._notify_state_change(
                    WorkflowState.SEARCHING,
                    f"Found {len(result.results)} results for '{queries[i]}'"
                )

        return valid_results

    async def _reason(self, context: ResearchContext) -> tuple[bool, list[str]]:
        """
        Reason phase: Analyze results and determine next steps.

        Args:
            context: The research context.

        Returns:
            Tuple of (is_sufficient, additional_queries).
        """
        self._notify_state_change(
            WorkflowState.REASONING,
            "Analyzing search results..."
        )

        prompt = REASON_PROMPT.format(
            question=context.original_question,
            results=context.get_all_results_markdown(),
            notes="\n".join(context.reasoning_notes) if context.reasoning_notes else "None",
            iteration=context.iteration,
            max_iterations=context.max_iterations,
        )

        response = await self.gemini.generate_content(
            contents=prompt,
            temperature=0.3,  # Lower temperature for more analytical response
        )

        # Parse the JSON response
        try:
            text = response.text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()

            data = json.loads(text)

            is_sufficient = data.get("sufficient", False)
            confidence = data.get("confidence", 0.5)
            analysis = data.get("analysis", "")
            additional_queries = data.get("additional_queries", [])

            # Store analysis in context
            context.reasoning_notes.append(
                f"Iteration {context.iteration}: {analysis} (confidence: {confidence:.2f})"
            )

            self._notify_state_change(
                WorkflowState.REASONING,
                f"Analysis complete. Sufficient: {is_sufficient}, Confidence: {confidence:.2f}"
            )

            return is_sufficient, additional_queries

        except json.JSONDecodeError:
            # Fallback: assume sufficient if we've done at least one iteration
            context.reasoning_notes.append(
                f"Iteration {context.iteration}: Could not parse reasoning response"
            )
            return context.iteration >= 1, []

    async def _report(self, context: ResearchContext) -> str:
        """
        Report phase: Generate final research report.

        Args:
            context: The research context.

        Returns:
            The final report as markdown.
        """
        self._notify_state_change(
            WorkflowState.REPORTING,
            "Generating comprehensive research report..."
        )

        prompt = REPORT_PROMPT.format(
            question=context.original_question,
            results=context.get_all_results_markdown(),
            notes="\n".join(context.reasoning_notes) if context.reasoning_notes else "None",
        )

        response = await self.gemini.generate_content(
            contents=prompt,
            temperature=0.5,
            max_output_tokens=4096,
        )

        report = response.text.strip()

        self._notify_state_change(
            WorkflowState.REPORTING,
            "Report generation complete."
        )

        return report

    async def run(self, question: str) -> ResearchContext:
        """
        Execute the full research workflow.

        Args:
            question: The research question from the user.

        Returns:
            ResearchContext with all results and final report.
        """
        context = ResearchContext(original_question=question)

        try:
            # Initial planning
            context.state = WorkflowState.PLANNING
            queries = await self._plan(context)
            context.search_queries.extend(queries)

            # Iterative search and reasoning loop
            while context.iteration < context.max_iterations:
                context.iteration += 1

                # Execute searches
                context.state = WorkflowState.SEARCHING
                results = await self._act(context, queries)
                context.add_search_results(results)

                # Analyze results
                context.state = WorkflowState.REASONING
                is_sufficient, additional_queries = await self._reason(context)

                if is_sufficient:
                    break

                if additional_queries and context.iteration < context.max_iterations:
                    queries = additional_queries[:2]  # Limit to 2 additional queries
                    context.search_queries.extend(queries)
                else:
                    break

            # Generate final report
            context.state = WorkflowState.REPORTING
            context.final_report = await self._report(context)

            context.state = WorkflowState.COMPLETED
            self._notify_state_change(
                WorkflowState.COMPLETED,
                "Research workflow completed successfully!"
            )

        except Exception as e:
            context.state = WorkflowState.ERROR
            context.error = str(e)
            self._notify_state_change(
                WorkflowState.ERROR,
                f"Workflow error: {e}"
            )

        return context


# Simplified interface for quick research
async def quick_research(
    question: str,
    use_mock: bool = False,
    verbose: bool = True
) -> str:
    """
    Quick helper to run a research query.

    Args:
        question: The research question.
        use_mock: Use mock MCP handler for testing.
        verbose: Print progress updates.

    Returns:
        The final research report.
    """
    def on_state_change(state: WorkflowState, message: str):
        if verbose:
            print(f"[{state.value.upper()}] {message}")

    workflow = DeepResearchWorkflow(
        use_mock=use_mock,
        on_state_change=on_state_change,
    )

    context = await workflow.run(question)

    if context.error:
        return f"Research failed: {context.error}"

    return context.final_report


if __name__ == "__main__":
    # Test the workflow with mock data
    async def test():
        report = await quick_research(
            "What are the latest developments in quantum computing?",
            use_mock=True,
            verbose=True,
        )
        print("\n" + "=" * 50)
        print("FINAL REPORT:")
        print("=" * 50)
        print(report)

    asyncio.run(test())
