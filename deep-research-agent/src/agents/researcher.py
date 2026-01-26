"""
Research Agent Module - Deep research capabilities.

This agent performs comprehensive research on any topic by:
1. Breaking down questions into search queries
2. Executing parallel searches
3. Analyzing and synthesizing information
4. Generating comprehensive reports
"""

import asyncio
from typing import Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..client import GeminiClient
from ..workflow import (
    DeepResearchWorkflow,
    WorkflowState,
    ResearchContext,
    PLAN_PROMPT,
    REASON_PROMPT,
    REPORT_PROMPT,
)
from ..mcp_handler import create_mcp_handler, SearchResponse

console = Console()


class ResearchState(Enum):
    """States for the research workflow."""
    IDLE = "idle"
    PLANNING = "planning"
    SEARCHING = "searching"
    REASONING = "reasoning"
    REPORTING = "reporting"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class ResearchResult:
    """Result of a research session."""
    question: str
    report: str
    queries_used: list[str] = field(default_factory=list)
    total_results: int = 0
    iterations: int = 0
    success: bool = False
    error: Optional[str] = None


class ResearchAgent:
    """
    Agent for conducting deep research on any topic.

    Uses the Plan-Act-Reason-Report workflow:
    1. PLAN: Analyze question and generate search queries
    2. ACT: Execute searches using MCP
    3. REASON: Evaluate if more information is needed
    4. REPORT: Generate comprehensive research report
    """

    def __init__(
        self,
        gemini_client: Optional[GeminiClient] = None,
        use_mock: bool = False,
        on_state_change: Optional[Callable[[ResearchState, str], None]] = None,
    ):
        """
        Initialize the Research Agent.

        Args:
            gemini_client: Optional pre-configured Gemini client
            use_mock: Use mock search for testing
            on_state_change: Callback for state change notifications
        """
        self.gemini = gemini_client or GeminiClient()
        self.use_mock = use_mock
        self.on_state_change = on_state_change
        self._workflow: Optional[DeepResearchWorkflow] = None

    def _notify_state_change(self, state: ResearchState, message: str) -> None:
        """Notify observers of state change."""
        if self.on_state_change:
            self.on_state_change(state, message)

    def _map_workflow_state(self, wf_state: WorkflowState) -> ResearchState:
        """Map workflow state to research state."""
        mapping = {
            WorkflowState.IDLE: ResearchState.IDLE,
            WorkflowState.PLANNING: ResearchState.PLANNING,
            WorkflowState.SEARCHING: ResearchState.SEARCHING,
            WorkflowState.REASONING: ResearchState.REASONING,
            WorkflowState.REPORTING: ResearchState.REPORTING,
            WorkflowState.COMPLETED: ResearchState.COMPLETED,
            WorkflowState.ERROR: ResearchState.ERROR,
        }
        return mapping.get(wf_state, ResearchState.IDLE)

    async def research(self, question: str) -> ResearchResult:
        """
        Conduct research on a question.

        Args:
            question: The research question

        Returns:
            ResearchResult with findings
        """
        # Create workflow callback that maps to our state
        def workflow_callback(wf_state: WorkflowState, message: str):
            research_state = self._map_workflow_state(wf_state)
            self._notify_state_change(research_state, message)

        # Create and run the workflow
        self._workflow = DeepResearchWorkflow(
            gemini_client=self.gemini,
            use_mock=self.use_mock,
            on_state_change=workflow_callback,
        )

        context = await self._workflow.run(question)

        # Convert to ResearchResult
        return ResearchResult(
            question=question,
            report=context.final_report,
            queries_used=context.search_queries,
            total_results=sum(len(r.results) for r in context.search_results),
            iterations=context.iteration,
            success=context.error is None,
            error=context.error,
        )


async def run_research_agent(question: str, use_mock: bool = False) -> str:
    """
    Main entry point for the research agent.

    Args:
        question: The research question
        use_mock: Use mock search for testing

    Returns:
        Markdown-formatted research report
    """
    console.print(Panel(
        f"[bold cyan]Research Question:[/bold cyan]\n{question}",
        title="[bold]Research Agent[/bold]",
        border_style="cyan"
    ))

    if use_mock:
        console.print("[yellow]Using MOCK search (for testing)[/yellow]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=False,
    ) as progress:
        task_id = progress.add_task("[cyan]Initializing...", total=None)

        def on_state_change(state: ResearchState, message: str):
            style_map = {
                ResearchState.PLANNING: "blue",
                ResearchState.SEARCHING: "magenta",
                ResearchState.REASONING: "yellow",
                ResearchState.REPORTING: "cyan",
                ResearchState.COMPLETED: "green",
                ResearchState.ERROR: "red",
            }
            style = style_map.get(state, "white")
            progress.update(task_id, description=f"[{style}]{message}[/{style}]")

        agent = ResearchAgent(
            use_mock=use_mock,
            on_state_change=on_state_change,
        )

        result = await agent.research(question)
        progress.update(task_id, visible=False)

    # Display summary
    if result.success:
        summary_lines = [
            f"[info]Total iterations:[/info] {result.iterations}",
            f"[info]Search queries used:[/info] {len(result.queries_used)}",
            f"[info]Total results collected:[/info] {result.total_results}",
        ]
        console.print(Panel(
            "\n".join(summary_lines),
            title="Research Summary",
            border_style="cyan"
        ))

        return result.report
    else:
        error_msg = f"""## Research Failed

**Error:** {result.error}

Please try again or rephrase your question.
"""
        return error_msg


async def quick_research(
    question: str,
    use_mock: bool = False,
    verbose: bool = True
) -> str:
    """
    Quick helper for research queries.

    Args:
        question: The research question
        use_mock: Use mock search for testing
        verbose: Print progress updates

    Returns:
        Research report as markdown
    """
    def on_state_change(state: ResearchState, message: str):
        if verbose:
            style_map = {
                ResearchState.PLANNING: "blue",
                ResearchState.SEARCHING: "magenta",
                ResearchState.REASONING: "yellow",
                ResearchState.REPORTING: "cyan",
                ResearchState.COMPLETED: "green",
                ResearchState.ERROR: "red",
            }
            style = style_map.get(state, "white")
            console.print(f"[{style}][{state.value.upper()}][/{style}] {message}")

    agent = ResearchAgent(
        use_mock=use_mock,
        on_state_change=on_state_change,
    )

    result = await agent.research(question)

    if result.success:
        return result.report
    else:
        return f"Research failed: {result.error}"


if __name__ == "__main__":
    # Test the research agent
    async def test():
        report = await run_research_agent(
            "What are the latest developments in quantum computing?",
            use_mock=True
        )
        console.print("\n")
        console.print(Panel(
            Markdown(report),
            title="Research Report",
            border_style="green",
            padding=(1, 2)
        ))

    asyncio.run(test())
