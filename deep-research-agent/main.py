#!/usr/bin/env python3
"""
Deep Research Agent - CLI Entry Point

A multi-step research agent that uses Gemini AI and MCP protocol
to conduct thorough research on any topic.

Usage:
    python main.py                    # Interactive mode
    python main.py --mock             # Use mock search (for testing)
    python main.py --question "..."   # Direct question mode
"""

import os
import sys
import asyncio
import argparse
from pathlib import Path

# Handle nested event loops (e.g., in Jupyter notebooks)
import nest_asyncio
nest_asyncio.apply()

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Rich for beautiful console output
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from rich.theme import Theme

# Import our modules
from src.workflow import DeepResearchWorkflow, WorkflowState
from src.client import GeminiClient


# Custom theme for the CLI
custom_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "success": "bold green",
    "state.planning": "blue",
    "state.searching": "magenta",
    "state.reasoning": "yellow",
    "state.reporting": "cyan",
    "state.completed": "green",
    "state.error": "red",
})

console = Console(theme=custom_theme)


def print_banner():
    """Print the application banner."""
    banner = """
╔═══════════════════════════════════════════════════════════════╗
║              Deep Research Agent v0.1.0                       ║
║        Powered by Gemini AI + MCP Protocol                    ║
╚═══════════════════════════════════════════════════════════════╝
    """
    console.print(banner, style="bold cyan")


def check_api_key() -> bool:
    """Check if the Google API key is configured."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key == "your_google_api_key_here":
        console.print(
            "\n[error]Error: GOOGLE_API_KEY not configured![/error]\n"
            "Please set your API key in one of these ways:\n"
            "  1. Create a .env file with: GOOGLE_API_KEY=your_key_here\n"
            "  2. Export environment variable: export GOOGLE_API_KEY=your_key_here\n"
            "\nGet your API key from: https://aistudio.google.com/apikey"
        )
        return False
    return True


def create_state_callback(progress: Progress, task_id) -> callable:
    """Create a callback function for workflow state updates."""

    state_styles = {
        WorkflowState.PLANNING: ("state.planning", "Planning"),
        WorkflowState.SEARCHING: ("state.searching", "Searching"),
        WorkflowState.REASONING: ("state.reasoning", "Reasoning"),
        WorkflowState.REPORTING: ("state.reporting", "Reporting"),
        WorkflowState.COMPLETED: ("state.completed", "Completed"),
        WorkflowState.ERROR: ("state.error", "Error"),
    }

    def callback(state: WorkflowState, message: str):
        style, label = state_styles.get(state, ("info", state.value))
        progress.update(task_id, description=f"[{style}]{label}: {message}[/{style}]")
        progress.refresh()

    return callback


async def run_research(question: str, use_mock: bool = False) -> None:
    """
    Run the research workflow for a given question.

    Args:
        question: The research question.
        use_mock: Whether to use mock search (for testing).
    """
    console.print(f"\n[info]Research Question:[/info] {question}\n")

    if use_mock:
        console.print("[warning]Using MOCK search server (for testing)[/warning]\n")

    # Create workflow with progress tracking
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=False,
    ) as progress:
        task_id = progress.add_task("[info]Initializing...", total=None)

        callback = create_state_callback(progress, task_id)

        try:
            workflow = DeepResearchWorkflow(
                use_mock=use_mock,
                on_state_change=callback,
            )

            context = await workflow.run(question)

            # Clear the progress
            progress.update(task_id, visible=False)

        except Exception as e:
            progress.update(task_id, visible=False)
            console.print(f"\n[error]Workflow error: {e}[/error]")
            return

    # Display results
    if context.error:
        console.print(Panel(
            f"[error]Research failed:[/error] {context.error}",
            title="Error",
            border_style="red"
        ))
        return

    # Show workflow summary
    summary_lines = [
        f"[info]Total iterations:[/info] {context.iteration}",
        f"[info]Search queries used:[/info] {len(context.search_queries)}",
        f"[info]Total results collected:[/info] {sum(len(r.results) for r in context.search_results)}",
    ]
    console.print(Panel(
        "\n".join(summary_lines),
        title="Research Summary",
        border_style="cyan"
    ))

    # Show the final report
    console.print("\n")
    console.print(Panel(
        Markdown(context.final_report),
        title="Research Report",
        border_style="green",
        padding=(1, 2)
    ))


def interactive_mode(use_mock: bool = False):
    """Run the agent in interactive mode."""
    print_banner()

    if not use_mock and not check_api_key():
        console.print("\n[warning]Tip: Use --mock flag to test without an API key[/warning]")
        sys.exit(1)

    console.print(
        "\nWelcome to the Deep Research Agent!\n"
        "Enter your research question, or type 'quit' to exit.\n"
    )

    while True:
        try:
            question = Prompt.ask("\n[bold cyan]Your question[/bold cyan]")

            if question.lower() in ['quit', 'exit', 'q']:
                console.print("\n[info]Goodbye![/info]")
                break

            if not question.strip():
                console.print("[warning]Please enter a valid question.[/warning]")
                continue

            asyncio.run(run_research(question, use_mock=use_mock))

        except KeyboardInterrupt:
            console.print("\n\n[info]Interrupted. Goodbye![/info]")
            break
        except Exception as e:
            console.print(f"\n[error]Error: {e}[/error]")


def direct_mode(question: str, use_mock: bool = False):
    """Run a single research query and exit."""
    print_banner()

    if not use_mock and not check_api_key():
        sys.exit(1)

    asyncio.run(run_research(question, use_mock=use_mock))


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Deep Research Agent - AI-powered research assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                    # Interactive mode
  python main.py --mock                             # Test with mock search
  python main.py -q "What is quantum computing?"   # Direct question
  python main.py --mock -q "AI trends 2024"        # Mock mode with question
        """
    )

    parser.add_argument(
        "-q", "--question",
        type=str,
        help="Research question (skip interactive mode)"
    )

    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock search server (for testing without real API)"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output"
    )

    args = parser.parse_args()

    # Set debug mode
    if args.debug:
        os.environ["DEBUG"] = "true"

    # Run in appropriate mode
    if args.question:
        direct_mode(args.question, use_mock=args.mock)
    else:
        interactive_mode(use_mock=args.mock)


if __name__ == "__main__":
    main()
