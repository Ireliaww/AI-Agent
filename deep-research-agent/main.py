#!/usr/bin/env python3
"""
Multi-Mode AI Assistant - Unified Entry Point

A multi-mode AI assistant powered by Gemini with:
- Deep Research: Information gathering and comprehensive reports
- Auto Coding: LeetCode solver with self-healing debug loop
- Smart Router: Automatic intent classification and task routing

Usage:
    python main.py                    # Interactive mode
    python main.py --mock             # Use mock services (for testing)
    python main.py -q "query"         # Direct query mode
    python main.py --mode coding      # Force coding mode
    python main.py --mode research    # Force research mode
"""

import os
import sys
import asyncio
import argparse
from typing import Optional

# Handle nested event loops
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
from rich.text import Text
from rich.table import Table

# Import our modules
from src.router import Router, Intent, classify_intent
from src.agents.researcher import run_research_agent
from src.agents.coder import run_coding_agent

# Custom theme
custom_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "success": "bold green",
    "coding": "bold magenta",
    "research": "bold blue",
    "router": "bold cyan",
})

console = Console(theme=custom_theme)

# ASCII Logo
ASCII_LOGO = r"""
[bold cyan]
    __  ___      ____  _      __  ___          __        ___    ____
   /  |/  /_  __/ / /_(_)    /  |/  /___  ____/ /__     /   |  /  _/
  / /|_/ / / / / / __/ /    / /|_/ / __ \/ __  / _ \   / /| |  / /
 / /  / / /_/ / / /_/ /    / /  / / /_/ / /_/ /  __/  / ___ |_/ /
/_/  /_/\__,_/_/\__/_/    /_/  /_/\____/\__,_/\___/  /_/  |_/___/

[/bold cyan]
[dim]Powered by Gemini AI | Deep Research + Auto Coding[/dim]
"""

HELP_TEXT = """
[bold cyan]Available Commands:[/bold cyan]
  [green]/help[/green]     - Show this help message
  [green]/mode[/green]     - Switch mode (coding/research/auto)
  [green]/clear[/green]    - Clear the screen
  [green]/quit[/green]     - Exit the assistant

[bold cyan]Modes:[/bold cyan]
  [magenta]CODING[/magenta]   - Solve coding problems (LeetCode, algorithms)
  [blue]RESEARCH[/blue] - Deep research on any topic
  [cyan]AUTO[/cyan]     - Automatic mode detection (default)

[bold cyan]Examples:[/bold cyan]
  "Write a function to find two numbers that add up to a target"
  "What are the latest developments in quantum computing?"
  "LeetCode: Implement binary search"
  "Research AI trends in 2024"
"""


class MultiModeAssistant:
    """
    Multi-Mode AI Assistant with automatic routing.
    """

    def __init__(self, use_mock: bool = False, forced_mode: Optional[str] = None):
        """
        Initialize the assistant.

        Args:
            use_mock: Use mock services for testing
            forced_mode: Force a specific mode ('coding', 'research', or None for auto)
        """
        self.use_mock = use_mock
        self.forced_mode = forced_mode
        self.router = Router()

    async def process_query(self, query: str) -> str:
        """
        Process a user query and return the result.

        Args:
            query: The user's query

        Returns:
            Markdown-formatted result
        """
        # Determine intent
        if self.forced_mode:
            if self.forced_mode == "coding":
                intent = Intent.CODING
                confidence = 1.0
                reason = "Forced coding mode"
            else:
                intent = Intent.RESEARCH
                confidence = 1.0
                reason = "Forced research mode"

            console.print(Panel(
                f"[green]Mode:[/green] {intent.value.upper()}\n[dim](forced)[/dim]",
                title="[bold cyan]Router[/bold cyan]",
                border_style="cyan"
            ))
        else:
            intent, confidence, reason = await classify_intent(query, show_ui=True)

        # Route to appropriate agent
        console.print()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            if intent == Intent.CODING:
                progress.add_task(
                    f"[magenta]Routing to Coding Agent...[/magenta]",
                    total=None
                )
                await asyncio.sleep(0.5)  # Brief pause for visual feedback

        if intent == Intent.CODING:
            result = await run_coding_agent(query)
        else:
            result = await run_research_agent(query, use_mock=self.use_mock)

        return result


def print_banner():
    """Print the application banner."""
    console.print(ASCII_LOGO)


def print_help():
    """Print help information."""
    console.print(Panel(
        HELP_TEXT,
        title="[bold]Help[/bold]",
        border_style="cyan"
    ))


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


async def interactive_mode(use_mock: bool = False, forced_mode: Optional[str] = None):
    """
    Run the assistant in interactive REPL mode.

    Args:
        use_mock: Use mock services for testing
        forced_mode: Force a specific mode
    """
    print_banner()

    if not use_mock and not check_api_key():
        console.print("\n[warning]Tip: Use --mock flag to test without an API key[/warning]")
        sys.exit(1)

    assistant = MultiModeAssistant(use_mock=use_mock, forced_mode=forced_mode)

    # Show current mode
    mode_display = forced_mode.upper() if forced_mode else "AUTO"
    console.print(f"\n[info]Current mode:[/info] [bold]{mode_display}[/bold]")

    if use_mock:
        console.print("[warning]Running in MOCK mode (for testing)[/warning]")

    console.print(
        "\nWelcome to the Multi-Mode AI Assistant!\n"
        "Enter your query, or type [green]/help[/green] for commands.\n"
    )

    while True:
        try:
            # Get user input
            query = Prompt.ask("\n[bold cyan]You[/bold cyan]")

            # Handle commands
            if query.lower() in ['/quit', '/exit', '/q', 'quit', 'exit']:
                console.print("\n[info]Goodbye![/info]")
                break

            if query.lower() in ['/help', '/h', 'help']:
                print_help()
                continue

            if query.lower() in ['/clear', '/cls']:
                console.clear()
                print_banner()
                continue

            if query.lower().startswith('/mode'):
                parts = query.split()
                if len(parts) > 1:
                    new_mode = parts[1].lower()
                    if new_mode in ['coding', 'code', 'c']:
                        assistant.forced_mode = 'coding'
                        console.print("[success]Switched to CODING mode[/success]")
                    elif new_mode in ['research', 'r']:
                        assistant.forced_mode = 'research'
                        console.print("[success]Switched to RESEARCH mode[/success]")
                    elif new_mode in ['auto', 'a']:
                        assistant.forced_mode = None
                        console.print("[success]Switched to AUTO mode[/success]")
                    else:
                        console.print("[warning]Invalid mode. Use: coding, research, or auto[/warning]")
                else:
                    current = assistant.forced_mode or "auto"
                    console.print(f"[info]Current mode: {current.upper()}[/info]")
                continue

            if not query.strip():
                console.print("[warning]Please enter a query or command.[/warning]")
                continue

            # Process the query
            result = await assistant.process_query(query)

            # Display result
            console.print("\n")
            console.print(Panel(
                Markdown(result),
                title="[bold green]Result[/bold green]",
                border_style="green",
                padding=(1, 2)
            ))

        except KeyboardInterrupt:
            console.print("\n\n[info]Interrupted. Goodbye![/info]")
            break
        except Exception as e:
            console.print(f"\n[error]Error: {e}[/error]")


async def direct_mode(
    query: str,
    use_mock: bool = False,
    forced_mode: Optional[str] = None
):
    """
    Run a single query and exit.

    Args:
        query: The query to process
        use_mock: Use mock services for testing
        forced_mode: Force a specific mode
    """
    print_banner()

    if not use_mock and not check_api_key():
        sys.exit(1)

    assistant = MultiModeAssistant(use_mock=use_mock, forced_mode=forced_mode)

    result = await assistant.process_query(query)

    console.print("\n")
    console.print(Panel(
        Markdown(result),
        title="[bold green]Result[/bold green]",
        border_style="green",
        padding=(1, 2)
    ))


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Multi-Mode AI Assistant - Deep Research & Auto Coding",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                              # Interactive mode
  python main.py --mock                       # Test with mock services
  python main.py -q "Write two sum function"  # Direct query
  python main.py --mode coding -q "..."       # Force coding mode
  python main.py --mode research -q "..."     # Force research mode
        """
    )

    parser.add_argument(
        "-q", "--query",
        type=str,
        help="Direct query (skip interactive mode)"
    )

    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock services (for testing without real API)"
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["coding", "research", "auto"],
        default="auto",
        help="Force a specific mode (default: auto)"
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

    # Determine forced mode
    forced_mode = None if args.mode == "auto" else args.mode

    # Run in appropriate mode
    if args.query:
        asyncio.run(direct_mode(
            args.query,
            use_mock=args.mock,
            forced_mode=forced_mode
        ))
    else:
        asyncio.run(interactive_mode(
            use_mock=args.mock,
            forced_mode=forced_mode
        ))


if __name__ == "__main__":
    main()
