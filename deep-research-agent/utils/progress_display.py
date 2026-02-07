"""
Progress Display - Real-time visualization of AI agent work

Provides rich visual feedback during:
- Paper analysis
- Code generation  
- File creation
- AI thinking process
"""

from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn
)
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
from rich.tree import Tree
from rich.live import Live
from rich.layout import Layout
from typing import Optional, Callable
import time


class ProgressDisplay:
    """Manages real-time progress display for AI agent operations"""
    
    def __init__(self):
        self.console = Console()
        self.current_progress: Optional[Progress] = None
        self.live_display: Optional[Live] = None
    
    def show_analysis_progress(self, paper_title: str):
        """
        Show progress during paper analysis
        
        Args:
            paper_title: Title of paper being analyzed
        """
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=self.console
        )
        
        self.current_progress = progress
        return progress
    
    def show_ai_thinking(self, thought: str, step: str = ""):
        """
        Display AI's current thinking in a panel
        
        Args:
            thought: What the AI is currently thinking
            step: Current step name
        """
        title = f"🧠 AI Thinking{': ' + step if step else ''}"
        
        panel = Panel(
            thought,
            title=title,
            border_style="cyan",
            padding=(1, 2)
        )
        
        self.console.print(panel)
    
    def show_generation_progress(
        self,
        file_path: str,
        total_tokens: int = 100,
        current_tokens: int = 0
    ):
        """
        Show progress for code generation
        
        Args:
            file_path: Path of file being generated
            total_tokens: Estimated total tokens
            current_tokens: Current token count
        """
        percentage = (current_tokens / total_tokens * 100) if total_tokens > 0 else 0
        
        self.console.print(
            f"📝 Generating `{file_path}`... "
            f"[cyan]{current_tokens}/{total_tokens} tokens[/cyan] "
            f"({percentage:.0f}%)"
        )
    
    def show_file_tree(self, project_name: str, files: list, directories: list):
        """
        Display project structure as a tree
        
        Args:
            project_name: Name of the project
            files: List of file paths
            directories: List of directory paths
        """
        tree = Tree(f"📁 {project_name}", style="bold cyan")
        
        # Build directory structure
        dir_nodes = {}
        for dir_path in sorted(directories):
            parts = dir_path.split('/')
            parent = tree
            current_path = ""
            
            for part in parts:
                current_path = f"{current_path}/{part}" if current_path else part
                if current_path not in dir_nodes:
                    dir_nodes[current_path] = parent.add(f"📂 {part}/", style="blue")
                parent = dir_nodes[current_path]
        
        # Add files
        for file_path in sorted(files):
            parts = file_path.split('/')
            if len(parts) > 1:
                dir_path = '/'.join(parts[:-1])
                parent = dir_nodes.get(dir_path, tree)
            else:
                parent = tree
            
            # Icon based on file type
            if file_path.endswith('.py'):
                icon = "🐍"
            elif file_path.endswith('.md'):
                icon = "📄"
            elif file_path.endswith(('.yaml', '.yml')):
                icon = "⚙️"
            elif file_path.endswith('.txt'):
                icon = "📝"
            else:
                icon = "📄"
            
            parent.add(f"{icon} {parts[-1]}", style="green")
        
        self.console.print(tree)
    
    def show_code_preview(
        self,
        code: str,
        language: str = "python",
        line_numbers: bool = True,
        title: str = ""
    ):
        """
        Show syntax-highlighted code preview
        
        Args:
            code: Code to display
            language: Programming language
            line_numbers: Whether to show line numbers
            title: Optional title for the preview
        """
        syntax = Syntax(
            code,
            language,
            theme="monokai",
            line_numbers=line_numbers,
            word_wrap=False
        )
        
        if title:
            panel = Panel(syntax, title=title, border_style="green")
            self.console.print(panel)
        else:
            self.console.print(syntax)
    
    def show_statistics_table(self, stats: dict):
        """
        Show statistics in a formatted table
        
        Args:
            stats: Dictionary of statistics
        """
        table = Table(title="📊 Generation Statistics", show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")
        
        for key, value in stats.items():
            table.add_row(key, str(value))
        
        self.console.print(table)
    
    def create_live_status(self) -> Live:
        """
        Create a live-updating status display
        
        Returns:
            Live display object
        """
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3)
        )
        
        live = Live(layout, console=self.console, refresh_per_second=4)
        self.live_display = live
        return live
    
    def update_live_status(self, section: str, content):
        """
        Update a section of the live display
        
        Args:
            section: Section name (header, body, footer)
            content: Content to display
        """
        if self.live_display and hasattr(self.live_display, 'renderable'):
            self.live_display.renderable[section].update(content)
    
    def show_step_header(self, step_number: int, step_name: str, total_steps: int):
        """
        Show a formatted step header
        
        Args:
            step_number: Current step number
            step_name: Name of the step
            total_steps: Total number of steps
        """
        progress_bar = "█" * step_number + "░" * (total_steps - step_number)
        
        panel = Panel(
            f"[bold cyan]Step {step_number}/{total_steps}:[/bold cyan] {step_name}\n"
            f"[dim]{progress_bar}[/dim]",
            border_style="blue",
            padding=(0, 2)
        )
        
        self.console.print(panel)
    
    def show_completion_summary(
        self,
        title: str,
        items: list,
        duration: float,
        success: bool = True
    ):
        """
        Show a completion summary
        
        Args:
            title: Summary title
            items: List of completed items
            duration: Time taken in seconds
            success: Whether the operation succeeded
        """
        status = "✅ Success" if success else "❌ Failed"
        style = "green" if success else "red"
        
        lines = [f"[bold {style}]{status}[/bold {style}]", ""]
        
        for item in items:
            if isinstance(item, dict):
                lines.append(f"  {item.get('icon', '•')} {item.get('text', str(item))}")
            else:
                lines.append(f"  ✓ {item}")
        
        lines.append("")
        lines.append(f"[dim]Completed in {duration:.1f}s[/dim]")
        
        panel = Panel(
            "\n".join(lines),
            title=f"📋 {title}",
            border_style=style,
            padding=(1, 2)
        )
        
        self.console.print(panel)
    
    def prompt_user(self, question: str, choices: list) -> str:
        """
        Prompt user for input with choices
        
        Args:
            question: Question to ask
            choices: List of choice strings
            
        Returns:
            User's choice
        """
        self.console.print(f"\n[bold yellow]{question}[/bold yellow]")
        
        for i, choice in enumerate(choices, 1):
            self.console.print(f"  [{i}] {choice}")
        
        while True:
            try:
                response = input("\nYour choice: ").strip()
                
                # Check if it's a number
                if response.isdigit():
                    idx = int(response) - 1
                    if 0 <= idx < len(choices):
                        return choices[idx]
                
                # Check if it matches a choice
                for choice in choices:
                    if response.lower() == choice.lower():
                        return choice
                
                self.console.print("[red]Invalid choice. Please try again.[/red]")
            except (KeyboardInterrupt, EOFError):
                return choices[0] if choices else ""


# Global instance
progress_display = ProgressDisplay()
