"""
File Tools Module - File operations and shell execution for AI Agent.

This module provides tools for:
- Saving code solutions to the solutions/ directory
- Running Python code and capturing output
- Shell command execution
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Tuple, Optional
from dataclasses import dataclass

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

console = Console()

# Solutions directory path
SOLUTIONS_DIR = Path(__file__).parent.parent.parent / "solutions"


@dataclass
class ExecutionResult:
    """Result of code execution."""
    success: bool
    stdout: str
    stderr: str
    return_code: int
    filename: str


def ensure_solutions_dir() -> Path:
    """Ensure the solutions directory exists."""
    SOLUTIONS_DIR.mkdir(parents=True, exist_ok=True)
    return SOLUTIONS_DIR


def save_solution(filename: str, code: str, show_ui: bool = True) -> str:
    """
    Save a code solution to the solutions/ directory.

    Args:
        filename: Name of the file (e.g., "two_sum.py")
        code: The Python code to save
        show_ui: Whether to display UI feedback

    Returns:
        Full path to the saved file
    """
    ensure_solutions_dir()

    # Ensure .py extension
    if not filename.endswith(".py"):
        filename = f"{filename}.py"

    filepath = SOLUTIONS_DIR / filename

    # Write the code to file
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(code)

    if show_ui:
        console.print(Panel(
            f"[green]Code saved to:[/green] {filepath}",
            title="[bold cyan]File Saved[/bold cyan]",
            border_style="green"
        ))

        # Show syntax-highlighted code preview
        syntax = Syntax(code, "python", theme="monokai", line_numbers=True)
        console.print(Panel(syntax, title="[bold]Code Preview[/bold]", border_style="cyan"))

    return str(filepath)


def run_solution(filename: str, timeout: int = 30, show_ui: bool = True) -> ExecutionResult:
    """
    Run a Python solution file using subprocess.

    Args:
        filename: Name of the file to run (relative to solutions/)
        timeout: Maximum execution time in seconds
        show_ui: Whether to display UI feedback

    Returns:
        ExecutionResult containing stdout, stderr, and return code
    """
    ensure_solutions_dir()

    # Handle full path or just filename
    if os.path.isabs(filename):
        filepath = Path(filename)
    else:
        if not filename.endswith(".py"):
            filename = f"{filename}.py"
        filepath = SOLUTIONS_DIR / filename

    if not filepath.exists():
        return ExecutionResult(
            success=False,
            stdout="",
            stderr=f"File not found: {filepath}",
            return_code=-1,
            filename=str(filepath)
        )

    if show_ui:
        console.print(Panel(
            f"[yellow]Running:[/yellow] {filepath}",
            title="[bold cyan]Executing Code[/bold cyan]",
            border_style="yellow"
        ))

    try:
        # Run the Python file
        result = subprocess.run(
            [sys.executable, str(filepath)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(SOLUTIONS_DIR)
        )

        success = result.returncode == 0 and not result.stderr

        execution_result = ExecutionResult(
            success=success,
            stdout=result.stdout,
            stderr=result.stderr,
            return_code=result.returncode,
            filename=str(filepath)
        )

        if show_ui:
            if success:
                console.print(Panel(
                    f"[green]Success![/green]\n\n{result.stdout}" if result.stdout else "[green]Success! (No output)[/green]",
                    title="[bold green]Execution Result[/bold green]",
                    border_style="green"
                ))
            else:
                error_msg = result.stderr or f"Process exited with code {result.returncode}"
                console.print(Panel(
                    f"[red]Error:[/red]\n{error_msg}",
                    title="[bold red]Execution Failed[/bold red]",
                    border_style="red"
                ))
                if result.stdout:
                    console.print(Panel(
                        result.stdout,
                        title="[bold]Standard Output[/bold]",
                        border_style="cyan"
                    ))

        return execution_result

    except subprocess.TimeoutExpired:
        return ExecutionResult(
            success=False,
            stdout="",
            stderr=f"Execution timed out after {timeout} seconds",
            return_code=-1,
            filename=str(filepath)
        )
    except Exception as e:
        return ExecutionResult(
            success=False,
            stdout="",
            stderr=str(e),
            return_code=-1,
            filename=str(filepath)
        )


def execute_shell(command: str, timeout: int = 60, show_ui: bool = True) -> ExecutionResult:
    """
    Execute a shell command.

    Args:
        command: Shell command to execute
        timeout: Maximum execution time in seconds
        show_ui: Whether to display UI feedback

    Returns:
        ExecutionResult with command output
    """
    if show_ui:
        console.print(Panel(
            f"[yellow]$ {command}[/yellow]",
            title="[bold cyan]Shell Command[/bold cyan]",
            border_style="yellow"
        ))

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )

        success = result.returncode == 0

        if show_ui:
            if result.stdout:
                console.print(Panel(
                    result.stdout,
                    title="[bold]Output[/bold]",
                    border_style="green" if success else "red"
                ))
            if result.stderr:
                console.print(Panel(
                    result.stderr,
                    title="[bold red]Errors[/bold red]",
                    border_style="red"
                ))

        return ExecutionResult(
            success=success,
            stdout=result.stdout,
            stderr=result.stderr,
            return_code=result.returncode,
            filename=command
        )

    except subprocess.TimeoutExpired:
        return ExecutionResult(
            success=False,
            stdout="",
            stderr=f"Command timed out after {timeout} seconds",
            return_code=-1,
            filename=command
        )
    except Exception as e:
        return ExecutionResult(
            success=False,
            stdout="",
            stderr=str(e),
            return_code=-1,
            filename=command
        )


def read_file(filepath: str) -> Optional[str]:
    """
    Read content from a file.

    Args:
        filepath: Path to the file

    Returns:
        File content or None if file doesn't exist
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return None
    except Exception as e:
        console.print(f"[red]Error reading file: {e}[/red]")
        return None


def list_solutions() -> list[str]:
    """
    List all solution files in the solutions directory.

    Returns:
        List of solution filenames
    """
    ensure_solutions_dir()
    return [f.name for f in SOLUTIONS_DIR.glob("*.py")]


# Tool definitions for Gemini function calling
FILE_TOOLS = [
    {
        "name": "save_solution",
        "description": "Save a Python code solution to a file. The code MUST include 'if __name__ == \"__main__\":' block with assert statements for testing.",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Name of the file (e.g., 'two_sum.py')"
                },
                "code": {
                    "type": "string",
                    "description": "The Python code to save. Must include test cases with assert statements."
                }
            },
            "required": ["filename", "code"]
        }
    },
    {
        "name": "run_solution",
        "description": "Run a Python solution file and capture the output. Use this after saving a solution to test it.",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Name of the file to run (e.g., 'two_sum.py')"
                }
            },
            "required": ["filename"]
        }
    }
]


if __name__ == "__main__":
    # Test the file tools
    test_code = '''
def two_sum(nums, target):
    """Find two numbers that add up to target."""
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []

if __name__ == "__main__":
    # Test cases
    assert two_sum([2, 7, 11, 15], 9) == [0, 1], "Test case 1 failed"
    assert two_sum([3, 2, 4], 6) == [1, 2], "Test case 2 failed"
    assert two_sum([3, 3], 6) == [0, 1], "Test case 3 failed"
    print("All test cases passed!")
'''

    print("Testing file_tools module...")
    filepath = save_solution("test_two_sum.py", test_code)
    result = run_solution("test_two_sum.py")
    print(f"Success: {result.success}")
