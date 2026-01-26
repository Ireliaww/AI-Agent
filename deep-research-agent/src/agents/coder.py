"""
Coding Agent Module - LeetCode problem solver with self-healing capabilities.

This agent implements a "Code - Run - Fix" loop that:
1. Generates code for a given problem
2. Runs the code with test cases
3. Automatically debugs and fixes errors (up to 3 retries)
"""

import asyncio
import re
from typing import Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..client import GeminiClient
from ..tools.file_tools import save_solution, run_solution, ExecutionResult

console = Console()


class CodingState(Enum):
    """States for the coding workflow."""
    IDLE = "idle"
    GENERATING = "generating"
    RUNNING = "running"
    DEBUGGING = "debugging"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class CodingContext:
    """Context for tracking coding progress."""
    problem: str
    filename: str = ""
    code: str = ""
    state: CodingState = CodingState.IDLE
    attempts: int = 0
    max_attempts: int = 3
    last_error: str = ""
    execution_results: list[ExecutionResult] = field(default_factory=list)
    success: bool = False


# System prompt for the Coding Agent
CODING_AGENT_SYSTEM = """You are an expert Python programmer specializing in algorithm problems (LeetCode-style).

When given a programming problem, you will:
1. Analyze the problem carefully
2. Write clean, efficient Python code
3. Include comprehensive test cases

CRITICAL REQUIREMENTS:
- Your solution MUST include an `if __name__ == "__main__":` block
- The main block MUST contain assert statements as test cases
- Test cases should cover: basic cases, edge cases, and examples from the problem
- Print "All test cases passed!" at the end if all asserts pass

Example structure:
```python
def solution(args):
    # Your implementation
    pass

if __name__ == "__main__":
    # Test cases with assert statements
    assert solution(test_input) == expected_output, "Test case 1 failed"
    assert solution(edge_case) == expected_output, "Edge case failed"
    print("All test cases passed!")
```

When debugging:
- Carefully read the error message
- Identify the root cause
- Fix the issue while maintaining correctness
- Do NOT remove test cases just to make tests pass
"""

GENERATE_CODE_PROMPT = """Problem:
{problem}

Generate a complete Python solution with test cases.
Remember:
1. Include `if __name__ == "__main__":` block
2. Add assert statements for testing
3. Include edge cases
4. Print success message at the end

Respond with ONLY the Python code, no explanations.
"""

DEBUG_CODE_PROMPT = """The previous code attempt failed with the following error:

Error:
{error}

Previous Code:
```python
{code}
```

Problem Description:
{problem}

Please analyze the error and provide a corrected version of the code.
Keep all test cases (do not remove tests to make it pass).
Respond with ONLY the corrected Python code.
"""


class CodingAgent:
    """
    Agent for solving coding problems with automatic debugging.

    Implements a Code-Run-Fix loop:
    1. Generate code based on problem description
    2. Run the code and capture output
    3. If errors occur, analyze and fix (up to max_attempts)
    """

    def __init__(
        self,
        gemini_client: Optional[GeminiClient] = None,
        on_state_change: Optional[Callable[[CodingState, str], None]] = None,
    ):
        """
        Initialize the Coding Agent.

        Args:
            gemini_client: Optional pre-configured Gemini client
            on_state_change: Callback for state change notifications
        """
        self.gemini = gemini_client or GeminiClient()
        self.on_state_change = on_state_change

    def _notify_state_change(self, state: CodingState, message: str) -> None:
        """Notify observers of state change."""
        if self.on_state_change:
            self.on_state_change(state, message)

    def _extract_code(self, response_text: str) -> str:
        """Extract Python code from response (handles markdown code blocks)."""
        text = response_text.strip()

        # Try to extract from markdown code block
        code_block_pattern = r"```(?:python)?\n?(.*?)```"
        matches = re.findall(code_block_pattern, text, re.DOTALL)

        if matches:
            return matches[0].strip()

        # If no code block, assume entire response is code
        return text

    def _generate_filename(self, problem: str) -> str:
        """Generate a filename from the problem description."""
        # Extract first few words and convert to snake_case
        words = re.findall(r'\w+', problem.lower())[:5]
        filename = "_".join(words)

        # Ensure valid filename
        filename = re.sub(r'[^\w_]', '', filename)
        if not filename:
            filename = "solution"

        return f"{filename}.py"

    async def _generate_code(self, context: CodingContext) -> str:
        """Generate code for the problem."""
        self._notify_state_change(
            CodingState.GENERATING,
            "Generating solution code..."
        )

        prompt = GENERATE_CODE_PROMPT.format(problem=context.problem)

        response = await self.gemini.generate_content(
            contents=prompt,
            system_instruction=CODING_AGENT_SYSTEM,
            temperature=0.3,
        )

        return self._extract_code(response.text)

    async def _debug_code(self, context: CodingContext) -> str:
        """Debug and fix the code based on error."""
        self._notify_state_change(
            CodingState.DEBUGGING,
            f"Analyzing error and fixing (attempt {context.attempts}/{context.max_attempts})..."
        )

        prompt = DEBUG_CODE_PROMPT.format(
            error=context.last_error,
            code=context.code,
            problem=context.problem,
        )

        response = await self.gemini.generate_content(
            contents=prompt,
            system_instruction=CODING_AGENT_SYSTEM,
            temperature=0.2,  # Lower temperature for debugging
        )

        return self._extract_code(response.text)

    async def solve(self, problem: str, filename: Optional[str] = None) -> CodingContext:
        """
        Solve a coding problem with automatic debugging.

        Args:
            problem: The problem description
            filename: Optional filename for the solution

        Returns:
            CodingContext with results
        """
        context = CodingContext(
            problem=problem,
            filename=filename or self._generate_filename(problem),
        )

        try:
            # Initial code generation
            context.state = CodingState.GENERATING
            context.code = await self._generate_code(context)
            context.attempts = 1

            # Code-Run-Fix loop
            while context.attempts <= context.max_attempts:
                # Save the code
                save_solution(context.filename, context.code, show_ui=True)

                # Run the code
                context.state = CodingState.RUNNING
                self._notify_state_change(
                    CodingState.RUNNING,
                    f"Running solution (attempt {context.attempts}/{context.max_attempts})..."
                )

                result = run_solution(context.filename, show_ui=True)
                context.execution_results.append(result)

                # Check if successful
                if result.success:
                    context.state = CodingState.COMPLETED
                    context.success = True
                    self._notify_state_change(
                        CodingState.COMPLETED,
                        "Solution passed all tests!"
                    )
                    break

                # Store error for debugging
                context.last_error = result.stderr or f"Exit code: {result.return_code}\n{result.stdout}"

                # Check if we have attempts left
                if context.attempts >= context.max_attempts:
                    context.state = CodingState.FAILED
                    self._notify_state_change(
                        CodingState.FAILED,
                        f"Failed after {context.max_attempts} attempts"
                    )
                    break

                # Debug and fix
                context.state = CodingState.DEBUGGING
                context.code = await self._debug_code(context)
                context.attempts += 1

        except Exception as e:
            context.state = CodingState.FAILED
            context.last_error = str(e)
            self._notify_state_change(
                CodingState.FAILED,
                f"Error: {e}"
            )

        return context


async def solve_problem(
    problem: str,
    filename: Optional[str] = None,
    verbose: bool = True,
) -> CodingContext:
    """
    Convenience function to solve a coding problem.

    Args:
        problem: The problem description
        filename: Optional filename for the solution
        verbose: Print progress updates

    Returns:
        CodingContext with results
    """
    def on_state_change(state: CodingState, message: str):
        if verbose:
            style_map = {
                CodingState.GENERATING: "cyan",
                CodingState.RUNNING: "yellow",
                CodingState.DEBUGGING: "magenta",
                CodingState.COMPLETED: "green",
                CodingState.FAILED: "red",
            }
            style = style_map.get(state, "white")
            console.print(f"[{style}][{state.value.upper()}][/{style}] {message}")

    agent = CodingAgent(on_state_change=on_state_change)
    return await agent.solve(problem, filename)


async def run_coding_agent(problem: str, filename: Optional[str] = None) -> str:
    """
    Main entry point for the coding agent.

    Args:
        problem: The problem description
        filename: Optional filename for the solution

    Returns:
        Markdown-formatted result summary
    """
    console.print(Panel(
        f"[bold cyan]Problem:[/bold cyan]\n{problem}",
        title="[bold]Coding Agent[/bold]",
        border_style="cyan"
    ))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=False,
    ) as progress:
        task_id = progress.add_task("[cyan]Initializing...", total=None)

        def on_state_change(state: CodingState, message: str):
            style_map = {
                CodingState.GENERATING: "cyan",
                CodingState.RUNNING: "yellow",
                CodingState.DEBUGGING: "magenta",
                CodingState.COMPLETED: "green",
                CodingState.FAILED: "red",
            }
            style = style_map.get(state, "white")
            progress.update(task_id, description=f"[{style}]{message}[/{style}]")

        agent = CodingAgent(on_state_change=on_state_change)
        context = await agent.solve(problem, filename)
        progress.update(task_id, visible=False)

    # Generate result summary
    if context.success:
        result = f"""## Solution Successful!

**File:** `{context.filename}`
**Attempts:** {context.attempts}

### Final Code:
```python
{context.code}
```

All test cases passed!
"""
    else:
        result = f"""## Solution Failed

**File:** `{context.filename}`
**Attempts:** {context.attempts}/{context.max_attempts}

### Last Error:
```
{context.last_error}
```

### Last Code Attempt:
```python
{context.code}
```
"""

    return result


if __name__ == "__main__":
    # Test the coding agent
    test_problem = """
    LeetCode 1. Two Sum

    Given an array of integers nums and an integer target, return indices of the
    two numbers such that they add up to target.

    You may assume that each input would have exactly one solution, and you may
    not use the same element twice.

    Example 1:
    Input: nums = [2,7,11,15], target = 9
    Output: [0,1]
    Explanation: Because nums[0] + nums[1] == 9, we return [0, 1].

    Example 2:
    Input: nums = [3,2,4], target = 6
    Output: [1,2]

    Example 3:
    Input: nums = [3,3], target = 6
    Output: [0,1]
    """

    async def test():
        result = await run_coding_agent(test_problem, "two_sum.py")
        console.print(Markdown(result))

    asyncio.run(test())
