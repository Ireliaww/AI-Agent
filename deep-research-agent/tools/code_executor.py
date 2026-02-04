"""
Safe Code Executor - 安全的代码执行器

使用conda虚拟环境隔离执行用户代码，提供用户确认机制和超时保护。
"""

import subprocess
import tempfile
import os
import time
from typing import Optional
from dataclasses import dataclass
from rich.console import Console
from rich.syntax import Syntax
from rich.panel import Panel

console = Console()

@dataclass
class ExecutionResult:
    """代码执行结果"""
    success: bool
    output: str
    error: Optional[str] = None
    execution_time: float = 0.0


class SafeCodeExecutor:
    """
    安全的代码执行器 - 使用conda虚拟环境
    
    特点：
    - 在独立的conda环境中执行代码
    - 用户确认机制（可选）
    - 超时保护
    - 美观的代码展示（使用rich）
    """
    
    def __init__(
        self, 
        venv_path: str = "./code_exec_venv",
        require_approval: bool = True,
        auto_create_venv: bool = True
    ):
        self.venv_path = venv_path
        self.require_approval = require_approval
        
        if auto_create_venv:
            self._ensure_venv()
    
    def _ensure_venv(self):
        """确保虚拟环境存在"""
        if not os.path.exists(self.venv_path):
            console.print("[yellow]Creating isolated conda environment...[/yellow]")
            result = subprocess.run([
                "conda", "create", "-p", self.venv_path,
                "python=3.11", "-y"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                console.print("[green]✓ Conda environment created[/green]")
            else:
                console.print(f"[red]Failed to create conda environment: {result.stderr}[/red]")
    
    def execute_code(
        self, 
        code: str, 
        language: str = "python",
        timeout: int = 30
    ) -> ExecutionResult:
        """
        在隔离环境中执行代码
        
        Args:
            code: 要执行的代码
            language: 编程语言（目前仅支持python）
            timeout: 超时时间（秒）
        
        Returns:
            ExecutionResult包含执行结果
        """
        # 1. 用户确认机制
        if self.require_approval:
            if not self._request_approval(code, language):
                return ExecutionResult(
                    success=False,
                    output="",
                    error="User rejected code execution"
                )
        
        # 2. 在临时文件中写入代码
        with tempfile.NamedTemporaryFile(
            mode='w', 
            suffix='.py', 
            delete=False
        ) as f:
            f.write(code)
            temp_file = f.name
        
        try:
            # 3. 在虚拟环境中执行
            python_exe = os.path.join(
                self.venv_path, 
                "bin", 
                "python"
            )
            
            console.print("[cyan]⚡ Executing code in isolated environment...[/cyan]")
            
            start_time = time.time()
            result = subprocess.run(
                [python_exe, temp_file],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            execution_time = time.time() - start_time
            
            return ExecutionResult(
                success=result.returncode == 0,
                output=result.stdout,
                error=result.stderr if result.returncode != 0 else None,
                execution_time=execution_time
            )
            
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                output="",
                error=f"Execution timeout after {timeout} seconds"
            )
        except FileNotFoundError:
            return ExecutionResult(
                success=False,
                output="",
                error=f"Python executable not found at {python_exe}. Please ensure conda environment is created."
            )
        finally:
            # 清理临时文件
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def _request_approval(self, code: str, language: str = "python") -> bool:
        """请求用户批准执行代码（带语法高亮）"""
        syntax = Syntax(code, language, theme="monokai", line_numbers=True)
        
        console.print("\n")
        console.print(Panel(
            syntax,
            title="[bold red]🔒 CODE EXECUTION REQUEST[/bold red]",
            border_style="red"
        ))
        
        response = console.input("\n[bold yellow]Allow execution? (yes/no):[/bold yellow] ").strip().lower()
        return response in ['yes', 'y']


if __name__ == "__main__":
    # 测试代码执行器
    executor = SafeCodeExecutor(require_approval=False)
    
    test_code = """
print("Hello from safe executor!")
for i in range(3):
    print(f"Count: {i}")
"""
    
    result = executor.execute_code(test_code)
    
    if result.success:
        console.print(f"\n[green]✓ Execution successful ({result.execution_time:.2f}s)[/green]")
        console.print(f"[cyan]Output:[/cyan]\n{result.output}")
    else:
        console.print(f"\n[red]✗ Execution failed[/red]")
        console.print(f"[red]Error:[/red]\n{result.error}")
