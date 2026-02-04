"""
Test suite for SafeCodeExecutor
"""

import pytest
import time
from tools.code_executor import SafeCodeExecutor, ExecutionResult


def test_basic_execution():
    """测试基本代码执行"""
    executor = SafeCodeExecutor(require_approval=False, auto_create_venv=False)
    result = executor.execute_code("print('Hello, World!')")
    
    assert result.success
    assert "Hello, World!" in result.output
    assert result.error is None


def test_execution_with_output():
    """测试带输出的代码执行"""
    executor = SafeCodeExecutor(require_approval=False, auto_create_venv=False)
    code = """
for i in range(3):
    print(f"Count: {i}")
"""
    result = executor.execute_code(code)
    
    assert result.success
    assert "Count: 0" in result.output
    assert "Count: 1" in result.output
    assert "Count: 2" in result.output


def test_execution_with_error():
    """测试错误代码执行"""
    executor = SafeCodeExecutor(require_approval=False, auto_create_venv=False)
    result = executor.execute_code("print(undefined_variable)")
    
    assert not result.success
    assert result.error is not None
    assert "NameError" in result.error


def test_timeout():
    """测试超时保护"""
    executor = SafeCodeExecutor(require_approval=False, auto_create_venv=False)
    code = """
import time
time.sleep(10)
print("This should not print")
"""
    
    result = executor.execute_code(code, timeout=1)
    
    assert not result.success
    assert "timeout" in result.error.lower()


def test_execution_time_tracking():
    """测试执行时间跟踪"""
    executor = SafeCodeExecutor(require_approval=False, auto_create_venv=False)
    code = """
import time
time.sleep(0.5)
print("Done")
"""
    
    result = executor.execute_code(code)
    
    assert result.success
    assert result.execution_time >= 0.5
    assert result.execution_time < 1.0


def test_user_rejection():
    """测试用户拒绝执行"""
    executor = SafeCodeExecutor(require_approval=True, auto_create_venv=False)
    # 注意：这个测试在自动化环境中无法运行，因为需要用户输入
    # 仅作为示例
    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
