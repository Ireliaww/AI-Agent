"""
Anthropic Agent 示例 - 带工具调用的 Agent（内存型）
使用 Anthropic Claude 的 Tool Use 实现简单的计算器 Agent
"""

import os
import anthropic

# 定义工具
tools = [
    {
        "name": "calculate",
        "description": "计算数学表达式，支持加减乘除",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "数学表达式，如 '2+3*4'"
                }
            },
            "required": ["expression"]
        }
    },
    {
        "name": "get_current_time",
        "description": "获取当前系统时间",
        "input_schema": {
            "type": "object",
            "properties": {}
        }
    }
]

def calculate(expression: str) -> str:
    """计算数学表达式"""
    try:
        allowed_chars = set("0123456789+-*/(). ")
        if not all(c in allowed_chars for c in expression):
            return "错误: 表达式包含非法字符"
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"计算错误: {str(e)}"

def get_current_time() -> str:
    """获取当前时间"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 工具映射
tool_functions = {
    "calculate": calculate,
    "get_current_time": get_current_time
}

def run_agent(client, messages):
    """运行 Agent 并处理工具调用"""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system="你是一个有用的助手，可以进行数学计算和查询时间。请用中文回答。",
        tools=tools,
        messages=messages
    )

    # 检查是否需要调用工具
    if response.stop_reason == "tool_use":
        # 收集所有工具调用结果
        tool_results = []
        assistant_content = response.content

        for block in response.content:
            if block.type == "tool_use":
                tool_name = block.name
                tool_input = block.input

                print(f"  [调用工具] {tool_name}({tool_input})")

                # 执行工具
                if tool_name in tool_functions:
                    if tool_input:
                        result = tool_functions[tool_name](**tool_input)
                    else:
                        result = tool_functions[tool_name]()
                else:
                    result = f"未知工具: {tool_name}"

                print(f"  [工具结果] {result}")

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result
                })

        # 添加助手消息和工具结果
        messages.append({"role": "assistant", "content": assistant_content})
        messages.append({"role": "user", "content": tool_results})

        # 获取最终回复
        final_response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system="你是一个有用的助手，可以进行数学计算和查询时间。请用中文回答。",
            tools=tools,
            messages=messages
        )

        return final_response.content[0].text

    return response.content[0].text

def main():
    # 检查 API Key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("错误: 请设置 ANTHROPIC_API_KEY 环境变量")
        return

    print("=" * 50)
    print("Anthropic Agent - 计算器助手")
    print("=" * 50)

    client = anthropic.Anthropic(api_key=api_key)

    # 测试问题
    test_questions = [
        "请帮我计算 (100 + 50) * 2 等于多少？",
        "现在几点了？",
        "如果我有 1000 元，花掉 350 元，还剩多少？"
    ]

    print("\n开始测试对话...\n")

    for question in test_questions:
        print(f"用户: {question}")
        # 每个问题单独的对话历史
        messages = [{"role": "user", "content": question}]

        response = run_agent(client, messages)
        print(f"Agent: {response}")
        print("-" * 40)

    print("\n✅ Anthropic Agent 测试完成！")

if __name__ == "__main__":
    main()
