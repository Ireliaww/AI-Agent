"""
LangChain Agent 示例 - 使用 Anthropic Claude（内存型）
基于 LangChain 官方文档的快速入门示例
"""

import os
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

# 定义工具
@tool
def calculate(expression: str) -> str:
    """计算数学表达式，支持加减乘除。"""
    try:
        allowed_chars = set("0123456789+-*/(). ")
        if not all(c in allowed_chars for c in expression):
            return "错误: 表达式包含非法字符"
        result = eval(expression)
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"

@tool
def get_greeting(name: str) -> str:
    """根据名字生成问候语。"""
    return f"你好，{name}！很高兴认识你！"

def main():
    # 检查 API Key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("错误: 请设置 ANTHROPIC_API_KEY 环境变量")
        return

    print("=" * 50)
    print("LangChain Agent - Claude 对话助手")
    print("=" * 50)

    # 初始化 Claude 模型
    model = init_chat_model(
        "claude-sonnet-4-20250514",
        model_provider="anthropic",
        temperature=0.7,
        max_tokens=1024
    )

    # 内存保存器（对话历史）
    memory = MemorySaver()

    # 创建 ReAct Agent
    agent = create_react_agent(
        model,
        tools=[calculate, get_greeting],
        checkpointer=memory
    )

    # 配置（用于追踪对话线程）
    config = {"configurable": {"thread_id": "test-thread-1"}}

    # 测试对话
    test_questions = [
        "你好！请用 get_greeting 工具跟小明打个招呼",
        "请帮我计算 (100 + 50) * 2 等于多少？",
        "再算一下 1024 / 4 + 100"
    ]

    print("\n开始测试对话...\n")

    for question in test_questions:
        print(f"用户: {question}")
        print("-" * 40)

        response = agent.invoke(
            {"messages": [{"role": "user", "content": question}]},
            config=config
        )

        # 获取最后一条助手消息
        last_message = response["messages"][-1]
        print(f"Agent: {last_message.content}")
        print("=" * 50)

    print("\n✅ LangChain Agent 测试完成！")

if __name__ == "__main__":
    main()
