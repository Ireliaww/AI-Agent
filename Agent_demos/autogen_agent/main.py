"""
AutoGen Agent 示例 - 双 Agent 对话（内存型）
使用 AutoGen 0.4+ 新版 API 实现两个 Agent 之间的协作对话
"""

import os
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.anthropic import AnthropicChatCompletionClient

async def main():
    # 检查 API Key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("错误: 请设置 ANTHROPIC_API_KEY 环境变量")
        return

    print("=" * 50)
    print("AutoGen Agent - 双 Agent 协作")
    print("=" * 50)

    # 创建 Anthropic 模型客户端
    model_client = AnthropicChatCompletionClient(
        model="claude-sonnet-4-20250514",
        api_key=api_key
    )

    # 创建助手 Agent
    assistant = AssistantAgent(
        name="assistant",
        model_client=model_client,
        system_message="""你是一个知识渊博的助手。
你的任务是回答用户的问题，提供准确、有帮助的信息。
回答要简洁明了，用中文回答。
回答完成后请在最后说 "TERMINATE" 表示结束。"""
    )

    # 终止条件
    termination = TextMentionTermination("TERMINATE")

    # 测试问题
    test_questions = [
        "请用一句话解释什么是人工智能？",
        "Python 和 JavaScript 的主要区别是什么？",
        "给我讲一个关于编程的小笑话。"
    ]

    print("\n开始测试对话...\n")

    for question in test_questions:
        print(f"问题: {question}")
        print("-" * 40)

        # 创建团队（单个 agent 的简单对话）
        team = RoundRobinGroupChat(
            participants=[assistant],
            termination_condition=termination
        )

        # 运行对话
        result = await team.run(task=question)

        # 打印结果
        for message in result.messages:
            if hasattr(message, 'content') and message.content:
                content = message.content
                # 移除 TERMINATE 标记以便更清晰显示
                if isinstance(content, str):
                    content = content.replace("TERMINATE", "").strip()
                    if content:
                        print(f"Agent: {content}")

        print("=" * 50)
        print()

    print("✅ AutoGen Agent 测试完成！")

if __name__ == "__main__":
    asyncio.run(main())
