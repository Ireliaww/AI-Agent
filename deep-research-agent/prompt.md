请作为一名高级 AI 架构师，帮我将当前的 AI Agent 项目重构并升级为一个 "Multi-Mode AI Assistant"。

该助手需要具备两个核心能力：Deep Research (深度调研) 和 Auto Coding (自动编程/刷题)，并由一个智能 Router (路由) 统一分发任务。

### 1. 核心技术栈 (Mandatory)
- SDK: google-genai (最新版官方 SDK)。
- Model: gemini-2.5-flash (如果不可用，降级为 gemini-1.5-pro)。
- UI/CLI: 使用 rich 库实现高亮、Loading 动画和 Markdown 渲染；使用 questionary 实现交互菜单。
- 环境管理: 检查并更新 requirements.txt 和 venv。

### 2. 架构重构 (Refactoring)
请将项目结构重构为以下模块化形式：
- main.py: 统一入口 (CLI Loop)
- .env: 环境变量
- src/router.py: 意图识别模块
- src/client.py: Gemini Client 封装
- src/tools/file_tools.py: 文件读写、Shell 执行工具
- src/tools/search_tools.py: Google Search MCP/Tool 封装
- src/agents/researcher.py: 现有的 Research Agent 逻辑
- src/agents/coder.py: 新增的 Coding Agent 逻辑

### 3. 功能模块详情

#### A. 智能路由 (src/router.py)
- 实现 classify_intent(query: str) -> str。
- 使用 Gemini 判断用户意图。
- 规则: 涉及代码编写、LeetCode、Debug 的归为 CODING；涉及信息查询、新闻、报告的归为 RESEARCH。

#### B. 新增：Coding Agent (src/agents/coder.py)
实现一个针对 LeetCode 问题的 "Code - Run - Fix" 自愈闭环：
1. 工具定义 (src/tools/file_tools.py):
   - save_solution(filename, code): 保存代码到 solutions/ 目录。必须要求 Gemini 在代码中包含 if __name__ == "__main__": 和 assert 测试用例。
   - run_solution(filename): 使用 subprocess 运行代码，捕获 stdout 和 stderr。
2. 核心循环:
   - Step 1: 根据题目生成代码和测试用例 -> save_solution。
   - Step 2: 自动 run_solution。
   - Step 3 (Debug): 如果 stderr 不为空或 return_code != 0，Gemini 必须阅读报错信息，分析原因，重写代码并再次保存运行。允许最多重试 3 次。
3. UI 反馈: 每一步操作（写文件、运行、修复）都要用 rich 打印出状态。

#### C. 现有：Research Agent (src/agents/researcher.py)
- 迁移你之前写的 Deep Research 逻辑到这里。
- 确保它继续使用 Google Search MCP/Tool。

### 4. 统一入口与 UI (main.py)
实现一个类似 Claude Code 的 REPL 循环：
1. 启动时显示 ASCII Logo。
2. 用户输入 Query。
3. 调用 router.classify_intent。
4. 使用 rich.status 显示 "Routing to [Agent Name]..."。
5. 动态加载对应的 Agent 模块并执行任务。
6. 使用 rich.markdown 渲染最终结果（代码块需高亮）。

### 执行计划
1. 先整理文件结构，创建文件夹。
2. 安装 rich, questionary 等新依赖。
3. 编写 file_tools.py 和 coder.py (这是新功能，优先实现)。
4. 编写 router.py 和 main.py 将所有功能串联起来。

请开始执行，先从更新 requirements.txt 和创建目录结构开始。