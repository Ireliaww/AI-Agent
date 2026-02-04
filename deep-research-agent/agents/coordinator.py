"""
Coordinator Agent - 多agent协作编排器

职责：
- 分析任务类型
- 智能分发到合适的agent
- 编排多个agents完成复杂任务
- 支持论文复现workflow (Research → Coding)
"""

from typing import Optional
import asyncio
from rich.console import Console

console = Console()


class CoordinatorAgent:
    """
    协调多个agents的执行
    
    可以处理：
    - 纯研究任务（调用ResearchAgent）
    - 纯编程任务（调用CodingAgent）
    - 复杂任务（先研究再编程）
    """
    
    def __init__(self, research_agent, coding_agent, gemini_client):
        """
        初始化Coordinator
        
        Args:
            research_agent: ResearchAgent实例
            coding_agent: CodingAgent实例
            gemini_client: GeminiClient实例
        """
        self.research_agent = research_agent
        self.coding_agent = coding_agent
        self.gemini = gemini_client
    
    async def process_request(self, user_request: str) -> str:
        """
        智能分发或协作处理请求
        
        Args:
            user_request: 用户请求
            
        Returns:
            处理结果（markdown格式）
        """
        console.print("[cyan]🤖 Coordinator analyzing task...[/cyan]")
        
        # 1. 分析任务类型
        task_type = await self._analyze_task(user_request)
        console.print(f"[cyan]📋 Task type: {task_type}[/cyan]")
        
        if task_type == "paper_reproduction":
            # 论文复现：Enhanced Research Agent → Enhanced Coding Agent
            return await self._handle_paper_reproduction(user_request)
            
        elif task_type == "research_then_code":
            # 复杂任务：先研究，再编码
            return await self._handle_research_then_code(user_request)
            
        elif task_type == "research_only":
            console.print("[yellow]→ Routing to Research Agent[/yellow]")
            result = await self.research_agent.research(user_request)
            return result.report
            
        elif task_type == "coding_only":
            console.print("[yellow]→ Routing to Coding Agent[/yellow]")
            from src.agents.coder import run_coding_agent
            return await run_coding_agent(user_request)
        
        else:
            # 默认：使用简单路由
            return await self._simple_route(user_request)
    
    async def _handle_research_then_code(self, user_request: str) -> str:
        """
        处理需要先研究再编码的复杂任务
        
        Args:
            user_request: 用户请求
            
        Returns:
            组合的研究+代码结果
        """
        console.print("[yellow]→ Step 1: Researching topic...[/yellow]")
        research_result = await self.research_agent.research(user_request)
        
        console.print("[yellow]→ Step 2: Writing code based on research...[/yellow]")
        
        # 将研究结果传递给coding agent
        coding_request = f"""Based on this research about {user_request}:

{research_result.report}

Please write clean, well-documented code to solve this problem. Include:
1. Clear implementation
2. Test cases
3. Usage examples
"""
        
        from src.agents.coder import run_coding_agent
        code_result = await run_coding_agent(coding_request)
        
        # 组合结果
        combined_result = f"""# Research Findings

{research_result.report}

---

# Code Implementation

{code_result}
"""
        
        console.print("[green]✓ Multi-agent task completed successfully[/green]")
        return combined_result
    
    async def _handle_paper_reproduction(self, user_request: str) -> str:
        """
        处理论文复现任务 (Enhanced Research Agent → Enhanced Coding Agent)
        
        Args:
            user_request: 用户请求（包含论文信息）
            
        Returns:
            完整的论文分析+代码实现
        """
        console.print("[yellow]→ Step 1: Analyzing paper (Enhanced Research Agent)...[/yellow]")
        
        # Use Enhanced Research Agent if available
        if hasattr(self.research_agent, 'analyze_paper'):
            # Extract paper identifier from request
            paper_analysis = await self.research_agent.analyze_paper(
                user_request,
                create_index=True,
                deep_analysis=True
            )
            
            console.print("[yellow]→ Step 2: Implementing paper (Enhanced Coding Agent)...[/yellow]")
            
            # Use Enhanced Coding Agent if available
            if hasattr(self.coding_agent, 'implement_from_paper'):
                implementation = await self.coding_agent.implement_from_paper(
                    paper_analysis,
                    framework="pytorch"  # Default to PyTorch
                )
                
                # Format result
                result = f"""# Paper Reproduction Complete

## Paper Analysis

**Title:** {paper_analysis.content.title}
**Authors:** {', '.join(paper_analysis.content.authors[:5])}

### Abstract
{paper_analysis.content.abstract}

### Understanding
{paper_analysis.understanding.get_summary()}

### Related Work
- Found {len(paper_analysis.related_papers)} related papers
- Found {len(paper_analysis.implementations)} existing implementations

---

## Implementation

**Project Name:** {implementation.project.name}
**Framework:** {implementation.project.framework}
**Files Generated:** {len(implementation.project.files)}

### Project Structure
```
{implementation.project.get_tree()}
```

### README
{implementation.readme[:1000]}...

✅ **Complete implementation saved!**
"""
                
                console.print("[green]✓ Paper reproduction completed successfully[/green]")
                return result
        
        # Fallback to regular workflow
        return await self._handle_research_then_code(user_request)
    
    async def _analyze_task(self, request: str) -> str:
        """
        使用LLM分析任务需要哪些agents
        
        Args:
            request: 用户请求
            
        Returns:
            任务类型 (paper_reproduction, research_only, coding_only, research_then_code)
        """
        prompt = f"""Analyze this request and classify it into ONE of these categories:
- "paper_reproduction": reproduce/implement a research paper (e.g., "implement BERT", "reproduce ResNet", "code the Transformer paper")
- "research_only": needs only research/information gathering (e.g., "explain X", "what is Y", "compare A and B")
- "coding_only": needs only coding/programming (e.g., "write a function to X", "implement Y algorithm", "LeetCode problem")
- "research_then_code": needs research first, then coding based on findings (e.g., "research X and write code for it", "learn about Y then implement it")

Request: {request}

Respond with ONLY one word: the classification.
Do not include any explanation, just the classification.
"""
        
        try:
            response = await self.gemini.generate_content(prompt)
            classification = response.text.strip().lower()
            
            # Validate
            valid_types = ["paper_reproduction", "research_only", "coding_only", "research_then_code"]
            if classification in valid_types:
                return classification
            
            # Fallback: try to infer from keywords
            return self._infer_from_keywords(request)
            
        except Exception as e:
            console.print(f"[yellow]Warning: LLM classification failed ({e}), using keyword fallback[/yellow]")
            return self._infer_from_keywords(request)
    
    def _infer_from_keywords(self, request: str) -> str:
        """
        基于关键词推断任务类型（备用方法）
        
        Args:
            request: 用户请求
            
        Returns:
            任务类型
        """
        request_lower = request.lower()
        
        # 论文复现关键词
        paper_keywords = ["paper", "reproduce", "bert", "transformer", "resnet", "attention", "论文"]
        # 编程关键词
        coding_keywords = ["code", "write", "implement", "function", "class", "leetcode", "algorithm", "program"]
        # 研究关键词
        research_keywords = ["research", "explain", "what", "who", "why", "how", "compare", "analyze", "tell me about"]
        # 复合任务关键词
        combined_keywords = ["and", "then", "after", "based on"]
        
        has_paper = any(keyword in request_lower for keyword in paper_keywords)
        has_coding = any(keyword in request_lower for keyword in coding_keywords)
        has_research = any(keyword in request_lower for keyword in research_keywords)
        has_combined = any(keyword in request_lower for keyword in combined_keywords)
        
        # Paper reproduction has priority
        if has_paper and (has_coding or "implement" in request_lower or "reproduce" in request_lower):
            return "paper_reproduction"
        elif has_coding and has_research and has_combined:
            return "research_then_code"
        elif has_coding:
            return "coding_only"
        elif has_research:
            return "research_only"
        else:
            # 默认为研究任务
            return "research_only"
    
    async def _simple_route(self, request: str) ->str:
        """
        简单路由逻辑（最后兜底）
        
        Args:
            request: 用户请求
            
        Returns:
            路由结果
        """
        task_type = self._infer_from_keywords(request)
        
        if task_type == "coding_only":
            console.print("[yellow]→ Simple routing to Coding Agent[/yellow]")
            from src.agents.coder import run_coding_agent
            return await run_coding_agent(request)
        else:
            console.print("[yellow]→ Simple routing to Research Agent[/yellow]")
            result = await self.research_agent.research(request)
            return result.report


if __name__ == "__main__":
    # 测试示例
    print("Coordinator Agent module loaded successfully")
    print("Use this module by importing: from agents.coordinator import CoordinatorAgent")
