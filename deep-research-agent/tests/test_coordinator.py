"""
Test suite for Coordinator Agent
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from agents.coordinator import CoordinatorAgent


class TestCoordinatorAgent:
    """测试Coordinator Agent的路由和编排功能"""
    
    @pytest.fixture
    def mock_research_agent(self):
        """Mock Research Agent"""
        agent = Mock()
        result = Mock()
        result.report = "# Research Report\n\nThis is a research result."
        agent.research = AsyncMock(return_value=result)
        return agent
    
    @pytest.fixture
    def mock_coding_agent(self):
        """Mock Coding Agent"""
        agent = Mock()
        agent.process = AsyncMock(return_value="```python\nprint('Hello')\n```")
        return agent
    
    @pytest.fixture
    def mock_gemini_client(self):
        """Mock Gemini Client"""
        client = Mock()
        return client
    
    @pytest.fixture
    def coordinator(self, mock_research_agent, mock_coding_agent, mock_gemini_client):
        """创建Coordinator实例"""
        return CoordinatorAgent(
            research_agent=mock_research_agent,
            coding_agent=mock_coding_agent,
            gemini_client=mock_gemini_client
        )
    
    def test_infer_from_keywords_coding(self, coordinator):
        """测试编程任务关键词识别"""
        request = "Write a function to calculate fibonacci numbers"
        task_type = coordinator._infer_from_keywords(request)
        assert task_type == "coding_only"
    
    def test_infer_from_keywords_research(self, coordinator):
        """测试研究任务关键词识别"""
        request = "Explain how quantum computing works"
        task_type = coordinator._infer_from_keywords(request)
        assert task_type == "research_only"
    
    def test_infer_from_keywords_combined(self, coordinator):
        """测试复合任务关键词识别"""
        request = "Research fibonacci algorithm and then write code for it"
        task_type = coordinator._infer_from_keywords(request)
        assert task_type == "research_then_code"
    
    @pytest.mark.asyncio
    async def test_process_request_research_only(
        self, coordinator, mock_research_agent, mock_coding_agent, mock_gemini_client
    ):
        """测试纯研究任务路由"""
        # Mock LLM response
        llm_response = Mock()
        llm_response.text = "research_only"
        mock_gemini_client.generate_content_async = AsyncMock(return_value=llm_response)
        
        result = await coordinator.process_request("What is quantum computing?")
        
        # 验证调用了research agent
        mock_research_agent.research.assert_called_once()
        mock_coding_agent.process.assert_not_called()
        assert "Research Report" in result
    
    @pytest.mark.asyncio
    async def test_process_request_coding_only(
        self, coordinator, mock_research_agent, mock_coding_agent, mock_gemini_client
    ):
        """测试纯编程任务路由"""
        # Mock LLM response
        llm_response = Mock()
        llm_response.text = "coding_only"
        mock_gemini_client.generate_content_async = AsyncMock(return_value=llm_response)
        
        result = await coordinator.process_request("Write a fibonacci function")
        
        # 验证调用了coding agent
        mock_coding_agent.process.assert_called_once()
        mock_research_agent.research.assert_not_called()
        assert "python" in result
    
    @pytest.mark.asyncio
    async def test_process_request_research_then_code(
        self, coordinator, mock_research_agent, mock_coding_agent, mock_gemini_client
    ):
        """测试复合任务编排"""
        # Mock LLM response
        llm_response = Mock()
        llm_response.text = "research_then_code"
        mock_gemini_client.generate_content_async = AsyncMock(return_value=llm_response)
        
        result = await coordinator.process_request(
            "Research bubble sort and implement it in Python"
        )
        
        # 验证两个agents都被调用
        mock_research_agent.research.assert_called_once()
        mock_coding_agent.process.assert_called_once()
        
        # 验证结果包含两部分  
        assert "Research Findings" in result
        assert "Code Implementation" in result
    
    @pytest.mark.asyncio
    async def test_fallback_on_llm_failure(
        self, coordinator, mock_research_agent, mock_coding_agent, mock_gemini_client
    ):
        """测试LLM失败时的回退机制"""
        # Mock LLM failure
        mock_gemini_client.generate_content_async = AsyncMock(
            side_effect=Exception("API Error")
        )
        
        result = await coordinator.process_request("Write a hello world program")
        
        # 应该使用关键词fallback，并路由到coding agent
        mock_coding_agent.process.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
