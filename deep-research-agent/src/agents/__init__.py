"""
Agents module for AI Agent.

Contains specialized agents for different tasks:
- Researcher: Deep research and information gathering
- Coder: Code generation and debugging (LeetCode solver)
"""

from .researcher import ResearchAgent
from .coder import CodingAgent

__all__ = ["ResearchAgent", "CodingAgent"]
