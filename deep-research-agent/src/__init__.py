"""
Multi-Mode AI Assistant - Deep Research & Auto Coding

This package provides a Gemini-powered AI assistant with:
- Deep Research Agent: Information gathering and comprehensive reports
- Coding Agent: LeetCode solver with self-healing debug loop
- Smart Router: Automatic intent classification and task routing
"""

__version__ = "0.2.0"

from .client import GeminiClient
from .router import Router, Intent, classify_intent

__all__ = [
    "GeminiClient",
    "Router",
    "Intent",
    "classify_intent",
]
