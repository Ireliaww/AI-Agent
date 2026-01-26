"""
Router Module - Intent classification for Multi-Mode AI Assistant.

This module determines which agent should handle a user's query:
- CODING: For code writing, LeetCode, debugging tasks
- RESEARCH: For information queries, news, reports
"""

import asyncio
import json
from typing import Optional
from enum import Enum

from rich.console import Console
from rich.panel import Panel

from .client import GeminiClient

console = Console()


class Intent(Enum):
    """User intent classification."""
    CODING = "coding"
    RESEARCH = "research"
    UNKNOWN = "unknown"


# System instruction for the router
ROUTER_SYSTEM = """You are an intent classifier for a Multi-Mode AI Assistant.

Your job is to analyze user queries and determine which agent should handle them:

1. CODING - For queries involving:
   - Writing code or programs
   - LeetCode, algorithm problems, data structures
   - Debugging or fixing code
   - Code optimization or refactoring
   - Programming exercises or challenges
   - Questions about implementing specific features
   - Keywords: "code", "implement", "write", "leetcode", "debug", "function", "algorithm", "solve", "program"

2. RESEARCH - For queries involving:
   - Information gathering or fact-finding
   - News, current events, trends
   - Explanations of concepts (without code)
   - Reports, summaries, analysis
   - "What is", "How does", "Why", "Explain"
   - Product comparisons, reviews
   - Historical information
   - Keywords: "what", "why", "how", "explain", "research", "find", "news", "compare", "analyze"

Guidelines:
- If the query explicitly mentions code, programming, or LeetCode, classify as CODING
- If the query asks to explain something conceptually without code, classify as RESEARCH
- When in doubt, consider if the expected output is code (CODING) or text/report (RESEARCH)
- Queries like "write a function to..." or "solve this problem..." are CODING
- Queries like "what is the difference between..." or "explain..." are RESEARCH
"""

CLASSIFY_PROMPT = """Classify the following user query into either CODING or RESEARCH.

User Query: {query}

Respond with ONLY a JSON object in this exact format:
{{
    "intent": "CODING" or "RESEARCH",
    "confidence": 0.0-1.0,
    "reason": "Brief explanation of why this classification was chosen"
}}
"""


class Router:
    """
    Intent router for the Multi-Mode AI Assistant.

    Uses Gemini to classify user queries and route them
    to the appropriate agent.
    """

    def __init__(self, gemini_client: Optional[GeminiClient] = None):
        """
        Initialize the router.

        Args:
            gemini_client: Optional pre-configured Gemini client
        """
        self.gemini = gemini_client or GeminiClient()

    async def classify_intent(
        self,
        query: str,
        show_ui: bool = True
    ) -> tuple[Intent, float, str]:
        """
        Classify a user query to determine which agent should handle it.

        Args:
            query: The user's query
            show_ui: Whether to display UI feedback

        Returns:
            Tuple of (Intent, confidence, reason)
        """
        # Quick keyword-based pre-classification for obvious cases
        query_lower = query.lower()

        # Strong CODING indicators
        coding_keywords = [
            "leetcode", "write a function", "implement", "write code",
            "solve this problem", "algorithm", "data structure",
            "debug", "fix this code", "optimize this", "refactor",
            "def ", "class ", "python code", "javascript code",
            "write a program", "coding problem", "programming"
        ]

        for keyword in coding_keywords:
            if keyword in query_lower:
                if show_ui:
                    console.print(Panel(
                        f"[green]Detected intent:[/green] CODING\n[dim](keyword match: '{keyword}')[/dim]",
                        title="[bold cyan]Router[/bold cyan]",
                        border_style="cyan"
                    ))
                return Intent.CODING, 0.95, f"Keyword match: '{keyword}'"

        # Strong RESEARCH indicators
        research_keywords = [
            "what is", "explain", "how does", "why is",
            "research", "find information", "news about",
            "compare", "analyze", "summarize", "report on",
            "latest developments", "current state of"
        ]

        for keyword in research_keywords:
            if keyword in query_lower:
                if show_ui:
                    console.print(Panel(
                        f"[green]Detected intent:[/green] RESEARCH\n[dim](keyword match: '{keyword}')[/dim]",
                        title="[bold cyan]Router[/bold cyan]",
                        border_style="cyan"
                    ))
                return Intent.RESEARCH, 0.95, f"Keyword match: '{keyword}'"

        # Use Gemini for ambiguous cases
        if show_ui:
            console.print("[cyan]Analyzing query intent...[/cyan]")

        try:
            prompt = CLASSIFY_PROMPT.format(query=query)

            response = await self.gemini.generate_content(
                contents=prompt,
                system_instruction=ROUTER_SYSTEM,
                temperature=0.1,  # Low temperature for consistent classification
            )

            # Parse JSON response
            text = response.text.strip()

            # Handle markdown code blocks
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()

            data = json.loads(text)

            intent_str = data.get("intent", "RESEARCH").upper()
            confidence = float(data.get("confidence", 0.5))
            reason = data.get("reason", "AI classification")

            intent = Intent.CODING if intent_str == "CODING" else Intent.RESEARCH

            if show_ui:
                console.print(Panel(
                    f"[green]Detected intent:[/green] {intent.value.upper()}\n"
                    f"[dim]Confidence: {confidence:.2f}[/dim]\n"
                    f"[dim]Reason: {reason}[/dim]",
                    title="[bold cyan]Router[/bold cyan]",
                    border_style="cyan"
                ))

            return intent, confidence, reason

        except json.JSONDecodeError:
            # Fallback: try to extract intent from text
            text_lower = response.text.lower()
            if "coding" in text_lower:
                return Intent.CODING, 0.6, "Extracted from response"
            return Intent.RESEARCH, 0.6, "Default fallback"

        except Exception as e:
            if show_ui:
                console.print(f"[yellow]Classification error: {e}[/yellow]")
            # Default to RESEARCH on error
            return Intent.RESEARCH, 0.5, f"Error fallback: {e}"


# Global router instance
_router: Optional[Router] = None


def get_router() -> Router:
    """Get or create global Router instance."""
    global _router
    if _router is None:
        _router = Router()
    return _router


async def classify_intent(
    query: str,
    show_ui: bool = True
) -> tuple[Intent, float, str]:
    """
    Convenience function to classify a query's intent.

    Args:
        query: The user's query
        show_ui: Display UI feedback

    Returns:
        Tuple of (Intent, confidence, reason)
    """
    router = get_router()
    return await router.classify_intent(query, show_ui=show_ui)


if __name__ == "__main__":
    # Test the router
    async def test():
        test_queries = [
            "Write a function to find the two numbers that add up to a target",
            "What are the latest developments in AI?",
            "Implement a binary search algorithm in Python",
            "Explain how quantum computing works",
            "Debug this code: def foo(): return x + y",
            "Research the history of machine learning",
            "LeetCode problem: Two Sum",
            "Compare React vs Vue for web development",
        ]

        router = Router()

        for query in test_queries:
            print(f"\nQuery: {query}")
            intent, confidence, reason = await router.classify_intent(query, show_ui=False)
            print(f"Intent: {intent.value}, Confidence: {confidence:.2f}")
            print(f"Reason: {reason}")
            print("-" * 50)

    asyncio.run(test())
