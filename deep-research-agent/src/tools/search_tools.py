"""
Search Tools Module - Web search capabilities for AI Agent.

This module provides a wrapper around the MCP handler for web search functionality.
"""

import asyncio
from typing import Optional, List

from rich.console import Console
from rich.panel import Panel

from ..mcp_handler import (
    MCPHandler,
    MockMCPHandler,
    SearchResponse,
    SearchResult,
    create_mcp_handler
)

console = Console()


class SearchTools:
    """
    Web search tools wrapper using MCP protocol.

    Provides a simplified interface for web searches using
    Brave Search or other MCP-compatible search servers.
    """

    def __init__(self, use_mock: bool = False):
        """
        Initialize search tools.

        Args:
            use_mock: Use mock search for testing
        """
        self.use_mock = use_mock
        self.handler = create_mcp_handler(use_mock=use_mock)

    async def search(self, query: str, show_ui: bool = True) -> SearchResponse:
        """
        Perform a web search.

        Args:
            query: Search query string
            show_ui: Whether to display UI feedback

        Returns:
            SearchResponse with results
        """
        if show_ui:
            console.print(Panel(
                f"[yellow]Searching:[/yellow] {query}",
                title="[bold cyan]Web Search[/bold cyan]",
                border_style="yellow"
            ))

        try:
            async with self.handler.connect():
                response = await self.handler.call_search_tool(query)

            if show_ui:
                if response.success:
                    console.print(Panel(
                        f"[green]Found {len(response.results)} results[/green]",
                        title="[bold green]Search Complete[/bold green]",
                        border_style="green"
                    ))
                else:
                    console.print(Panel(
                        f"[red]Search failed:[/red] {response.error}",
                        title="[bold red]Search Error[/bold red]",
                        border_style="red"
                    ))

            return response

        except Exception as e:
            if show_ui:
                console.print(Panel(
                    f"[red]Error:[/red] {str(e)}",
                    title="[bold red]Search Error[/bold red]",
                    border_style="red"
                ))
            return SearchResponse(query=query, error=str(e))

    async def multi_search(
        self,
        queries: List[str],
        show_ui: bool = True
    ) -> List[SearchResponse]:
        """
        Perform multiple searches in parallel.

        Args:
            queries: List of search queries
            show_ui: Whether to display UI feedback

        Returns:
            List of SearchResponse objects
        """
        if show_ui:
            console.print(Panel(
                f"[yellow]Executing {len(queries)} parallel searches...[/yellow]",
                title="[bold cyan]Multi-Search[/bold cyan]",
                border_style="yellow"
            ))

        try:
            async with self.handler.connect():
                tasks = [self.handler.call_search_tool(query) for query in queries]
                results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            responses = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    responses.append(SearchResponse(
                        query=queries[i],
                        error=str(result)
                    ))
                else:
                    responses.append(result)

            if show_ui:
                success_count = sum(1 for r in responses if r.success)
                console.print(Panel(
                    f"[green]Completed {success_count}/{len(queries)} searches successfully[/green]",
                    title="[bold green]Multi-Search Complete[/bold green]",
                    border_style="green"
                ))

            return responses

        except Exception as e:
            if show_ui:
                console.print(Panel(
                    f"[red]Error:[/red] {str(e)}",
                    title="[bold red]Multi-Search Error[/bold red]",
                    border_style="red"
                ))
            return [SearchResponse(query=q, error=str(e)) for q in queries]


# Global instance for convenience
_search_tools: Optional[SearchTools] = None


def get_search_tools(use_mock: bool = False) -> SearchTools:
    """Get or create global SearchTools instance."""
    global _search_tools
    if _search_tools is None or _search_tools.use_mock != use_mock:
        _search_tools = SearchTools(use_mock=use_mock)
    return _search_tools


async def search_web(
    query: str,
    use_mock: bool = False,
    show_ui: bool = True
) -> SearchResponse:
    """
    Convenience function for single web search.

    Args:
        query: Search query
        use_mock: Use mock search for testing
        show_ui: Display UI feedback

    Returns:
        SearchResponse with results
    """
    tools = get_search_tools(use_mock=use_mock)
    return await tools.search(query, show_ui=show_ui)


async def search_web_multi(
    queries: List[str],
    use_mock: bool = False,
    show_ui: bool = True
) -> List[SearchResponse]:
    """
    Convenience function for multiple web searches.

    Args:
        queries: List of search queries
        use_mock: Use mock search for testing
        show_ui: Display UI feedback

    Returns:
        List of SearchResponse objects
    """
    tools = get_search_tools(use_mock=use_mock)
    return await tools.multi_search(queries, show_ui=show_ui)


# Tool definitions for Gemini function calling
SEARCH_TOOLS = [
    {
        "name": "search_web",
        "description": "Search the web for information using Brave Search. Use this to find current information, articles, documentation, etc.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                }
            },
            "required": ["query"]
        }
    }
]


if __name__ == "__main__":
    # Test the search tools
    async def test():
        print("Testing search tools with mock...")
        result = await search_web("Python programming", use_mock=True)
        print(result.to_markdown())

        print("\nTesting multi-search with mock...")
        results = await search_web_multi(
            ["Python basics", "Python advanced"],
            use_mock=True
        )
        for r in results:
            print(r.to_markdown())

    asyncio.run(test())
