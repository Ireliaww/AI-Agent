"""
MCP Handler Module - Interface for Model Context Protocol servers.

This module manages communication with MCP servers, particularly the
Google Search MCP server for web search capabilities.
"""

import os
import asyncio
import json
from typing import Optional, Any
from dataclasses import dataclass, field
from contextlib import asynccontextmanager

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


@dataclass
class SearchResult:
    """Represents a single search result."""

    title: str
    url: str
    snippet: str
    source: str = "google"

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "source": self.source,
        }

    def __str__(self) -> str:
        return f"**{self.title}**\n{self.url}\n{self.snippet}"


@dataclass
class SearchResponse:
    """Represents a complete search response."""

    query: str
    results: list[SearchResult] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        """Check if search was successful."""
        return self.error is None and len(self.results) > 0

    def to_markdown(self) -> str:
        """Convert search results to markdown format."""
        if self.error:
            return f"**Search Error for '{self.query}'**: {self.error}"

        if not self.results:
            return f"**No results found for '{self.query}'**"

        lines = [f"### Search Results for: {self.query}\n"]
        for i, result in enumerate(self.results, 1):
            lines.append(f"**{i}. {result.title}**")
            lines.append(f"   URL: {result.url}")
            lines.append(f"   {result.snippet}\n")

        return "\n".join(lines)


class MCPHandler:
    """
    Handler for MCP (Model Context Protocol) server communication.

    This class manages the lifecycle of MCP server connections and
    provides methods to call tools exposed by the server.
    """

    def __init__(
        self,
        server_command: Optional[str] = None,
        server_args: Optional[list[str]] = None,
        env_vars: Optional[dict[str, str]] = None,
    ):
        """
        Initialize the MCP handler.

        Args:
            server_command: Command to start the MCP server.
                           Defaults to MCP_GOOGLE_SEARCH_CMD env var.
            server_args: Additional arguments for the server command.
            env_vars: Additional environment variables for the server process.
        """
        self.server_command = server_command or os.getenv(
            "MCP_GOOGLE_SEARCH_CMD",
            "python src/mock_search_server.py"
        )
        self.server_args = server_args or []
        self.env_vars = env_vars or {}

        # Auto-detect and add common API keys from environment
        if "BRAVE_API_KEY" not in self.env_vars:
            brave_key = os.getenv("BRAVE_API_KEY")
            if brave_key and brave_key != "your_brave_api_key_here":
                self.env_vars["BRAVE_API_KEY"] = brave_key

        self._session: Optional[ClientSession] = None
        self._available_tools: list[dict] = []

    def _parse_server_command(self) -> tuple[str, list[str]]:
        """Parse the server command into command and arguments."""
        parts = self.server_command.split()
        if not parts:
            raise ValueError("Server command is empty")

        command = parts[0]
        args = parts[1:] + self.server_args
        return command, args

    @asynccontextmanager
    async def connect(self):
        """
        Async context manager for MCP server connection.

        Usage:
            async with handler.connect():
                results = await handler.call_search_tool("query")
        """
        command, args = self._parse_server_command()

        # Merge current environment with custom env vars
        env = {**os.environ, **self.env_vars} if self.env_vars else None

        server_params = StdioServerParameters(
            command=command,
            args=args,
            env=env,
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize the session
                await session.initialize()

                # Store session and discover tools
                self._session = session
                await self._discover_tools()

                try:
                    yield self
                finally:
                    self._session = None
                    self._available_tools = []

    async def _discover_tools(self) -> None:
        """Discover available tools from the MCP server."""
        if not self._session:
            return

        try:
            tools_response = await self._session.list_tools()
            self._available_tools = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema,
                }
                for tool in tools_response.tools
            ]
        except Exception as e:
            print(f"Warning: Could not discover tools: {e}")
            self._available_tools = []

    @property
    def available_tools(self) -> list[dict]:
        """Get list of available tools from the MCP server."""
        return self._available_tools

    def get_tools_for_gemini(self) -> list[dict]:
        """
        Convert MCP tools to Gemini function declarations.

        Returns:
            List of tool definitions compatible with Gemini's tools parameter.
        """
        gemini_tools = []
        for tool in self._available_tools:
            # Convert MCP tool schema to Gemini function declaration
            gemini_tool = {
                "function_declarations": [{
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
                }]
            }
            gemini_tools.append(gemini_tool)
        return gemini_tools

    async def call_tool(self, tool_name: str, arguments: dict) -> Any:
        """
        Call a tool on the MCP server.

        Args:
            tool_name: Name of the tool to call.
            arguments: Arguments to pass to the tool.

        Returns:
            The tool's response.

        Raises:
            RuntimeError: If not connected to server.
        """
        if not self._session:
            raise RuntimeError("Not connected to MCP server. Use 'async with handler.connect():'")

        result = await self._session.call_tool(tool_name, arguments)
        return result

    async def call_search_tool(self, query: str) -> SearchResponse:
        """
        Convenience method to call the search tool.

        Args:
            query: The search query.

        Returns:
            SearchResponse object with results.
        """
        try:
            # Try common search tool names (in order of preference)
            search_tool_names = [
                "brave_web_search",    # Brave Search MCP
                "brave_search",        # Brave Search alternate
                "search",              # Generic
                "google_search",       # Google Search
                "web_search",          # Generic web search
            ]

            tool_name = None
            for name in search_tool_names:
                if any(t["name"] == name for t in self._available_tools):
                    tool_name = name
                    break

            if not tool_name and self._available_tools:
                # Use the first available tool that looks like a search tool
                for tool in self._available_tools:
                    if "search" in tool["name"].lower():
                        tool_name = tool["name"]
                        break
                # Fallback to first tool
                if not tool_name:
                    tool_name = self._available_tools[0]["name"]

            if not tool_name:
                return SearchResponse(
                    query=query,
                    error="No search tool available"
                )

            result = await self.call_tool(tool_name, {"query": query})

            # Parse the result
            return self._parse_search_result(query, result)

        except Exception as e:
            return SearchResponse(query=query, error=str(e))

    def _parse_search_result(self, query: str, result: Any) -> SearchResponse:
        """Parse MCP tool result into SearchResponse."""
        results = []

        try:
            # Handle different result formats
            if hasattr(result, "content"):
                content = result.content
                if isinstance(content, list):
                    for item in content:
                        if hasattr(item, "text"):
                            # Try to parse as JSON
                            try:
                                data = json.loads(item.text)
                                results.extend(self._extract_results_from_data(data, item.text))
                            except json.JSONDecodeError:
                                # Plain text result - might be formatted text from Brave
                                results.extend(self._parse_text_results(item.text))
        except Exception as e:
            return SearchResponse(query=query, error=f"Failed to parse result: {e}")

        return SearchResponse(query=query, results=results)

    def _extract_results_from_data(self, data: Any, raw_text: str = "") -> list[SearchResult]:
        """Extract search results from parsed JSON data."""
        results = []

        if isinstance(data, list):
            for entry in data:
                if isinstance(entry, dict):
                    results.append(SearchResult(
                        title=entry.get("title", ""),
                        url=entry.get("url", entry.get("link", "")),
                        snippet=entry.get("snippet", entry.get("description", "")),
                    ))
        elif isinstance(data, dict):
            # Handle Brave Search format (has 'web' key with results)
            if "web" in data and "results" in data["web"]:
                for entry in data["web"]["results"]:
                    results.append(SearchResult(
                        title=entry.get("title", ""),
                        url=entry.get("url", ""),
                        snippet=entry.get("description", entry.get("snippet", "")),
                        source="brave"
                    ))
            # Handle generic 'results' key
            elif "results" in data:
                for entry in data["results"]:
                    results.append(SearchResult(
                        title=entry.get("title", ""),
                        url=entry.get("url", entry.get("link", "")),
                        snippet=entry.get("snippet", entry.get("description", "")),
                    ))
            # Single result object
            elif "title" in data or "url" in data:
                results.append(SearchResult(
                    title=data.get("title", "Search Result"),
                    url=data.get("url", ""),
                    snippet=data.get("snippet", data.get("description", raw_text[:500])),
                ))

        return results

    def _parse_text_results(self, text: str) -> list[SearchResult]:
        """Parse plain text search results (fallback)."""
        results = []

        # If it's a simple text response, wrap it as a single result
        if text.strip():
            # Try to extract multiple results if formatted with common patterns
            lines = text.strip().split('\n')
            current_title = ""
            current_url = ""
            current_snippet = ""

            for line in lines:
                line = line.strip()
                if line.startswith("Title:") or line.startswith("**"):
                    if current_title and (current_url or current_snippet):
                        results.append(SearchResult(
                            title=current_title,
                            url=current_url,
                            snippet=current_snippet,
                        ))
                    current_title = line.replace("Title:", "").replace("**", "").strip()
                    current_url = ""
                    current_snippet = ""
                elif line.startswith("URL:") or line.startswith("http"):
                    current_url = line.replace("URL:", "").strip()
                elif line and not line.startswith("-"):
                    current_snippet += " " + line

            # Add last result
            if current_title:
                results.append(SearchResult(
                    title=current_title,
                    url=current_url,
                    snippet=current_snippet.strip(),
                ))

            # Fallback: if no structured results found, use whole text
            if not results:
                results.append(SearchResult(
                    title="Search Result",
                    url="",
                    snippet=text[:500],
                ))

        return results


class MockMCPHandler(MCPHandler):
    """
    Mock MCP handler for testing without a real MCP server.

    This handler returns simulated search results for testing purposes.
    """

    def __init__(self):
        """Initialize mock handler."""
        super().__init__()
        self._mock_tools = [
            {
                "name": "search",
                "description": "Search the web for information",
                "input_schema": {
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

    @asynccontextmanager
    async def connect(self):
        """Mock connection - no actual server needed."""
        self._available_tools = self._mock_tools
        try:
            yield self
        finally:
            self._available_tools = []

    async def call_search_tool(self, query: str) -> SearchResponse:
        """Return mock search results."""
        # Simulate some delay
        await asyncio.sleep(0.5)

        mock_results = [
            SearchResult(
                title=f"Mock Result 1 for: {query}",
                url=f"https://example.com/result1?q={query.replace(' ', '+')}",
                snippet=f"This is a mock search result about {query}. "
                        "It contains relevant information that would typically "
                        "be found in a real search result.",
                source="mock"
            ),
            SearchResult(
                title=f"Mock Result 2 for: {query}",
                url=f"https://example.com/result2?q={query.replace(' ', '+')}",
                snippet=f"Another mock result discussing {query}. "
                        "This provides additional context and information "
                        "about the search topic.",
                source="mock"
            ),
            SearchResult(
                title=f"Wikipedia: {query}",
                url=f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}",
                snippet=f"From Wikipedia: {query} is a topic of interest. "
                        "This encyclopedia entry provides comprehensive information.",
                source="mock"
            ),
        ]

        return SearchResponse(query=query, results=mock_results)


# Utility function to create appropriate handler
def create_mcp_handler(use_mock: bool = False) -> MCPHandler:
    """
    Factory function to create an MCP handler.

    Args:
        use_mock: If True, returns a mock handler for testing.

    Returns:
        MCPHandler or MockMCPHandler instance.
    """
    if use_mock:
        return MockMCPHandler()
    return MCPHandler()


if __name__ == "__main__":
    # Test the mock handler
    async def test_mock():
        handler = MockMCPHandler()
        async with handler.connect():
            print("Available tools:", handler.available_tools)
            result = await handler.call_search_tool("Python programming")
            print(result.to_markdown())

    asyncio.run(test_mock())
