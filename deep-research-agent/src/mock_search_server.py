#!/usr/bin/env python3
"""
Mock MCP Search Server - Simulates Google Search MCP server for testing.

This server implements the MCP protocol over stdio and returns mock search
results for testing the Deep Research Agent workflow.

Usage:
    python src/mock_search_server.py

The server communicates via stdin/stdout using the MCP JSON-RPC protocol.
"""

import asyncio
import json
import sys
import hashlib
from typing import Any
from datetime import datetime

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent


# Mock search data - simulates real search results
MOCK_SEARCH_DATABASE = {
    "quantum computing": [
        {
            "title": "IBM Unveils New 1000-Qubit Quantum Processor",
            "url": "https://newsroom.ibm.com/quantum-computing-2024",
            "snippet": "IBM has announced a major breakthrough with their new 1000-qubit quantum processor, marking a significant milestone in the race toward practical quantum computing. The new chip demonstrates improved error correction and longer coherence times."
        },
        {
            "title": "Google's Quantum Supremacy: What It Means for the Future",
            "url": "https://blog.google/technology/ai/quantum-supremacy-explained",
            "snippet": "Google's achievement of quantum supremacy represents a pivotal moment in computing history. Their Sycamore processor performed calculations in 200 seconds that would take classical supercomputers 10,000 years."
        },
        {
            "title": "Quantum Computing Applications in Drug Discovery",
            "url": "https://nature.com/articles/quantum-drug-discovery",
            "snippet": "Pharmaceutical companies are increasingly turning to quantum computing for drug discovery. The technology's ability to simulate molecular interactions could revolutionize how we develop new medicines."
        },
    ],
    "artificial intelligence": [
        {
            "title": "The Rise of Large Language Models: GPT-5 and Beyond",
            "url": "https://techreview.com/ai/llm-evolution",
            "snippet": "Large language models have evolved rapidly, with GPT-5 demonstrating unprecedented capabilities in reasoning, coding, and creative tasks. Researchers discuss the implications for AI safety and alignment."
        },
        {
            "title": "AI in Healthcare: Transforming Diagnosis and Treatment",
            "url": "https://healthtech.org/ai-medical-applications",
            "snippet": "Artificial intelligence is revolutionizing healthcare, from early disease detection to personalized treatment plans. Machine learning algorithms can now detect certain cancers with higher accuracy than human specialists."
        },
        {
            "title": "The Ethics of AI: Navigating Bias and Fairness",
            "url": "https://ethics.stanford.edu/ai-fairness",
            "snippet": "As AI systems become more prevalent, concerns about bias and fairness have taken center stage. Researchers are developing new frameworks to ensure AI systems treat all users equitably."
        },
    ],
    "climate change": [
        {
            "title": "IPCC Report: Climate Action Needed by 2030",
            "url": "https://ipcc.ch/reports/ar6",
            "snippet": "The latest IPCC report emphasizes the urgent need for climate action. Without significant reductions in greenhouse gas emissions by 2030, the world faces irreversible climate impacts."
        },
        {
            "title": "Renewable Energy Surpasses Fossil Fuels in Europe",
            "url": "https://eurostat.eu/energy-statistics-2024",
            "snippet": "For the first time, renewable energy sources have generated more electricity than fossil fuels in Europe. Wind and solar power led the charge, accounting for 40% of total generation."
        },
        {
            "title": "Carbon Capture Technology: Promise and Challenges",
            "url": "https://sciencemag.org/carbon-capture-review",
            "snippet": "Carbon capture and storage (CCS) technology is gaining momentum as a tool to combat climate change. However, scaling these technologies remains a significant challenge."
        },
    ],
    "default": [
        {
            "title": "Research Overview: {query}",
            "url": "https://scholar.google.com/search?q={query}",
            "snippet": "Comprehensive research on {query} reveals multiple perspectives and ongoing developments in this field. Experts continue to explore new approaches and methodologies."
        },
        {
            "title": "Wikipedia: {query}",
            "url": "https://en.wikipedia.org/wiki/{query}",
            "snippet": "{query} is a topic of significant interest in various academic and professional fields. This encyclopedia entry provides foundational information and historical context."
        },
        {
            "title": "Latest News: {query}",
            "url": "https://news.google.com/search?q={query}",
            "snippet": "Recent developments related to {query} have garnered attention from researchers, policymakers, and the general public. Stay updated with the latest news and analysis."
        },
    ],
}


def get_mock_results(query: str) -> list[dict]:
    """
    Get mock search results for a query.

    Args:
        query: The search query.

    Returns:
        List of mock search result dictionaries.
    """
    query_lower = query.lower()

    # Check for matching keywords in our database
    for keyword, results in MOCK_SEARCH_DATABASE.items():
        if keyword != "default" and keyword in query_lower:
            return results

    # Return default results with query substitution
    default_results = []
    for result in MOCK_SEARCH_DATABASE["default"]:
        default_results.append({
            "title": result["title"].format(query=query),
            "url": result["url"].format(query=query.replace(" ", "+")),
            "snippet": result["snippet"].format(query=query),
        })

    return default_results


def generate_result_id(query: str) -> str:
    """Generate a unique result ID based on query and timestamp."""
    data = f"{query}-{datetime.now().isoformat()}"
    return hashlib.md5(data.encode()).hexdigest()[:8]


# Create the MCP server
server = Server("mock-google-search")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="search",
            description="Search the web for information using Google Search (mock)",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to execute"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return (default: 3, max: 10)",
                        "default": 3,
                        "minimum": 1,
                        "maximum": 10
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="google_search",
            description="Alias for search - Search the web using Google (mock)",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    }
                },
                "required": ["query"]
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""
    if name not in ["search", "google_search"]:
        raise ValueError(f"Unknown tool: {name}")

    query = arguments.get("query", "")
    num_results = arguments.get("num_results", 3)

    if not query:
        return [TextContent(
            type="text",
            text=json.dumps({"error": "Query parameter is required"})
        )]

    # Simulate network delay
    await asyncio.sleep(0.2)

    # Get mock results
    results = get_mock_results(query)[:num_results]

    # Format response
    response = {
        "query": query,
        "result_id": generate_result_id(query),
        "timestamp": datetime.now().isoformat(),
        "source": "mock-google-search",
        "results": results
    }

    return [TextContent(
        type="text",
        text=json.dumps(response, indent=2)
    )]


async def main():
    """Run the mock MCP server."""
    # Print startup message to stderr (stdout is for MCP protocol)
    print("Mock Google Search MCP Server starting...", file=sys.stderr)

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
