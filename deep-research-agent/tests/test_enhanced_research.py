"""
Test script for Enhanced Research Agent with RAG
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from agents.enhanced_research_agent import EnhancedResearchAgent, PaperAnalysis
from src.client import GeminiClient

console = Console()


async def test_enhanced_research_agent():
    """Test Enhanced Research Agent with a small arXiv paper"""
    
    console.print("[bold cyan]Testing Enhanced Research Agent with RAG[/bold cyan]\n")
    
    # Test 1: Import test
    console.print("1. Testing imports...")
    try:
        from rag import ChromaVectorStore, TextChunker
        from tools.pdf_parser import PDFParser
        from tools.academic_search import AcademicSearchTools
        console.print("   [green]✓ All imports successful[/green]")
    except ImportError as e:
        console.print(f"   [red]✗ Import failed: {e}[/red]")
        return False
    
    # Test 2: Agent initialization
    console.print("\n2. Testing agent initialization...")
    try:
        gemini = GeminiClient()
        agent = EnhancedResearchAgent(gemini)
        console.print("   [green]✓ Enhanced Research Agent initialized[/green]")
    except Exception as e:
        console.print(f"   [red]✗ Initialization failed: {e}[/red]")
        return False
    
    # Test 3: Paper search (without full analysis to save API calls)
    console.print("\n3. Testing paper search...")
    try:
        papers = await agent.academic_search.search_arxiv("attention mechanism", max_results=2)
        console.print(f"   [green]✓ Found {len(papers)} papers[/green]")
        for i, paper in enumerate(papers, 1):
            console.print(f"   {i}. {paper.title[:60]}...")
    except Exception as e:
        console.print(f"   [red]✗ Search failed: {e}[/red]")
        return False
    
    # Test 4: Test RAG components (without full paper analysis)
    console.print("\n4. Testing RAG components...")
    try:
        # Test text chunker
        chunker = TextChunker(min_tokens=100, max_tokens=200)
        test_pages = [{
            "page_number": 1,
            "text": "This is a test paper about machine learning. " * 50
        }]
        chunks = chunker.chunk_text(test_pages)
        console.print(f"   [green]✓ Text chunker created {len(chunks)} chunks[/green]")
        
        # Note: Full ChromaDB test would require API key and create persistent storage
        # Skipping to avoid side effects
        console.print("   [yellow]⊙ ChromaDB test skipped (would create persistent storage)[/yellow]")
        
    except Exception as e:
        console.print(f"   [red]✗ RAG component test failed: {e}[/red]")
        return False
    
    console.print("\n[bold green]✓ All basic tests passed![/bold green]")
    console.print("\n[bold cyan]Enhanced Research Agent is ready to use![/bold cyan]")
    console.print("\n[dim]To fully test with paper analysis, run:[/dim]")
    console.print("[dim]  python agents/enhanced_research_agent.py[/dim]")
    console.print("[dim]  (This will download and analyze 'Attention Is All You Need')[/dim]")
    
    return True


if __name__ == "__main__":
    result = asyncio.run(test_enhanced_research_agent())
    sys.exit(0 if result else 1)
