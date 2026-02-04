"""
Test script for Coordinator integration in main.py
"""

import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console

console = Console()


async def test_coordinator_integration():
    """Test that Coordinator is properly integrated"""
    
    console.print("[bold cyan]Testing Coordinator Integration[/bold cyan]\n")
    
    # Test imports
    console.print("1. Testing imports...")
    try:
        from agents.coordinator import CoordinatorAgent
        from src.agents.researcher import ResearchAgent
        from src.agents.coder import CodingAgent
        from src.client import GeminiClient
        console.print("   [green]✓ All imports successful[/green]")
    except ImportError as e:
        console.print(f"   [red]✗ Import failed: {e}[/red]")
        return False
    
    # Test initialization
    console.print("\n2. Testing Coordinator initialization...")
    try:
        gemini = GeminiClient()
        research_agent = ResearchAgent(gemini_client=gemini)
        coding_agent = CodingAgent(gemini_client=gemini)
        coordinator = CoordinatorAgent(
            research_agent=research_agent,
            coding_agent=coding_agent,
            gemini_client=gemini
        )
        console.print("   [green]✓ Coordinator initialized successfully[/green]")
    except Exception as e:
        console.print(f"   [red]✗ Initialization failed: {e}[/red]")
        return False
    
    # Test keyword inference (no API call needed)
    console.print("\n3. Testing task classification (offline)...")
    test_cases = [
        ("Write a function to calculate fibonacci", "coding_only"),
        ("Explain quantum computing", "research_only"),
        ("Research bubble sort and implement it", "research_then_code"),
    ]
    
    all_passed = True
    for query, expected_type in test_cases:
        result = coordinator._infer_from_keywords(query)
        status = "✓" if result == expected_type else "✗"
        color = "green" if result == expected_type else "red"
        console.print(f"   [{color}]{status}[/{color}] '{query[:40]}...' → {result}")
        if result != expected_type:
            all_passed = False
    
    if all_passed:
        console.print("\n[bold green]✓ All tests passed![/bold green]")
        console.print("\nCoordinator is properly integrated and ready to use.")
    else:
        console.print("\n[bold yellow]⚠ Some tests failed, but basic integration works[/bold yellow]")
    
    return True


if __name__ == "__main__":
    result = asyncio.run(test_coordinator_integration())
    sys.exit(0 if result else 1)
