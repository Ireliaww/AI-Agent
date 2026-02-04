"""
Test Paper Reproduction Workflow

Tests the complete pipeline:
1. EnhancedResearchAgent analyzes paper
2. EnhancedCodingAgent implements it
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.panel import Panel
from agents import EnhancedResearchAgent, EnhancedCodingAgent
from src.client import GeminiClient

console = Console()


async def test_paper_reproduction():
    """Test complete paper reproduction workflow"""
    
    console.print(Panel(
        "[bold cyan]Testing Paper Reproduction Workflow[/bold cyan]\n"
        "Research Agent → Coding Agent → Complete Implementation",
        border_style="cyan"
    ))
    
    # Initialize
    console.print("\n[bold]1. Initializing agents...[/bold]")
    gemini = GeminiClient()
    research_agent = EnhancedResearchAgent(gemini)
    coding_agent = EnhancedCodingAgent(gemini)
    console.print("   [green]✓ Agents initialized[/green]")
    
    # Step 1: Analyze Paper (using a simpler test paper for quick validation)
    console.print("\n[bold]2. Testing Research Agent[/bold]")
    console.print("   [dim]Note: Using search only (no full analysis) to save API calls[/dim]")
    
    try:
        # Quick search test
        papers = await research_agent.academic_search.search_arxiv(
            "ResNet deep residual learning",
            max_results=1
        )
        
        if papers:
            test_paper = papers[0]
            console.print(f"   [green]✓ Found paper: {test_paper.title[:60]}...[/green]")
            
            # Create mock analysis for testing
            from agents.enhanced_research_agent import PaperAnalysis, PaperUnderstanding, PaperContent
            
            mock_understanding = PaperUnderstanding(
                contributions="Introduced residual learning framework with skip connections",
                methodology="Deep residual networks with shortcut connections that skip layers",
                key_equations=["y = F(x, {Wi}) + x"],
                experiments="ImageNet classification, CIFAR-10",
                results="Won ImageNet 2015, reduced error rate significantly",
                limitations="Requires careful initialization"
            )
            
            mock_content = PaperContent(
                title=test_paper.title,
                authors=test_paper.authors,
                abstract=test_paper.abstract,
                full_text="",
                sections={},
                equations=[],
                references=[]
            )
            
            mock_analysis = PaperAnalysis(
                content=mock_content,
                understanding=mock_understanding,
                related_papers=[],
                implementations=[]
            )
            
            console.print("   [green]✓ Created mock analysis for testing[/green]")
        else:
            console.print("   [red]✗ No papers found[/red]")
            return False
            
    except Exception as e:
        console.print(f"   [red]✗ Research agent error: {e}[/red]")
        return False
    
    # Step 2: Implement Paper
    console.print("\n[bold]3. Testing Coding Agent - Paper Implementation[/bold]")
    
    try:
        implementation = await coding_agent.implement_from_paper(
            mock_analysis,
            framework="pytorch"
        )
        
        console.print(f"   [green]✓ Generated implementation[/green]")
        console.print(f"   [green]✓ Project: {implementation.project.name}[/green]")
        console.print(f"   [green]✓ Files: {len(implementation.project.files)}[/green]")
        console.print(f"   [green]✓ Requirements: {len(implementation.requirements)}[/green]")
        
        # Show project structure
        console.print("\n[bold]4. Project Structure:[/bold]")
        console.print(implementation.project.get_tree())
        
        # Show sample: model code preview
        console.print("\n[bold]5. Model Code Preview (first 500 chars):[/bold]")
        model_preview = implementation.model_code[:500]
        console.print(f"[dim]{model_preview}...[/dim]")
        
        console.print("\n[green]✅ Paper reproduction workflow test passed![/green]")
        return True
        
    except Exception as e:
        console.print(f"   [red]✗ Coding agent error: {e}[/red]")
        import traceback
        traceback.print_exc()
        return False


async def test_project_generation():
    """Test standalone ML project generation"""
    
    console.print("\n" + "="*60)
    console.print(Panel(
        "[bold cyan]Testing Standalone Project Generation[/bold cyan]",
        border_style="cyan"
    ))
    
    try:
        gemini = GeminiClient()
        coding_agent = EnhancedCodingAgent(gemini)
        
        console.print("\n[bold]Generating PyTorch CNN project...[/bold]")
        
        project = await coding_agent.generate_ml_project(
            project_name="image_ classifier",
            description="CNN for MNIST digit classification with data augmentation",
            framework="pytorch",
            include_training=True,
            include_evaluation=True
        )
        
        console.print(f"\n[green]✓ Generated {len(project.files)} files[/green]")
        console.print("\n[bold]Project Structure:[/bold]")
        console.print(project.get_tree())
        
        return True
        
    except Exception as e:
        console.print(f"[red]✗ Project generation error: {e}[/red]")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests"""
    
    console.print("[bold cyan]=" * 30 + "[/bold cyan]")
    console.print("[bold cyan]Enhanced Coding Agent Test Suite[/bold cyan]")
    console.print("[bold cyan]=" * 30 + "[/bold cyan]")
    
    # Test 1: Paper reproduction
    result1 = await test_paper_reproduction()
    
    # Test 2: Project generation
    result2 = await test_project_generation()
    
    # Summary
    console.print("\n" + "="*60)
    console.print("[bold]Test Summary:[/bold]")
    console.print(f"  Paper Reproduction: {'✅ PASS' if result1 else '❌ FAIL'}")
    console.print(f"  Project Generation: {'✅ PASS' if result2 else '❌ FAIL'}")
    
    if result1 and result2:
        console.print("\n[bold green]🎉 All tests passed![/bold green]")
        return 0
    else:
        console.print("\n[bold red]Some tests failed[/bold red]")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
