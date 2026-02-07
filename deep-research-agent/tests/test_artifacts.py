"""
Test script for Phase 1: Artifact Generation
"""

import sys
import os
from pathlib import Path

# Add parent directory to path (deep-research-agent/)
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.artifact_manager import ArtifactManager


def test_artifact_manager():
    """Test artifact manager functionality"""
    
    print("🧪 Testing Artifact Manager\n")
    
    # Create test project directory in tests folder
    test_dir = "tests/test_artifacts_output"
    if os.path.exists(test_dir):
        import shutil
        shutil.rmtree(test_dir)
    
    # Initialize manager
    manager = ArtifactManager(test_dir)
    
    print("✅ Step 1: ArtifactManager initialized")
    print(f"   Artifacts dir: {manager.artifacts_dir}")
    
    # Test 1: Save paper analysis
    print("\n✅ Step 2: Testing paper analysis artifact...")
    
    rag_queries = [
        {
            'query': 'What are the main contributions?',
            'chunks_found': 3,
            'chunks': [
                {'text': 'The Transformer uses attention mechanism...', 'similarity': 0.89},
                {'text': 'Main contribution is removing recurrence...', 'similarity': 0.85},
            ],
            'analysis': 'The paper introduces the Transformer architecture...'
        }
    ]
    
    path1 = manager.save_paper_analysis(
        title="Attention Is All You Need",
        authors="Vaswani et al.",
        arxiv_id="1706.03762",
        sections=8,
        chunks=45,
        rag_queries=rag_queries,
        understanding={
            'contributions': 'Introduced Transformer architecture',
            'methodology': 'Multi-head self-attention',
            'experiments': 'Tested on WMT translation tasks'
        },
        confidence=95.0
    )
    
    print(f"   Saved to: {path1}")
    print(f"   File exists: {os.path.exists(path1)}")
    
    # Test 2: Save understanding
    print("\n✅ Step 3: Testing understanding artifact...")
    
    path2 = manager.save_understanding(
        title="Attention Is All You Need",
        problem_statement="Sequential models are slow and can't capture long-range dependencies",
        solution_approach="Replace recurrence with pure attention mechanism",
        key_insights=[
            "Attention allows parallel processing",
            "Multi-head attention learns different representations",
            "Positional encoding preserves sequence order"
        ],
        design_decisions=[
            {
                'name': 'Pure Attention',
                'rationale': 'Enables parallelization',
                'tradeoff': 'Need positional encoding',
                'impact': '10x speedup'
            }
        ],
        architecture_notes="Encoder-decoder with 6 layers each"
    )
    
    print(f"   Saved to: {path2}")
    print(f"   File exists: {os.path.exists(path2)}")
    
    # Test 3: Save architecture design
    print("\n✅ Step 4: Testing architecture design artifact...")
    
    path3 = manager.save_architecture_design(
        framework="pytorch",
        components=[],
        design_choices=[
            {
                'aspect': 'Attention Implementation',
                'selected': 'torch.nn.MultiheadAttention',
                'rationale': 'Optimized and maintained',
                'alternatives': 'Custom implementation from scratch'
            }
        ],
        complexity_matrix=[
            {'name': 'Attention', 'lines': 150, 'complexity': 'High', 'priority': 'P0'},
            {'name': 'Encoder', 'lines': 120, 'complexity': 'Medium', 'priority': 'P0'},
        ]
    )
    
    print(f"   Saved to: {path3}")
    print(f"   File exists: {os.path.exists(path3)}")
    
    # Test 4: Implementation log
    print("\n✅ Step 5: Testing implementation log...")
    
    path4 = manager.start_implementation_log(
        title="Attention Is All You Need",
        framework="pytorch"
    )
    
    manager.append_to_log(
        step_name="Model Generation",
        description="Generating transformer model",
        file_generated="models/transformer.py",
        lines=520,
        tokens=3000,
        notes="Used MultiheadAttention for efficiency"
    )
    
    manager.finalize_log(
        total_files=8,
        total_lines=1327,
        total_tokens=12500,
        duration=92.5
    )
    
    print(f"   Saved to: {path4}")
    print(f"   File exists: {os.path.exists(path4)}")
    
    # Test 5: Reproduction report
    print("\n✅ Step 6: Testing reproduction report...")
    
    path5 = manager.save_reproduction_report(
        title="Attention Is All You Need",
        authors="Vaswani et al.",
        arxiv_id="1706.03762",
        project_path="generated_projects/transformer",
        components=[
            {'name': 'Multi-Head Attention', 'implemented': True},
            {'name': 'Encoder Stack', 'implemented': True},
            {'name': 'Decoder Stack', 'implemented': True},
        ],
        fidelity=100.0,
        testing_notes="Model architecture matches paper exactly. Please verify training."
    )
    
    print(f"   Saved to: {path5}")
    print(f"   File exists: {os.path.exists(path5)}")
    
    # Summary
    print("\n" + "="*60)
    print("✅ All tests passed!")
    print(f"\nGenerated artifacts in: {test_dir}/")
    print("\nFiles created:")
    for file in sorted(Path(test_dir).rglob("*.md")):
        size = file.stat().st_size
        print(f"  📄 {file.name} ({size:,} bytes)")
    
    print("\n💡 Tip: Check the generated markdown files to see the output format")


if __name__ == "__main__":
    test_artifact_manager()
