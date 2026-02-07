"""
Demo script for Phase 2: Progress Display

Shows all the visual components that will appear during paper reproduction
"""

import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.progress_display import progress_display


def demo_progress_display():
    """Demonstrate all progress display features"""
    
    progress_display.console.print("\n[bold cyan]🎯 Progress Display Demo[/bold cyan]\n")
    
    # 1. Step headers
    progress_display.console.print("[bold]1. Step Headers[/bold]")
    progress_display.show_step_header(1, "Paper Analysis", 5)
    time.sleep(1)
    progress_display.show_step_header(2, "Model Architecture Generation", 5)
    time.sleep(1)
    
    # 2. AI Thinking panel
    progress_display.console.print("\n[bold]2. AI Thinking Display[/bold]")
    progress_display.show_ai_thinking(
        "Analyzing paper methodology to extract key architectural components...\n"
        "The paper proposes a novel attention mechanism that...",
        "Model Design"
    )
    time.sleep(1)
    
    # 3. File tree
    progress_display.console.print("\n[bold]3. Project Structure Tree[/bold]")
    progress_display.show_file_tree(
        "attention_is_all_you_need",
        files=[
            "models/__init__.py",
            "models/transformer.py",
            "models/attention.py",
            "data/dataset.py",
            "train.py",
            "config.yaml",
            "README.md",
            "requirements.txt"
        ],
        directories=["models", "data", "utils"]
    )
    time.sleep(1)
    
    # 4. Code preview
    progress_display.console.print("\n[bold]4. Code Syntax Highlighting[/bold]")
    sample_code = """class TransformerModel(nn.Module):
    def __init__(self, d_model=512, nhead=8):
        super().__init__()
        self.encoder = TransformerEncoder(d_model, nhead)
        self.decoder = TransformerDecoder(d_model, nhead)
    
    def forward(self, src, tgt):
        enc_output = self.encoder(src)
        return self.decoder(tgt, enc_output)
"""
    progress_display.show_code_preview(
        sample_code,
        language="python",
        title="📄 models/transformer.py (preview)"
    )
    time.sleep(1)
    
    # 5. Statistics table
    progress_display.console.print("\n[bold]5. Statistics Table[/bold]")
    progress_display.show_statistics_table({
        "Total Files": 8,
        "Total Lines": "1,327",
        "Token Usage": "12,500",
        "Duration": "92.5s",
        "Framework": "PyTorch"
    })
    time.sleep(1)
    
    # 6. Completion summary
    progress_display.console.print("\n[bold]6. Completion Summary[/bold]")
    progress_display.show_completion_summary(
        "Code Generation Complete",
        [
            {'icon': '📐', 'text': 'Model architecture (520 lines)'},
            {'icon': '🏋️', 'text': 'Training script (350 lines)'},
            {'icon': '🏗️', 'text': 'Project structure (8 files)'},
            {'icon': '📄', 'text': 'README documentation'},
            {'icon': '📦', 'text': 'Dependencies (12 packages)'},
        ],
        duration=92.5,
        success=True
    )
    
    progress_display.console.print("\n[green]✅ Demo complete![/green]\n")
    progress_display.console.print("[dim]These visual components will appear during actual paper reproduction[/dim]\n")


if __name__ == "__main__":
    demo_progress_display()
