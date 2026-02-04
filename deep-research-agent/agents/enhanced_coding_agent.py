"""
Enhanced Coding Agent for Large-Scale ML Projects

Capabilities:
- Generate complete ML project structures
- Implement papers based on PaperAnalysis
- Create training/evaluation scripts
- Generate PyTorch/TensorFlow models
"""

import os
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from src.client import GeminiClient
from agents.enhanced_research_agent import PaperAnalysis, PaperUnderstanding


@dataclass
class ProjectStructure:
    """ML project structure definition"""
    name: str
    description: str
    framework: str  # pytorch, tensorflow, jax
    directories: List[str] = field(default_factory=list)
    files: Dict[str, str] = field(default_factory=dict)  # filepath -> content
    
    def get_tree(self) -> str:
        """Get project tree visualization"""
        tree_lines = [f"{self.name}/"]
        for dir_path in sorted(self.directories):
            tree_lines.append(f"├── {dir_path}/")
        for file_path in sorted(self.files.keys()):
            tree_lines.append(f"├── {file_path}")
        return "\n".join(tree_lines)


@dataclass
class PaperImplementation:
    """Paper implementation result"""
    paper_title: str
    project: ProjectStructure
    model_code: str
    training_code: str
    readme: str
    requirements: List[str] = field(default_factory=list)


class EnhancedCodingAgent:
    """
    Enhanced Coding Agent for ML Projects and Paper Reproduction
    
    Features:
    - Large-scale project generation
    - Paper-based code implementation
    - Multiple framework support (PyTorch, TensorFlow)
    - Complete training pipelines
    """
    
    def __init__(self, gemini_client: GeminiClient):
        self.gemini = gemini_client
    
    async def generate_ml_project(
        self,
        project_name: str,
        description: str,
        framework: str = "pytorch",
        include_training: bool = True,
        include_evaluation: bool = True
    ) -> ProjectStructure:
        """
        Generate a complete ML project structure
        
        Args:
            project_name: Project name
            description: What the project should do
            framework: ML framework (pytorch/tensorflow/jax)
            include_training: Include training scripts
            include_evaluation: Include evaluation scripts
            
        Returns:
            Complete project structure with code
        """
        print(f"🏗️ Generating {framework} project: {project_name}")
        
        # Step 1: Plan project structure
        structure = await self._plan_project_structure(
            project_name,
            description,
            framework,
            include_training,
            include_evaluation
        )
        
        # Step 2: Generate code for each component
        project = ProjectStructure(
            name=project_name,
            description=description,
            framework=framework,
            directories=structure["directories"],
            files={}
        )
        
        # Generate core files
        for file_spec in structure["files"]:
            print(f"  📝 Generating {file_spec['path']}...")
            content = await self._generate_file_content(
                file_spec,
                framework,
                description
            )
            project.files[file_spec["path"]] = content
        
        print(f"✅ Generated {len(project.files)} files")
        return project
    
    async def implement_from_paper(
        self,
        paper_analysis: PaperAnalysis,
        framework: str = "pytorch"
    ) -> PaperImplementation:
        """
        Implement a paper based on PaperAnalysis
        
        Args:
            paper_analysis: Analysis from EnhancedResearchAgent
            framework: Target framework
            
        Returns:
            Complete implementation
        """
        print(f"🎓 Implementing paper: {paper_analysis.content.title}")
        print(f"   Framework: {framework}")
        
        paper_title = paper_analysis.content.title
        understanding = paper_analysis.understanding
        
        # Step 1: Generate model architecture
        print("\n📐 Generating model architecture...")
        model_code = await self._generate_model_from_understanding(
            understanding,
            framework
        )
        
        # Step 2: Generate training script
        print("🏋️ Generating training script...")
        training_code = await self._generate_training_script(
            understanding,
            framework
        )
        
        # Step 3: Create project structure
        print("🏗️ Creating project structure...")
        project = await self._create_paper_project_structure(
            paper_title,
            framework,
            model_code,
            training_code
        )
        
        # Step 4: Generate README
        print("📄 Generating README...")
        readme = await self._generate_readme(
            paper_analysis,
            framework
        )
        
        # Step 5: Extract dependencies
        requirements = self._extract_requirements(framework)
        
        print(f"✅ Implementation complete!")
        
        return PaperImplementation(
            paper_title=paper_title,
            project=project,
            model_code=model_code,
            training_code=training_code,
            readme=readme,
            requirements=requirements
        )
    
    async def _plan_project_structure(
        self,
        name: str,
        description: str,
        framework: str,
        include_training: bool,
        include_evaluation: bool
    ) -> Dict:
        """Plan project directory structure"""
        
        # Standard ML project structure
        directories = [
            "models",
            "data",
            "configs",
            "utils"
        ]
        
        if include_training:
            directories.append("trainers")
        
        if include_evaluation:
            directories.append("evaluation")
        
        # Define files to generate
        files = [
            {"path": "models/model.py", "description": "Main model architecture"},
            {"path": "configs/default.yaml", "description": "Default configuration"},
            {"path": "utils/helpers.py", "description": "Helper functions"},
            {"path": "requirements.txt", "description": "Python dependencies"},
            {"path": "README.md", "description": "Project documentation"}
        ]
        
        if include_training:
            files.append({"path": "train.py", "description": "Training script"})
            files.append({"path": "trainers/trainer.py", "description": "Trainer class"})
        
        if include_evaluation:
            files.append({"path": "evaluate.py", "description": "Evaluation script"})
        
        return {
            "directories": directories,
            "files": files
        }
    
    async def _generate_file_content(
        self,
        file_spec: Dict,
        framework: str,
        project_description: str
    ) -> str:
        """Generate content for a specific file"""
        
        prompt = f"""Generate a complete, production-ready implementation for this file in a {framework} ML project.

Project Description: {project_description}

File: {file_spec['path']}
Purpose: {file_spec['description']}

Requirements:
- Use {framework} best practices
- Include comprehensive docstrings
- Add type hints
- Include error handling
- Make it modular and reusable
- Add inline comments for complex logic

Generate the complete file content:"""
        
        try:
            response = await self.gemini.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"# Error generating {file_spec['path']}: {e}\n# TODO: Implement manually"
    
    async def _generate_model_from_understanding(
        self,
        understanding: PaperUnderstanding,
        framework: str
    ) -> str:
        """Generate model code from paper understanding"""
        
        prompt = f"""Based on this research paper analysis, generate a complete {framework} implementation of the model.

**Paper Methodology:**
{understanding.methodology}

**Key Equations:**
{chr(10).join(understanding.key_equations) if understanding.key_equations else 'See methodology'}

Requirements:
1. Implement the exact architecture described
2. Use {framework} best practices
3. Include detailed docstrings explaining each component
4. Add type hints
5. Make it modular with clear forward() method
6. Handle edge cases

Generate complete {framework} model code:"""
        
        try:
            response = await self.gemini.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"# Error generating model: {e}\n# TODO: Implement manually"
    
    async def _generate_training_script(
        self,
        understanding: PaperUnderstanding,
        framework: str
    ) -> str:
        """Generate training script from paper"""
        
        prompt = f"""Generate a complete training script for the paper based on this analysis.

**Experiments & Setup:**
{understanding.experiments}

**Results to Reproduce:**
{understanding.results}

Generate a {framework} training script that:
1. Loads data as described in the paper
2. Implements the training loop
3. Uses the same hyperparameters mentioned
4. Includes logging and checkpointing
5. Supports both training and validation
6. Saves model weights

Framework: {framework}

Generate complete training code:"""
        
        try:
            response = await self.gemini.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"# Error generating training code: {e}\n# TODO: Implement manually"
    
    async def _create_paper_project_structure(
        self,
        paper_title: str,
        framework: str,
        model_code: str,
        training_code: str
    ) -> ProjectStructure:
        """Create project structure for paper implementation"""
        
        # Sanitize title for project name
        project_name = "".join(c if c.isalnum() or c == ' ' else '' for c in paper_title)
        project_name = "_".join(project_name.lower().split()[:5])  # Max 5 words
        
        project = ProjectStructure(
            name=project_name,
            description=f"Implementation of '{paper_title}'",
            framework=framework,
            directories=[
                "models",
                "data",
                "configs",
                "utils",
                "experiments"
            ],
            files={
                "models/model.py": model_code,
                "train.py": training_code,
                "configs/default.yaml": self._generate_default_config(framework),
                "utils/__init__.py": "# Utility functions",
                "experiments/reproduce.py": "# Experiment reproduction script"
            }
        )
        
        return project
    
    def _generate_default_config(self, framework: str) -> str:
        """Generate default config file"""
        return f"""# Default configuration for {framework} training

model:
  name: model
  hidden_size: 256
  num_layers: 3

training:
  batch_size: 32
  learning_rate: 0.001
  epochs: 100
  device: cuda

data:
  train_path: data/train
  val_path: data/val
  num_workers: 4
"""
    
    async def _generate_readme(
        self,
        paper_analysis: PaperAnalysis,
        framework: str
    ) -> str:
        """Generate README for paper implementation"""
        
        prompt = f"""Generate a comprehensive README.md for this paper implementation.

**Paper Title:** {paper_analysis.content.title}
**Authors:** {', '.join(paper_analysis.content.authors[:5])}
**Framework:** {framework}

**Paper Abstract:**
{paper_analysis.content.abstract}

**Main Contributions:**
{paper_analysis.understanding.contributions}

Include in README:
1. Title and paper citation
2. Brief overview
3. Installation instructions
4. Usage examples
5. Training instructions
6. Results comparison with paper
7. Citation
8. License (MIT)

Generate Markdown README:"""
        
        try:
            response = await self.gemini.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"# {paper_analysis.content.title}\n\nImplementation of the paper.\n\n## TODO\nComplete README"
    
    def _extract_requirements(self, framework: str) -> List[str]:
        """Extract Python requirements based on framework"""
        
        base_reqs = [
            "numpy>=1.21.0",
            "matplotlib>=3.4.0",
            "tqdm>=4.62.0",
            "pyyaml>=6.0"
        ]
        
        if framework == "pytorch":
            return base_reqs + [
                "torch>=2.0.0",
                "torchvision>=0.15.0"
            ]
        elif framework == "tensorflow":
            return base_reqs + [
                "tensorflow>=2.12.0"
            ]
        elif framework == "jax":
            return base_reqs + [
                "jax>=0.4.0",
                "flax>=0.7.0"
            ]
        else:
            return base_reqs
    
    def save_project(self, project: ProjectStructure, output_dir: str):
        """
        Save project structure to disk
        
        Args:
            project: Project to save
            output_dir: Output directory
        """
        import os
        
        project_path = os.path.join(output_dir, project.name)
        
        # Create directories
        for dir_path in project.directories:
            full_path = os.path.join(project_path, dir_path)
            os.makedirs(full_path, exist_ok=True)
            
            # Create __init__.py for Python packages
            init_file = os.path.join(full_path, "__init__.py")
            if not os.path.exists(init_file):
                with open(init_file, 'w') as f:
                    f.write("")
        
        # Write files
        for file_path, content in project.files.items():
            full_path = os.path.join(project_path, file_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            
            with open(full_path, 'w') as f:
                f.write(content)
        
        print(f"✅ Project saved to: {project_path}")
        print(f"\n{project.get_tree()}")


if __name__ == "__main__":
    import asyncio
    from src.client import GeminiClient
    
    async def test():
        client = GeminiClient()
        agent = EnhancedCodingAgent(client)
        
        # Test project generation
        print("Testing ML project generation...")
        project = await agent.generate_ml_project(
            project_name="image_classifier",
            description="CNN image classifier for CIFAR-10",
            framework="pytorch"
        )
        
        print("\n" + "="*50)
        print("PROJECT STRUCTURE:")
        print("="*50)
        print(project.get_tree())
        
        # Save to disk
        agent.save_project(project, "/tmp")
    
    asyncio.run(test())
