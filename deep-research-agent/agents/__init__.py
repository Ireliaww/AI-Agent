"""
Agents package

Provides AI agent implementations:
- CoordinatorAgent: Intelligent multi-agent orchestration
- EnhancedResearchAgent: RAG-powered paper analysis
- EnhancedCodingAgent: Large-scale ML project generation
"""

from .coordinator import CoordinatorAgent
from .enhanced_research_agent import EnhancedResearchAgent, PaperAnalysis, PaperUnderstanding
from .enhanced_coding_agent import EnhancedCodingAgent, PaperImplementation, ProjectStructure

__all__ = [
    'CoordinatorAgent',
    'EnhancedResearchAgent', 'PaperAnalysis', 'PaperUnderstanding',
    'EnhancedCodingAgent', 'PaperImplementation', 'ProjectStructure'
]
