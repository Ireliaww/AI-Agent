# Changelog

All notable changes to the AI-Agent Paper Reproduction System will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-02-04

### Added
- **Dual-Mode Research Agent**: EnhancedResearchAgent now supports both paper analysis and general research
- **Smart Paper Title Extraction**: Intelligent extraction from natural language using regex patterns
  - Supports quoted text extraction
  - Handles "reproduce X paper" patterns
  - Fallback to full query search
- **General Research Method**: Added `research()` method to EnhancedResearchAgent using DeepResearchWorkflow

### Fixed
- **Code Extraction Bug**: Fixed regex pattern in `_extract_code()` to properly remove markdown code blocks
  - Changed from `\\n?` (literal backslash-n) to `\s*\n` (actual whitespace and newline)
  - Prevents `SyntaxError: invalid syntax` in generated Python files
- **Method Naming**: Corrected all calls from `generate_content_async` to `generate_content`
  - Fixed in `coordinator.py`, `enhanced_coding_agent.py`, `enhanced_research_agent.py`
- **CodingAgent Integration**: Fixed incorrect `coding_agent.process()` calls
  - Now uses `run_coding_agent()` helper function
  - Properly handles `CodingContext` return type
- **Import Path**: Fixed `src.workflows` → `src.workflow` in EnhancedResearchAgent

### Changed
- Updated system architecture diagram to v2 showing dual-mode capabilities
- Enhanced README with detailed feature descriptions and emoji formatting
- Improved paper search accuracy with better title extraction
- Version bump from 1.0.0 → 1.1.0

### Documentation
- Created comprehensive v1.1.0 update summary
- Updated main README with bug fixes section
- Added detailed architecture diagram showing dual-mode research agent
- Updated task.md with all completed items

---

## [1.0.0] - 2026-02-02

### Added
- **Multi-Agent System**: Coordinator-based architecture with specialized agents
- **Enhanced Research Agent**: RAG-powered paper analysis
  - PDF parsing with PyMuPDF
  - ChromaDB vector indexing
  - 300-500 token intelligent chunking
  - Semantic search with cosine similarity
  - 6-aspect deep Q&A analysis
- **Enhanced Coding Agent**: Large-scale ML project generation
  - PyTorch/TensorFlow/JAX support
  - Complete project structure (8-12 files)
  - Training loops and evaluation scripts
  - Professional documentation
- **Academic Search Integration**:
  - arXiv API for paper search and download
  - Papers with Code integration
  - GitHub repository search
- **RAG System**:
  - ChromaDB vector store wrapper
  - Gemini text-embedding-004 (768D)
  - Smart text chunking preserving context
- **Deployment Ready**:
  - FastAPI server
  - Docker containerization
  - Google Cloud Run deployment scripts

### Features
- Automatic paper reproduction from title/arXiv ID
- Interactive paper Q&A
- Custom ML project generation
- LLM-based task classification
- Multi-agent orchestration
- Error handling and retry logic

### Documentation
- Comprehensive README with examples
- System architecture diagrams
- Implementation plan
- Testing guide
- Deployment instructions

---

## Release Notes

Detailed release notes for each version:
- [v1.1.0 Release Notes](docs/releases/v1.1.0.md)
- [v1.0.0 Release Notes](docs/releases/v1.0.0.md)

## Links
- [GitHub Repository](https://github.com/Ireliaww/AI-Agent)
- [Documentation](deep-research-agent/README.md)
