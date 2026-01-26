"""
Tools module for AI Agent.

Contains file operations, shell execution, and search tools.
"""

from .file_tools import save_solution, run_solution
from .search_tools import search_web

__all__ = ["save_solution", "run_solution", "search_web"]
