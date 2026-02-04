"""
RAG module for deep-research-agent

Provides RAG capabilities using ChromaDB and Gemini embeddings:
- Vector storage with semantic search
- Intelligent text chunking (300-500 tokens)
- Paper indexing and querying
"""

from .vector_store.chroma_store import ChromaVectorStore
from .pdf_processor.text_chunker import TextChunker

__all__ = ['ChromaVectorStore', 'TextChunker']
