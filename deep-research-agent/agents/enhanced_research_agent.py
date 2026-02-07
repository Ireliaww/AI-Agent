"""
Enhanced Research Agent with RAG capabilities

Provides advanced paper analysis using:
- PDF parsing and indexing
- Semantic search via ChromaDB  
- Deep paper understanding
- Academic search integration
"""

import os
import hashlib
import tempfile
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

from tools.pdf_parser import PDFParser, PaperContent
from tools.academic_search import AcademicSearchTools, Paper, CodeImplementation
from rag.vector_store.chroma_store import ChromaVectorStore
from rag.pdf_processor.text_chunker import TextChunker
from src.client import GeminiClient

# Import DeepResearchWorkflow for general research
from src.workflow import DeepResearchWorkflow, ResearchContext
from src.agents.researcher import ResearchResult


@dataclass
class PaperUnderstanding:
    """Deep understanding of a paper using RAG"""
    contributions: str
    methodology: str
    key_equations: List[str] = field(default_factory=list)
    experiments: str = ""
    results: str = ""
    limitations: str = ""
    qa_details: Dict = field(default_factory=dict)  # Detailed Q&A records
    
    def get_summary(self) -> str:
        """Get a concise summary"""
        return f"""**Main Contributions:**
{self.contributions}

**Methodology:**
{self.methodology}

**Experiments & Results:**
{self.experiments}
{self.results}

**Limitations:**
{self.limitations}
"""


@dataclass
class PaperAnalysis:
    """Complete paper analysis result"""
    content: PaperContent  # Parsed content
    understanding: PaperUnderstanding  # RAG-enhanced understanding
    related_papers: List[Paper] = field(default_factory=list)
    implementations: List[CodeImplementation] = field(default_factory=list)
    vector_store: Optional[ChromaVectorStore] = None
    collection_name: str = ""


class EnhancedResearchAgent:
    """
    RAG-Enhanced Research Agent for academic paper analysis
    
    Features:
    - PDF paper parsing and indexing
    - Semantic search within papers
    - Deep understanding via targeted Q&A
    - Academic search (arXiv, Papers with Code, GitHub)
    - Multi-paper comparison (future)
    """
    
    def __init__(self, gemini_client: GeminiClient, use_mock: bool = False):
        self.gemini = gemini_client
        self.use_mock = use_mock
        self.pdf_parser = PDFParser()
        self.academic_search = AcademicSearchTools()
        self.text_chunker = TextChunker(min_tokens=300, max_tokens=500)
        
        # Current paper's vector store (set during analysis)
        self.current_vector_store = None
        self.current_collection_name = None
        
        # Workflow for general research
        self._workflow = None
    
    
    def _extract_paper_title(self, user_request: str) -> str:
        """
        Extract paper identifier from user request
        
        Supports:
        - arXiv IDs: "1706.03762", "arxiv:1706.03762"
        - arXiv URLs: "https://arxiv.org/abs/1706.03762"
        - Quoted titles: '"Attention Is All You Need"'
        - Natural language: "Reproduce BERT paper"
        
        Returns:
            Paper identifier (arXiv ID, URL, or title)
        """
        import re
        
        request_lower = user_request.lower().strip()
        
        # Pattern 0: arXiv URLs (highest priority - most precise)
        arxiv_url_patterns = [
            r'(?:https?://)?arxiv\.org/(?:abs|pdf)/([\d\.]+)',
            r'(?:https?://)?ar5iv\.org/(?:abs|pdf)/([\d\.]+)',
        ]
        for pattern in arxiv_url_patterns:
            match = re.search(pattern, user_request, re.IGNORECASE)
            if match:
                arxiv_id = match.group(1)
                print(f"✓ Detected arXiv ID from URL: {arxiv_id}")
                return arxiv_id
        
        # Pattern 1: arXiv IDs (e.g., "1706.03762", "arxiv:1706.03762")
        arxiv_id_match = re.search(r'(?:arxiv:?\s*)?([\d]{4}\.[\d]{4,5})', user_request, re.IGNORECASE)
        if arxiv_id_match:
            arxiv_id = arxiv_id_match.group(1)
            print(f"✓ Detected arXiv ID: {arxiv_id}")
            return arxiv_id
        
        # Pattern 2: Text in quotes (high confidence)
        quoted = re.findall(r'["\u2018\u2019\u201c\u201d]([^"\u2018\u2019\u201c\u201d]+)["\u2018\u2019\u201c\u201d]', user_request)
        if quoted:
            title = quoted[0].strip()
            print(f"✓ Extracted title from quotes: '{title}'")
            return title
        
        # Pattern 3: "reproduce/implement X paper" -> X
        patterns = [
            r'reproduce\s+(?:the\s+)?["\u2018\u201c]?([^"\u2018\u201c\u201d\u2019]+?)["\u2019\u201d]?\s+(?:paper|model|architecture)',
            r'implement\s+(?:the\s+)?["\u2018\u201c]?([^"\u2018\u201c\u201d\u2019]+?)["\u2019\u201d]?\s+(?:paper|model|architecture|from)',
            r'analyze\s+(?:the\s+)?["\u2018\u201c]?([^"\u2018\u201c\u201d\u2019]+?)["\u2019\u201d]?\s+(?:paper|from)',
            r'code\s+(?:the\s+)?["\u2018\u201c]?([^"\u2018\u201c\u201d\u2019]+?)["\u2019\u201d]?\s+(?:paper|model)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, user_request, re.IGNORECASE)
            if match:
                title = match.group(1).strip()
                print(f"✓ Extracted title from pattern: '{title}'")
                return title
        
        # Fallback: return entire request (will be searched)
        print(f"⚠ Using full query for search: '{user_request}'")
        return user_request
    
    async def research(self, question: str) -> ResearchResult:
        """
        Conduct general research (non-paper specific)
        
        This provides the same capability as the original ResearchAgent.
        
        Args:
            question: The research question
            
        Returns:
            ResearchResult with findings
        """
        # Create and run the deep research workflow
        workflow = DeepResearchWorkflow(
            gemini_client=self.gemini,
            use_mock=self.use_mock,
        )
        
        context = await workflow.run(question)
        
        # Convert to ResearchResult
        return ResearchResult(
            question=question,
            report=context.final_report,
            queries_used=context.search_queries,
            total_results=sum(len(r.results) for r in context.search_results),
            iterations=context.iteration,
            success=context.error is None,
            error=context.error,
        )
    
    def _create_collection_name(self, title: str) -> str:
        """Create a unique collection name from paper title"""
        # Use hash to create short unique ID
        title_hash = hashlib.md5(title.encode()).hexdigest()[:8]
        # Sanitize title for collection name
        safe_title = "".join(c if c.isalnum() else "_" for c in title[:30])
        return f"paper_{safe_title}_{title_hash}".lower()
    
    async def _get_paper(self, paper_input: str) -> Tuple[Optional[Paper], str]:
        """
        Get paper and return (Paper object, pdf_path)
        
        Args:
            paper_input: PDF path, arXiv ID, or arxiv URL, or paper title
            
        Returns:
            (Paper metadata, local PDF path)
        """
        # Case 1: Local PDF file
        if paper_input.endswith('.pdf') and os.path.exists(paper_input):
            return None, paper_input
        
        # Case 2: arXiv ID or URL
        if 'arxiv.org' in paper_input or paper_input.replace('.', '').isdigit():
            # Extract arXiv ID
            arxiv_id = paper_input.split('/')[-1].replace('.pdf', '')
            
            # Get paper metadata
            paper = await self.academic_search.get_arxiv_paper(arxiv_id)
            if not paper:
                raise ValueError(f"Could not find arXiv paper: {arxiv_id}")
            
            # Download PDF
            pdf_path = os.path.join(tempfile.gettempdir(), f"{arxiv_id}.pdf")
            success = await self.academic_search.download_arxiv_pdf(arxiv_id, pdf_path)
            
            if not success:
                raise ValueError(f"Failed to download PDF for {arxiv_id}")
            
            return paper, pdf_path
        
        # Case 3: Search by title
        else:
            print(f"🔍 Searching arXiv for: '{paper_input}'")
            papers = await self.academic_search.search_arxiv(paper_input, max_results=3)
            if not papers:
                raise ValueError(f"Could not find paper: {paper_input}")
            
            # Show top results for user verification
            print(f"\n📚 Found {len(papers)} paper(s):")
            for i, p in enumerate(papers[:3], 1):
                print(f"  {i}. {p.title}")
                print(f"     arXiv:{p.arxiv_id} | {p.authors[0] if p.authors else 'Unknown'}")
            
            # Use first result (most relevant)
            paper = papers[0]
            arxiv_id = paper.arxiv_id
            print(f"\n✓ Using: {paper.title}")
            print(f"  arXiv ID: {arxiv_id}")
            
            # Download PDF
            pdf_path = os.path.join(tempfile.gettempdir(), f"{arxiv_id}.pdf")
            success = await self.academic_search.download_arxiv_pdf(arxiv_id, pdf_path)
            
            if not success:
                raise ValueError(f"Failed to download PDF")
            
            return paper, pdf_path
    
    async def analyze_paper(
        self,
        paper_input: str,
        create_index: bool = True,
        deep_analysis: bool = True,
        artifact_manager = None  # New: Optional artifact manager for saving analysis
    ) -> PaperAnalysis:
        """
        Deeply analyze a paper with RAG
        
        Args:
            paper_input: PDF path, arXiv ID, or paper title
            create_index: Whether to create vector index
            deep_analysis: Whether to perform deep Q&A analysis
            artifact_manager: Optional ArtifactManager to save artifacts
            
        Returns:
            Complete paper analysis
        """
        print(f"📄 Analyzing paper: {paper_input}")
        
        # Extract paper title if request is verbose
        paper_query = self._extract_paper_title(paper_input)
        print(f"🔍 Extracted query: {paper_query}")
        
        # Step 1: Get paper PDF
        paper_meta, pdf_path = await self._get_paper(paper_query)
        
        # Step 2: Parse PDF
        print("📖 Parsing PDF...")
        paper_content = self.pdf_parser.parse(pdf_path)
        print(f"✓ Parsed: {paper_content.title}")
        print(f"  Authors: {', '.join(paper_content.authors[:3])}")
        print(f"  Sections: {len(paper_content.sections)}")
        
        # Step 3: Create RAG index
        collection_name = None
        if create_index:
            print("\n🔍 Creating vector index...")
            collection_name = self._create_collection_name(paper_content.title)
            self.current_collection_name = collection_name
            
            self.current_vector_store = ChromaVectorStore(
                collection_name=collection_name,
                persist_directory="chroma_db_papers"
            )
            
            # Convert sections to pages format
            pages = []
            for i, (section_name, section_text) in enumerate(paper_content.sections.items()):
                if section_text.strip():  # Only add non-empty sections
                    pages.append({
                        "page_number": i + 1,
                        "text": f"## {section_name}\n\n{section_text}"
                    })
            
            # Chunk and index
            if pages:
                chunks = self.text_chunker.chunk_text(pages)
                self.current_vector_store.add_chunks(
                    chunks, 
                    source_file=paper_content.title
                )
                print(f"✓ Indexed {len(chunks)} chunks")
            else:
                print("⚠ No sections to index, falling back to full text")
                # Fallback: use full text
                pages = [{"page_number": 1, "text": paper_content.full_text[:10000]}]
                chunks = self.text_chunker.chunk_text(pages)
                self.current_vector_store.add_chunks(chunks, source_file=paper_content.title)
        
        # Step 4: Deep understanding via RAG Q&A
        understanding = None
        if deep_analysis and self.current_vector_store:
            print("\n🧠 Performing deep analysis...")
            understanding = await self._understand_paper_with_rag(
                paper_content,
                self.current_vector_store
            )
        else:
            # Basic understanding without RAG
            understanding = PaperUnderstanding(
                contributions="See abstract",
                methodology="See paper methodology section",
                key_equations=paper_content.equations[:5],
                experiments="See paper experiments section",
                results="See paper results section"
            )
        
        # Step 5: Search related work
        print("\n🔗 Searching related work...")
        related_papers = await self.academic_search.search_arxiv(
            paper_content.title,
            max_results=5
        )
        
        implementations = await self.academic_search.search_papers_with_code(
            paper_content.title
        )
        
        
        print(f"✓ Found {len(related_papers)} related papers")
        if related_papers:
            for i, paper in enumerate(related_papers[:3], 1):
                print(f"  {i}. {paper.title}")
                print(f"     Authors: {', '.join(paper.authors[:3])}")
                print(f"     arXiv: {paper.arxiv_id}")
        
        print(f"✓ Found {len(implementations)} code implementations")
        if implementations:
            for i, impl in enumerate(implementations[:3], 1):
                print(f"  {i}. {impl.repo_name} ({impl.stars} ⭐)")
                print(f"     URL: {impl.repo_url}")
        
        
        # Save artifacts if artifact_manager provided
        if artifact_manager and understanding:
            print("\n💾 Saving analysis artifacts...")
            
            # Prepare RAG query results for artifact
            rag_queries = []
            if hasattr(understanding, 'qa_details') and understanding.qa_details:
                for question, qa_result in understanding.qa_details.items():
                    rag_queries.append({
                        'query': question,
                        'chunks_found': len(qa_result.get('chunks', [])),
                        'chunks': [
                            {
                                'text': chunk.get('text', '')[:300],
                                'similarity': chunk.get('score', 0.0)
                            }
                            for chunk in qa_result.get('chunks', [])
                        ],
                        'analysis': qa_result.get('answer', '')
                    })
            
            # Save paper analysis artifact
            try:
                artifact_path = artifact_manager.save_paper_analysis(
                    title=paper_content.title,
                    authors=', '.join(paper_content.authors[:5]),
                    arxiv_id=paper_meta.arxiv_id if paper_meta else "Unknown",
                    sections=len(paper_content.sections),
                    chunks=len(chunks) if 'chunks' in locals() else 0,
                    rag_queries=rag_queries,
                    understanding={
                        'contributions': understanding.contributions,
                        'methodology': understanding.methodology,
                        'experiments': understanding.experiments
                    },
                    confidence=95.0
                )
                print(f"✓ Saved: {artifact_path}")
            except Exception as e:
                print(f"⚠️ Failed to save artifact: {e}")
        
        return PaperAnalysis(
            content=paper_content,
            understanding=understanding,
            related_papers=related_papers,
            implementations=implementations,
            vector_store=self.current_vector_store,
            collection_name=collection_name or ""
        )
    
    async def _understand_paper_with_rag(
        self,
        paper_content: PaperContent,
        vector_store: ChromaVectorStore
    ) -> PaperUnderstanding:
        """Use RAG to deeply understand the paper"""
        
        # Key questions to ask
        key_questions = [
            "What are the main contributions of this paper?",
            "What is the proposed methodology or algorithm?",
            "What are the key mathematical formulations or equations?",
            "What datasets and evaluation metrics were used in experiments?",
            "What are the main experimental results and findings?",
            "What limitations are mentioned in the paper?"
        ]
        
        qa_results = {}
        
        for question in key_questions:
            print(f"  ? {question}")
            
            # Semantic search for relevant chunks
            relevant_chunks = vector_store.similarity_search(
                query=question,
                k=3,
                threshold=0.6
            )
            
            if relevant_chunks:
                # Build context from chunks
                context = "\n\n".join([
                    f"[Section {chunk.get('page_number', '?')}] {chunk['text']}"
                    for chunk in relevant_chunks
                ])
                
                # Ask LLM with context
                answer = await self._ask_llm_with_context(question, context)
                qa_results[question] = {
                    "answer": answer,
                    "chunks": relevant_chunks
                }
            else:
                qa_results[question] = {
                    "answer": "Information not found in the paper.",
                    "chunks": []
                }
        
        # Extract and structure understanding
        return PaperUnderstanding(
            contributions=qa_results[key_questions[0]]["answer"],
            methodology=qa_results[key_questions[1]]["answer"],
            key_equations=self._extract_equations_from_context(qa_results[key_questions[2]]),
            experiments=qa_results[key_questions[3]]["answer"],
            results=qa_results[key_questions[4]]["answer"],
            limitations=qa_results[key_questions[5]]["answer"],
            qa_details=qa_results
        )
    
    async def _ask_llm_with_context(self, question: str, context: str) -> str:
        """Ask LLM a question with provided context"""
        prompt = f"""Based on the following context from a research paper, answer the question.

CONTEXT:
{context}

QUESTION: {question}

Provide a clear, concise answer based solely on the context above. If the context doesn't contain enough information, say so.

ANSWER:"""
        
        try:
            response = await self.gemini.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Error generating answer: {e}"
    
    def _extract_equations_from_context(self, qa_result: Dict) -> List[str]:
        """Extract equations from Q&A result"""
        answer = qa_result.get("answer", "")
        chunks = qa_result.get("chunks", [])
        
        equations = []
        
        # Look for equations in answer
        import re
        # Simple pattern for equations
        eq_pattern = r'\$(.*?)\$|\\\[(.*?)\\\]'
        found_eqs = re.findall(eq_pattern, answer)
        for eq_tuple in found_eqs:
            eq = eq_tuple[0] or eq_tuple[1]
            if eq:
                equations.append(eq)
        
        return equations[:10]  # Limit to 10
    
    async def query_paper(
        self,
        question: str,
        vector_store: Optional[ChromaVectorStore] = None
    ) -> str:
        """
        Ask a question about the indexed paper
        
        Args:
            question: User's question
            vector_store: Vector store to query (uses current if None)
            
        Returns:
            Answer based on paper content
        """
        if vector_store is None:
            vector_store = self.current_vector_store
        
        if vector_store is None:
            return "No paper indexed. Please analyze a paper first using analyze_paper()."
        
        # Retrieve relevant chunks
        chunks = vector_store.similarity_search(question, k=5, threshold=0.6)
        
        if not chunks:
            return "I couldn't find relevant information in the paper for this question."
        
        # Build context
        context = "\n\n".join([
            f"[Section {chunk.get('page_number', '?')}] {chunk['text'][:500]}"
            for chunk in chunks
        ])
        
        # Generate answer
        answer = await self._ask_llm_with_context(question, context)
        
        return answer


if __name__ == "__main__":
    import asyncio
    from src.client import GeminiClient
    
    async def test():
        client = GeminiClient()
        agent = EnhancedResearchAgent(client)
        
        # Test with a well-known paper
        print("Testing with Attention Is All You Need paper...")
        analysis = await agent.analyze_paper(
            "Attention Is All You Need",
            create_index=True,
            deep_analysis=True
        )
        
        print("\n" + "="*50)
        print("ANALYSIS RESULTS:")
        print("="*50)
        print(analysis.understanding.get_summary())
        
        # Test querying
        print("\n" + "="*50)
        print("TESTING QUERY:")
        print("="*50)
        answer = await agent.query_paper("How does multi-head attention work?")
        print(f"\nQ: How does multi-head attention work?")
        print(f"A: {answer}")
    
    asyncio.run(test())
