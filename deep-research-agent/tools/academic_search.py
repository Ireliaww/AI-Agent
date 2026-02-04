"""
Academic Search Tools

Provides search capabilities for academic papers from:
- arXiv
- Papers with Code  
- GitHub repositories
- Semantic Scholar (future)
"""

import arxiv
import requests
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Paper:
    """Academic paper metadata"""
    title: str
    authors: List[str]
    abstract: str
    url: str
    pdf_url: Optional[str] = None
    published: Optional[datetime] = None
    arxiv_id: Optional[str] = None
    citations: int = 0
    
    def __str__(self):
        return f"{self.title} ({', '.join(self.authors[:3])})"


@dataclass
class CodeImplementation:
    """Code implementation of a paper"""
    paper_title: str
    repo_url: str
    repo_name: str
    stars: int
    description: str
    framework: str = ""


class AcademicSearchTools:
    """
    Unified interface for academic search
    
    Supports:
    - arXiv paper search
    - Papers with Code search
    - GitHub code search
    """
    
    def __init__(self):
        self.arxiv_max_results = 10
        self.github_token = None  # Set if available
    
    async def search_arxiv(
        self, 
        query: str, 
        max_results: int = 10,
        sort_by: str = "relevance"
    ) -> List[Paper]:
        """
        Search arXiv for papers
        
        Args:
            query: Search query
            max_results: Maximum number of results
            sort_by: Sort criterion ('relevance', 'lastUpdatedDate', 'submittedDate')
            
        Returns:
            List of Paper objects
        """
        try:
            # Map sort_by to arxiv sort criterion
            sort_map = {
                "relevance": arxiv.SortCriterion.Relevance,
                "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
                "submittedDate": arxiv.SortCriterion.SubmittedDate
            }
            sort_criterion = sort_map.get(sort_by, arxiv.SortCriterion.Relevance)
            
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=sort_criterion
            )
            
            papers = []
            for result in search.results():
                paper = Paper(
                    title=result.title,
                    authors=[author.name for author in result.authors],
                    abstract=result.summary,
                    url=result.entry_id,
                    pdf_url=result.pdf_url,
                    published=result.published,
                    arxiv_id=result.get_short_id()
                )
                papers.append(paper)
            
            return papers
        
        except Exception as e:
            print(f"Error searching arXiv: {e}")
            return []
    
    async def get_arxiv_paper(self, arxiv_id: str) -> Optional[Paper]:
        """
        Get a specific paper from arXiv by ID
        
        Args:
            arxiv_id: arXiv ID (e.g., "2103.00000" or "1706.03762")
            
        Returns:
            Paper object or None
        """
        try:
            search = arxiv.Search(id_list=[arxiv_id])
            result = next(search.results())
            
            return Paper(
                title=result.title,
                authors=[author.name for author in result.authors],
                abstract=result.summary,
                url=result.entry_id,
                pdf_url=result.pdf_url,
                published=result.published,
                arxiv_id=result.get_short_id()
            )
        except Exception as e:
            print(f"Error fetching arXiv paper {arxiv_id}: {e}")
            return None
    
    async def download_arxiv_pdf(self, arxiv_id: str, output_path: str) -> bool:
        """
        Download PDF from arXiv
        
        Args:
            arxiv_id: arXiv ID
            output_path: Where to save the PDF
            
        Returns:
            True if successful
        """
        try:
            paper = await self.get_arxiv_paper(arxiv_id)
            if not paper or not paper.pdf_url:
                return False
            
            # Download PDF
            response = requests.get(paper.pdf_url)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            return True
        except Exception as e:
            print(f"Error downloading PDF: {e}")
            return False
    
    async def search_papers_with_code(
        self, 
        query: str,
        max_results: int = 10
    ) -> List[CodeImplementation]:
        """
        Search Papers with Code for implementations
        
        Args:
            query: Search query (paper title or topic)
            max_results: Maximum results
            
        Returns:
            List of CodeImplementation objects
        """
        try:
            url = "https://paperswithcode.com/api/v1/papers/"
            params = {"q": query, "items_per_page": max_results}
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            implementations = []
            for item in data.get('results', []):
                # Get repositories for this paper
                paper_id = item.get('id')
                if paper_id:
                    repos = await self._get_pwc_repositories(paper_id)
                    implementations.extend(repos)
            
            return implementations[:max_results]
        
        except Exception as e:
            print(f"Error searching Papers with Code: {e}")
            return []
    
    async def _get_pwc_repositories(self, paper_id: str) -> List[CodeImplementation]:
        """Get repositories for a Papers with Code paper"""
        try:
            url = f"https://paperswithcode.com/api/v1/papers/{paper_id}/repositories/"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            repos = []
            for repo in data.get('results', []):
                impl = CodeImplementation(
                    paper_title=repo.get('paper', {}).get('title', ''),
                    repo_url=repo.get('url', ''),
                    repo_name=repo.get('name', ''),
                    stars=repo.get('stars', 0),
                    description=repo.get('description', ''),
                    framework=repo.get('framework', '')
                )
                repos.append(impl)
            
            return repos
        except Exception as e:
            print(f"Error fetching repositories: {e}")
            return []
    
    async def search_github_code(
        self,
        query: str,
        language: str = "python",
        max_results: int = 10
    ) -> List[Dict]:
        """
        Search GitHub for code repositories
        
        Args:
            query: Search query (e.g., paper title + "implementation")
            language: Programming language filter
            max_results: Maximum results
            
        Returns:
            List of repository information
        """
        try:
            url = "https://api.github.com/search/repositories"
            params = {
                "q": f"{query} language:{language}",
                "sort": "stars",
                "order": "desc",
                "per_page": max_results
            }
            
            headers = {}
            if self.github_token:
                headers["Authorization"] = f"token {self.github_token}"
            
            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            repos = []
            for item in data.get('items', []):
                repos.append({
                    "name": item.get('full_name'),
                    "url": item.get('html_url'),
                    "description": item.get('description', ''),
                    "stars": item.get('stargazers_count', 0),
                    "language": item.get('language', ''),
                    "updated": item.get('updated_at', '')
                })
            
            return repos
        
        except Exception as e:
            print(f"Error searching GitHub: {e}")
            return []


if __name__ == "__main__":
    import asyncio
    
    async def test_search():
        tools = AcademicSearchTools()
        
        # Test arXiv search
        print("Searching arXiv for 'transformer attention'...")
        papers = await tools.search_arxiv("transformer attention", max_results=3)
        
        print(f"\nFound {len(papers)} papers:")
        for i, paper in enumerate(papers, 1):
            print(f"{i}. {paper.title}")
            print(f"   Authors: {', '.join(paper.authors[:3])}")
            print(f"   arXiv: {paper.arxiv_id}")
            print()
        
        # Test Papers with Code
        print("\nSearching Papers with Code for 'BERT'...")
        implementations = await tools.search_papers_with_code("BERT", max_results=3)
        
        print(f"\nFound {len(implementations)} implementations:")
        for i, impl in enumerate(implementations, 1):
            print(f"{i}. {impl.repo_name}")
            print(f"   Stars: {impl.stars}")
            print(f"   URL: {impl.repo_url}")
            print()
    
    asyncio.run(test_search())
