"""
Web Search Tool using Brave Search API

Provides web search capabilities for finding blog posts, tutorials,
discussions, and other online resources about papers and implementations.
"""

import os
import requests
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class SearchResult:
    """Web search result"""
    title: str
    url: str
    description: str
    published: Optional[str] = None


class WebSearcher:
    """
    Web search using Brave Search API
    
    Note: Requires BRAVE_API_KEY environment variable
    Get your key at: https://brave.com/search/api/
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize web searcher
        
        Args:
            api_key: Brave Search API key (or set BRAVE_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("BRAVE_API_KEY")
        self.base_url = "https://api.search.brave.com/res/v1/web/search"
    
    async def search(
        self, 
        query: str, 
        count: int = 10,
        freshness: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Search the web using Brave Search API
        
        Args:
            query: Search query
            count: Number of results (max 20)
            freshness: Time filter ('pd' = past day, 'pw' = past week, 
                                   'pm' = past month, 'py' = past year)
        
        Returns:
            List of SearchResult objects
        """
        if not self.api_key:
            print("⚠️  BRAVE_API_KEY not set - using mock results")
            return self._mock_search(query, count)
        
        try:
            headers = {
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": self.api_key
            }
            
            params = {
                "q": query,
                "count": min(count, 20)  # Max 20 results
            }
            
            if freshness:
                params["freshness"] = freshness
            
            response = requests.get(
                self.base_url,
                headers=headers,
                params=params,
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            results = []
            for item in data.get("web", {}).get("results", []):
                result = SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    description=item.get("description", ""),
                    published=item.get("published", None)
                )
                results.append(result)
            
            return results
        
        except Exception as e:
            print(f"⚠️  Web search error: {e}")
            return self._mock_search(query, count)
    
    def _mock_search(self, query: str, count: int) -> List[SearchResult]:
        """
        Mock search results when API key is not available
        
        Args:
            query: Search query
            count: Number of results
        
        Returns:
            List of mock SearchResult objects
        """
        # Return helpful mock results
        return [
            SearchResult(
                title=f"Understanding {query} - Tutorial",
                url=f"https://example.com/tutorial",
                description=f"A comprehensive tutorial on {query} with code examples and explanations."
            ),
            SearchResult(
                title=f"{query} Implementation Guide",
                url=f"https://example.com/guide",
                description=f"Step-by-step guide to implementing {query} from scratch."
            ),
            SearchResult(
                title=f"Discussion: {query}",
                url=f"https://example.com/discussion",
                description=f"Community discussion and insights about {query} and its applications."
            )
        ][:count]
    
    async def search_paper_resources(
        self,
        paper_title: str,
        max_results: int = 10
    ) -> Dict[str, List[SearchResult]]:
        """
        Search for various types of resources about a paper
        
        Args:
            paper_title: Paper title
            max_results: Max results per category
        
        Returns:
            Dictionary with categories of search results
        """
        results = {
            "tutorials": [],
            "blog_posts": [],
            "discussions": [],
            "videos": []
        }
        
        # Combined search query to avoid rate limits (429)
        # We search once and categorize results client-side
        print(f"  🔍 Searching for resources about: {paper_title}")
        
        search_query = f'"{paper_title}" tutorials implementation explanation blog reddit'
        
        # Get more results to filter from
        all_results = await self.search(
            search_query,
            count=20  # Fetch more to categorize
        )
        
        # Categorize results
        for result in all_results:
            lower_title = result.title.lower()
            lower_desc = result.description.lower()
            lower_url = result.url.lower()
            
            # Helper to check keywords
            def has_keyword(text, keywords):
                return any(k in text for k in keywords)
            
            # Categorize based on keywords in URL/Title/Description
            if has_keyword(lower_url, ["reddit", "news.ycombinator.com", "forum", "discuss"]) or \
               has_keyword(lower_title, ["discussion", "thread", "comment"]):
                results["discussions"].append(result)
                
            elif has_keyword(lower_title, ["tutorial", "guide", "how to", "walkthrough", "step-by-step"]) or \
                 has_keyword(lower_desc, ["tutorial", "guide", "implementation"]):
                results["tutorials"].append(result)
                
            elif has_keyword(lower_url, ["medium.com", "blog", "substack"]) or \
                 has_keyword(lower_title, ["explained", "understanding", "overview", "review"]):
                results["blog_posts"].append(result)
            
            else:
                # Default to blog posts if it looks like content
                results["blog_posts"].append(result)
        
        # Limit results per category
        for key in results:
            results[key] = results[key][:max_results]
            
        print(f"  ✓ Found {sum(len(l) for l in results.values())} resources")
        
        return results


if __name__ == "__main__":
    import asyncio
    
    async def test_search():
        searcher = WebSearcher()
        
        # Test basic search
        print("Testing basic search...")
        results = await searcher.search("transformer attention mechanism", count=3)
        
        print(f"\nFound {len(results)} results:")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result.title}")
            print(f"   {result.url}")
            print(f"   {result.description[:100]}...")
            print()
        
        # Test paper resource search
        print("\nTesting paper resource search...")
        resources = await searcher.search_paper_resources(
            "Attention is All You Need",
            max_results=2
        )
        
        for category, items in resources.items():
            print(f"\n{category.upper()}:")
            for item in items:
                print(f"  - {item.title}")
    
    asyncio.run(test_search())
