"""
    deepverify.tools.lit_review.search_google
"""

import os
import sys
import httpx
from pydantic import BaseModel
from rich import print as rprint

from deepverify import config
from deepverify.cache import disk_cache

# --
# Output object

class SearchResult(BaseModel):
    # [TODO] return more fields?
    
    title   : str
    url     : str
    content : str
    
    @classmethod
    def from_serp(cls, result : dict):
        return cls(
            title   = result["title"],
            url     = result["link"],
            content = result.get("snippet", "[MISSING]"),    
        )


class SearchResults(BaseModel):
    query        : str
    results      : list[SearchResult]

# --
# Functions

# [MCP tool]
@disk_cache(cache_dir=config.CACHE_DIR / 'tools/lit_review/search_google', verbose=False)
async def search_google(query : str, _verbose : bool = False) -> SearchResults:
    """
    Search Google
    
    Args:
        query (str): The search query
        
    Returns:
        SearchResults:
            query (str): The search query
            results (list[SearchResult]): A list of search results
                title (str): The title of the search result
                url (str): The URL of the search result
                content (str): Snippet from the search result
    
    """
    # [TODO] expose more parameters?
    
    assert isinstance(query, str)
    
    API_KEY = os.environ.get("SERPAPI_API_KEY")
    if not API_KEY:
        raise Exception("SERPAPI_API_KEY is not set")

    url    = "https://serpapi.com/search.json"
    params = {"q": query, "api_key": API_KEY}
    
    try:
        async with httpx.AsyncClient(timeout=None) as client:
            if _verbose:
                rprint(f"[bright_black]asearch_serp: fetching : {query}[/bright_black]", file=sys.stderr)
            
            res = await client.get(url, params=params)
            if _verbose:
                rprint(f"[bright_black]asearch_serp: fetched  : {query}[/bright_black]", file=sys.stderr)

            if res.status_code != 200:
                rprint(f"[red]ERROR | search_serp: status_code != 200 - {res.status_code}[/red]", file=sys.stderr)
                raise Exception(f"ERROR | search_serp: status_code != 200 - {res.status_code}")
            
            data = res.json()
            if not data:
                rprint(f"[yellow]WARNING | search_serp: data is None[/yellow]", file=sys.stderr)
                return SearchResults(query=query, results=[])
            
            if 'organic_results' not in data:
                rprint(f"[yellow]WARNING | search_serp: organic_results not in data[/yellow]", file=sys.stderr)
                return SearchResults(query=query, results=[])
            
            return SearchResults(
                query   = query,
                results = [
                    SearchResult.from_serp(result) for result in data["organic_results"]
                ]
            )

    except Exception as e:
        rprint(f"[red]ERROR | search_serp: {e}[/red]", file=sys.stderr)
        raise e