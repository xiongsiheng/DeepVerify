import os
import aiohttp
import logging

from pydantic import BaseModel
from typing import Dict, Optional, Any, List


logger = logging.getLogger(__name__)

class SerpAPIGoogleScholarError(Exception):
    """Exception raised for SerpAPI Google Scholar-related errors."""
    pass

class GoogleScholarResult(BaseModel):
    title      : str
    authors    : list[str]
    year       : Optional[int] = None
    venue      : str
    citations  : int
    url        : Optional[str] = None
    abstract   : str
    source_id  : str

class GoogleScholarResponse(BaseModel):
    status        : str
    query         : str
    message       : str
    error_details : Optional[str] = None
    source        : str
    num_results   : int
    results       : list[GoogleScholarResult]


async def _fetch_serpapi_results(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fetch search results from SerpAPI for a single page.
    
    Args:
        params: Search parameters for SerpAPI
        
    Returns:
        Complete SerpAPI response as a dictionary
        
    Raises:
        SerpAPIGoogleScholarError: If an error occurs with the search
    """
    base_url = "https://serpapi.com/search"
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(base_url, params=params) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise SerpAPIGoogleScholarError(f"SerpAPI returned status code {response.status}: {error_text}")
                
                data = await response.json()
                
                # Check for errors in the response
                if "error" in data:
                    raise SerpAPIGoogleScholarError(f"SerpAPI error: {data['error']}")
                
                return data
                
        except aiohttp.ClientError as e:
            raise SerpAPIGoogleScholarError(f"Error connecting to SerpAPI: {str(e)}")



def _format_serpapi_results(query: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Format the SerpAPI results to match the expected structure of the lit_review module.
    
    Args:
        query: The original search query
        results: Raw results from SerpAPI
        
    Returns:
        Formatted results dictionary
    """
    formatted_results = []
    
    for item in results:
        try:
            # Extract publication year
            year = None
            if "publication_info" in item:
                # Try to extract year from publication info
                pub_info = item["publication_info"]
                if isinstance(pub_info, dict):
                    # Try formats like "2019", "Mar 2019", etc.
                    year_text = pub_info.get("summary", "")
                    # Extract 4-digit year
                    import re
                    year_match = re.search(r'\b(19|20)\d{2}\b', year_text)
                    if year_match:
                        year = int(year_match.group(0))
            
            # Extract authors
            authors = []
            if "publication_info" in item and "authors" in item["publication_info"]:
                author_data = item["publication_info"]["authors"]
                if isinstance(author_data, list):
                    authors = [author.get("name", "Unknown") for author in author_data]
                else:
                    # Sometimes it's just a string
                    authors = [str(author_data)]
            
            # Create formatted result
            formatted_results.append({
                "title"     : item.get("title", "Unknown Title"),
                "authors"   : authors,
                "year"      : year,
                "venue"     : item.get("publication_info", {}).get("summary", None),
                "citations" : item.get("inline_links", {}).get("cited_by", {}).get("total", 0),
                "url"       : item.get("link"),
                "abstract"  : item.get("snippet"),
                "source_id" : item.get("result_id")
            })
        except Exception as e:
            # Skip this result if there's an issue formatting it
            logger.warning(f"Error formatting result: {str(e)}")
            continue

    return GoogleScholarResponse(**{
        "status"        : "success",
        "message"       : "success",
        "error_details" : "None",
        "query"         : query,
        "num_results"   : len(formatted_results),
        "results"       : formatted_results,
        "source"        : "Google Scholar (SerpAPI)"
    })


async def _search_serpapi_google_scholar(
    query: str, 
    max_results: int = 10, 
    starting_year: Optional[int] = None,
    page: int = 0,
    api_key: Optional[str] = None,
    progress_callback: Optional[callable] = None,
    preserve_raw_response: bool = False
) -> Dict[str, Any]:
    """
    Search Google Scholar using SerpAPI for articles matching a query.
    
    Args:
        query: The search query
        max_results: Maximum number of results to return per page
        starting_year: Only include papers published on or after this year
        page: Page number to retrieve (0-based, will be converted to SerpAPI's start parameter)
        api_key: SerpAPI key (defaults to SERPAPI_KEY environment variable)
        progress_callback: Optional callback for progress updates
        preserve_raw_response: If True, return the complete SerpAPI response
        
    Returns:
        Dictionary containing search results (either raw SerpAPI response or formatted results)
        
    Raises:
        SerpAPIGoogleScholarError: If an error occurs with the search
    """
    try:
        # Get API key from parameter or environment variable
        serpapi_key = api_key or os.getenv("SERPAPI_KEY")
        if not serpapi_key:
            # Try alternative environment variable name
            serpapi_key = os.getenv("SERPAPI_API_KEY")
            
        if not serpapi_key:
            raise SerpAPIGoogleScholarError("No SerpAPI key provided. Set SERPAPI_KEY environment variable or pass api_key parameter.")
        
        logger.info(f"Searching Google Scholar via SerpAPI for: {query}")
        if progress_callback:
            progress_callback(f"Starting Google Scholar search for: {query}")
        
        # Calculate start parameter for pagination
        start = page * 10  # Google Scholar uses increments of 10 for pagination
        
        # Prepare SerpAPI parameters
        params = {
            "engine": "google_scholar",
            "q": query,
            "api_key": serpapi_key,
            "num": min(max_results, 20),  # SerpAPI only supports up to 20 results per page
        }
        
        # Add start parameter for pagination if not on first page
        if start > 0:
            params["start"] = start
        
        # Add year filter if specified
        if starting_year is not None:
            params["as_ylo"] = starting_year
        
        # Make the request
        serpapi_response = await _fetch_serpapi_results(params)
        
        # Return raw response if requested
        if preserve_raw_response:
            return serpapi_response
        
        # Otherwise format the results in a way that's compatible with the existing API
        organic_results = serpapi_response.get("organic_results", [])
        return _format_serpapi_results(query, organic_results)
            
    except SerpAPIGoogleScholarError as e:
        logger.error(f"SerpAPI Google Scholar error: {str(e)}")
        return GoogleScholarResponse(**{
            "status"      : "error",
            "query"       : query,
            "message"     : str(e),
            "source"      : "Google Scholar (SerpAPI)",
            "num_results" : 0,
            "results"     : []
        })
    except Exception as e:
        logger.error(f"Unexpected error in SerpAPI Google Scholar search: {str(e)}")
        return GoogleScholarResponse(**{
            "status"        : "error",
            "query"         : query,
            "message"       : f"Unexpected error in Google Scholar search: {str(e)}",
            "error_details" : str(e),
            "source"        : "Google Scholar (SerpAPI)",
            "num_results"   : 0,
            "results"       : []
        })

# [MCP tool]
async def search_google_scholar(query: str, max_results: int = 10, starting_year: Optional[int] = None, page: int = 0) -> GoogleScholarResponse:
    """
    Search Google Scholar for scholarly articles using SerpAPI.
    
    This tool searches Google Scholar for academic papers matching the query and 
    returns the raw SerpAPI response with all metadata preserved, including PDF links,
    citation counts, and related searches.
    
    Note: Requires a SerpAPI key to be set as an environment variable.
    
    Args:
        query (str): The search query
        max_results (int): Maximum number of results to return per page (default: 10)
        starting_year (int, optional): Only include papers published on or after this year
        page (int): Page number to retrieve (0-based, default: 0)
        
    Returns:
        Dict: Complete SerpAPI response including organic results, pagination info, and related searches
        
    Examples:
        # Basic search for machine learning papers
        search_google_scholar("machine learning")
        
        # Search for quantum computing papers published since 2020
        search_google_scholar("quantum computing", starting_year=2020)
        
        # Get the second page of results
        search_google_scholar("neural networks", page=1)
    """
    return await _search_serpapi_google_scholar(
        query=query,
        max_results=max_results,
        starting_year=starting_year,
        page=page,
        preserve_raw_response=False  # Return the complete raw response - [BKJ] switched from True to False
    )
