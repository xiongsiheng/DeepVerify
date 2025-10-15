"""
    deepverify.tools.lit_review.search_crossref
"""

import re
import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from habanero import Crossref

from deepverify.datamodels import LitReviewResults

# Configure logging
logger = logging.getLogger(__name__)

class CrossrefError(Exception):
    """Exception raised for Crossref API-related errors."""
    pass

async def _crossref_lookup(
    query: str,
    filter_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Query the Crossref API for information.
    
    Args:
        query: The search query or DOI
        filter_type: Type of lookup - "article", "author", or "doi"
        
    Returns:
        Dictionary containing Crossref data
        
    Raises:
        CrossrefError: If an error occurs during Crossref lookup
    """
    try:
        logger.info(f"Querying Crossref for: {query}")
        
        # Determine the lookup type based on the query and filter_type
        lookup_type = filter_type or _detect_lookup_type(query)
        
        # Run the appropriate lookup function based on type
        if lookup_type == "doi":
            return await _lookup_doi(query)
        elif lookup_type == "author":
            return await _lookup_author(query)
        else:  # Default to article/general search
            return await _lookup_works(query)
        
    except Exception as e:
        logger.error(f"Error in Crossref lookup: {str(e)}")
        raise CrossrefError(f"Failed to query Crossref: {str(e)}")

def _detect_lookup_type(query: str) -> str:
    """
    Detect the type of lookup based on the query.
    
    Args:
        query: The search query
        
    Returns:
        Lookup type string
    """
    # Check if query looks like a DOI
    if re.match(r'^\s*10\.\d{4,}\/\S+\s*$', query):
        return "doi"
    
    # Simple heuristic for author names - contains a space and no special punctuation
    if re.match(r'^[A-Za-z\s\'\-\.]+$', query) and ' ' in query:
        # Check if the query has a typical name format
        if re.match(r'^[A-Za-z\'\-\.]+\s+[A-Za-z\'\-\.]+$', query):
            return "author"
    
    # Default to article/general search
    return "article"

async def _lookup_doi(doi: str) -> Dict[str, Any]:
    """
    Look up a specific DOI in Crossref.
    
    Args:
        doi: The DOI string
        
    Returns:
        Dictionary containing publication data
    """
    try:
        # Clean the DOI string
        doi = doi.strip()
        
        # We need to run the Crossref call in a separate thread since it's synchronous
        # and can block the asyncio event loop
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, 
            lambda: _run_doi_lookup(doi)
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error looking up DOI: {str(e)}")
        raise e

def _run_doi_lookup(doi: str) -> Dict[str, Any]:
    """
    Execute the Crossref DOI lookup synchronously.
    
    Args:
        doi: The DOI string
        
    Returns:
        Dictionary containing publication data
    """
    try:
        cr = Crossref()
        
        # Look up the DOI
        response = cr.works(ids=doi)
        
        if "message" not in response:
            return {"status": "not_found", "query": doi, "source": "Crossref"}
        
        # Extract the relevant information
        message = response["message"]
        
        # Get authors
        authors = []
        if "author" in message:
            for author in message["author"]:
                name_parts = []
                if "given" in author:
                    name_parts.append(author["given"])
                if "family" in author:
                    name_parts.append(author["family"])
                authors.append(" ".join(name_parts))
        
        # Get references
        references = []
        if "reference" in message:
            for ref in message["reference"]:
                if "DOI" in ref:
                    references.append(ref["DOI"])
                    
        # Get publication year
        year = None
        if "published-print" in message and "date-parts" in message["published-print"]:
            year = message["published-print"]["date-parts"][0][0]
        elif "published-online" in message and "date-parts" in message["published-online"]:
            year = message["published-online"]["date-parts"][0][0]
        elif "created" in message and "date-parts" in message["created"]:
            year = message["created"]["date-parts"][0][0]
        
        # Construct result using actual data from the Crossref API
        result = {
            "doi": message.get("DOI", doi),
            "title": message.get("title", ["Unknown Title"])[0] if message.get("title") else "Unknown Title",
            "authors": authors,
            "type": message.get("type", "Unknown type"),
            "container_title": message.get("container-title", ["Unknown Journal"])[0] if message.get("container-title") else None,
            "publisher": message.get("publisher", "Unknown Publisher"),
            "volume": message.get("volume"),
            "issue": message.get("issue"),
            "page": message.get("page"),
            "year": year,
            "references": references[:20] if references else [], # Limit references to 20 for brevity
            "url": message.get("URL"),
            "source": "Crossref"
        }
        
        # Add a summary
        result["summary"] = _generate_publication_summary(result)
        
        return result
        
    except Exception as e:
        logger.error(f"Error in DOI lookup: {str(e)}")
        return {"status": "error", "message": str(e), "query": doi, "source": "Crossref"}

async def _lookup_author(author_name: str) -> Dict[str, Any]:
    """
    Look up works by an author in Crossref.
    
    Args:
        author_name: The name of the author
        
    Returns:
        Dictionary containing author's works
    """
    try:
        # We need to run the Crossref call in a separate thread since it's synchronous
        # and can block the asyncio event loop
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, 
            lambda: _run_author_lookup(author_name)
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error looking up author: {str(e)}")
        raise e

def _run_author_lookup(author_name: str) -> Dict[str, Any]:
    """
    Execute the Crossref author lookup synchronously.
    
    Args:
        author_name: The name of the author
        
    Returns:
        Dictionary containing author's works
    """
    try:
        cr = Crossref()
        
        # Build author query
        query = f'query.author="{author_name}"'
        
        # Look up works by the author, sorted by relevance and limited to 10
        response = cr.works(query=query, sort='relevance', limit=10)
        
        if "message" not in response or "items" not in response["message"]:
            return {"status": "not_found", "query": author_name, "source": "Crossref"}
        
        # Extract works
        works = []
        for item in response["message"]["items"]:
            # Get year
            year = None
            if "published-print" in item and "date-parts" in item["published-print"]:
                year = item["published-print"]["date-parts"][0][0]
            elif "published-online" in item and "date-parts" in item["published-online"]:
                year = item["published-online"]["date-parts"][0][0]
            elif "created" in item and "date-parts" in item["created"]:
                year = item["created"]["date-parts"][0][0]
            
            work = {
                "doi": item.get("DOI"),
                "title": item.get("title", ["Unknown Title"])[0] if item.get("title") else "Unknown Title",
                "container_title": item.get("container-title", ["Unknown Journal"])[0] if item.get("container-title") else None,
                "year": year,
                "type": item.get("type", "Unknown type")
            }
            works.append(work)
        
        # Construct result
        result = {
            "name": author_name,
            "query": author_name,
            "works": works,
            "total_works": response["message"].get("total-results", 0),
            "source": "Crossref"
        }
        
        # Add a summary
        result["summary"] = f"Found {result['total_works']} works by {author_name} in Crossref."
        
        return result
        
    except Exception as e:
        logger.error(f"Error in author lookup: {str(e)}")
        return {"status": "error", "message": str(e), "query": author_name, "source": "Crossref"}

async def _lookup_works(query: str) -> Dict[str, Any]:
    """
    Search for works in Crossref.
    
    Args:
        query: The search query
        
    Returns:
        Dictionary containing search results
    """
    try:
        # We need to run the Crossref call in a separate thread since it's synchronous
        # and can block the asyncio event loop
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, 
            lambda: _run_works_lookup(query)
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error in works lookup: {str(e)}")
        raise e

def _run_works_lookup(query: str) -> Dict[str, Any]:
    """
    Execute the Crossref works lookup synchronously.
    
    Args:
        query: The search query
        
    Returns:
        Dictionary containing search results
    """
    try:
        cr = Crossref()
        
        # Look up works matching the query, sorted by relevance and limited to 10
        response = cr.works(query=query, sort='relevance', limit=10)
        
        if "message" not in response or "items" not in response["message"]:
            return {"status": "not_found", "query": query, "source": "Crossref"}
        
        # Extract results
        results = []
        for item in response["message"]["items"]:
            # Get authors
            authors = []
            if "author" in item:
                for author in item["author"]:
                    name_parts = []
                    if "given" in author:
                        name_parts.append(author["given"])
                    if "family" in author:
                        name_parts.append(author["family"])
                    authors.append(" ".join(name_parts))
            
            # Get year
            year = None
            if "published-print" in item and "date-parts" in item["published-print"]:
                year = item["published-print"]["date-parts"][0][0]
            elif "published-online" in item and "date-parts" in item["published-online"]:
                year = item["published-online"]["date-parts"][0][0]
            elif "created" in item and "date-parts" in item["created"]:
                year = item["created"]["date-parts"][0][0]
            
            # Count citations if available
            citations = item.get("is-referenced-by-count", 0)
            
            result = {
                "doi": item.get("DOI"),
                "title": item.get("title", ["Unknown Title"])[0] if item.get("title") else "Unknown Title",
                "authors": authors,
                "journal": item.get("container-title", ["Unknown Journal"])[0] if item.get("container-title") else None,
                "year": year,
                "type": item.get("type", "Unknown type"),
                "citations": citations
            }
            results.append(result)
        
        # Construct result
        return {
            "query": query,
            "results": results,
            "total_results": response["message"].get("total-results", 0),
            "source": "Crossref"
        }
        
    except Exception as e:
        logger.error(f"Error in works lookup: {str(e)}")
        return {"status": "error", "message": str(e), "query": query, "source": "Crossref"}

def _generate_publication_summary(pub_data: Dict[str, Any]) -> str:
    """
    Generate a summary of publication data.
    
    Args:
        pub_data: Publication data
        
    Returns:
        Summary string
    """
    parts = []
    
    # Title
    title = pub_data.get("title", "Unknown title")
    parts.append(f'"{title}"')
    
    # Authors
    authors = pub_data.get("authors", [])
    if authors:
        if len(authors) == 1:
            parts.append(f"by {authors[0]}")
        elif len(authors) == 2:
            parts.append(f"by {authors[0]} and {authors[1]}")
        else:
            parts.append(f"by {authors[0]} et al.")
    
    # Journal and publication details
    journal = pub_data.get("container_title")
    volume = pub_data.get("volume")
    issue = pub_data.get("issue")
    page = pub_data.get("page")
    year = pub_data.get("year")
    
    pub_details = []
    if journal:
        pub_details.append(journal)
    if volume:
        pub_details.append(f"Vol. {volume}")
    if issue:
        pub_details.append(f"Issue {issue}")
    if page:
        pub_details.append(f"pp. {page}")
    if year:
        pub_details.append(f"({year})")
    
    if pub_details:
        parts.append(", ".join(pub_details))
    
    # DOI
    doi = pub_data.get("doi")
    if doi:
        parts.append(f"DOI: {doi}")
    
    return ". ".join(parts)


# [MCP tool]
async def search_crossref(query: str, filter_type: Optional[str] = None) -> LitReviewResults:
    """
    Query the Crossref API for publication metadata, authors, or DOIs.
    
    This tool searches Crossref for information based on the query and filter type.
    It can look up specific DOIs, search for works by an author, or find articles
    matching a keyword query.
    
    Args:
        query (str): The search query or DOI
        filter_type (str, optional): Type of entity to look up:
            - 'doi': Look up specific DOI (e.g., "10.1038/s41586-020-2649-2")
            - 'author': Search for works by an author (e.g., "John Smith")
            - 'article': Search for articles matching query (default)
        
    Returns:
        Dict: Crossref lookup results with publication metadata
        
    Examples:
        # DOI lookup for a specific paper
        search_crossref("10.1038/s41586-020-2649-2", filter_type="doi")
        
        # Author lookup to find publications
        search_crossref("Geoffrey Hinton", filter_type="author")
        
        # Article search by keyword
        search_crossref("quantum computing", filter_type="article")
    """
    
    logger.info(f"Querying Crossref for: {query} (filter_type: {filter_type})")
    
    # Run the crossref lookup directly
    result = await _crossref_lookup(
        query=query,
        filter_type=filter_type
    )
    
    # Create a standardized results object
    if 'summary' in result:
        summary = result['summary']
    elif result.get('total_results'):
        summary = f"Found {result.get('total_results', 0)} items matching '{query}' in Crossref."
    else:
        summary = f"Retrieved information from Crossref for '{query}'."
    
    provenance = f"Data retrieved from Crossref on {datetime.now().isoformat()}"
    
    return LitReviewResults(
        summary=summary,
        provenance=provenance,
        data=result
    )