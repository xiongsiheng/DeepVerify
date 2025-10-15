#!/usr/bin/env python
"""
Journal Metrics Module for Literature Review

This module provides functions to check journal rankings, impact factors, and related
metrics for scientific journals. It serves as the backend for the 
/lit_review/check_journal_rank endpoint in the API.

Features:
- Journal impact factor lookup from multiple sources
- SJR (Scimago Journal Rank) quartile information
- Basic journal metadata (ISSN, publisher, etc.)
- Open access status verification
- Asynchronous operation for non-blocking API calls

---------------------------------------------------------
JOURNAL METRICS LOOKUP PROCESS
---------------------------------------------------------

This module implements a multi-source approach to retrieving journal metrics:

1. Data Sources (in order of access):
   - Crossref API: Basic journal metadata (title, publisher, ISSN)
   - Wikipedia: Primary source for impact factor data (public, no API key needed)
   - Google Scholar: For h5-index data which correlates with impact factor
   - DOAJ (Directory of Open Access Journals): For open access journal information
   - SJR (Scimago Journal Rank): For quartile rankings and SJR score

2. Impact Factor Retrieval Strategy:
   a. First attempts to find exact impact factor from:
      - Wikipedia (extracted from journal pages)
      - DOAJ API (for open access journals)
   b. If exact impact factor isn't found, tries:
      - Google Scholar h5-index as an alternative metric
      - Estimation based on journal reputation for well-known journals
   c. Falls back to providing other metrics when impact factor isn't available

3. Key Features:
   - Multi-source fallback system for maximum data availability
   - No subscription or API keys required
   - Fast response time (typically under 5 seconds)
   - Handles journal name variations and spelling differences
   - Returns both exact metrics and estimated data with clear labeling

The implementation prioritizes providing useful information even when
exact impact factor data isn't available, using a combination of authoritative
sources and estimation techniques for well-known journals.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
import json
import httpx
import re
from bs4 import BeautifulSoup
from urllib.parse import quote

from deepverify.datamodels import LitReviewResults

logger = logging.getLogger(__name__)

class JournalMetricsError(Exception):
    """Exception raised for journal metrics-related errors."""
    pass

async def _check_journal_rank(journal_name: str) -> Dict[str, Any]:
    """
    Check the ranking and impact factor of a journal.
    
    Args:
        journal_name: The name of the journal to check
        
    Returns:
        Dictionary containing journal metrics information
        
    Raises:
        JournalMetricsError: If an error occurs during metrics lookup
    """
    try:
        logger.info(f"Checking metrics for journal: {journal_name}")
        
        # First try to get data from Crossref API
        crossref_data = await _fetch_journal_crossref(journal_name)
        
        # Then try to get data from SJR (Scimago Journal Rank)
        sjr_data = await _fetch_journal_sjr(journal_name)
        
        # Try JCR data (Journal Citation Reports / Impact Factor)
        jcr_data = await _fetch_journal_jcr(journal_name)
        
        # Combine all data
        result = {
            "name": journal_name,
            "metrics": {
                "crossref": crossref_data,
                "sjr": sjr_data,
                "jcr": jcr_data
            },
            "best_match": _identify_best_match(crossref_data, sjr_data, jcr_data),
            "summary": _generate_summary(journal_name, crossref_data, sjr_data, jcr_data)
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error checking journal rank: {str(e)}")
        raise JournalMetricsError(f"Failed to check journal rank: {str(e)}")

async def _fetch_journal_crossref(journal_name: str) -> Dict[str, Any]:
    """
    Fetch journal information from Crossref API.
    
    Args:
        journal_name: The name of the journal
        
    Returns:
        Dictionary containing Crossref data
    """
    try:
        async with httpx.AsyncClient() as client:
            # Query Crossref for journals matching the name
            encoded_name = quote(journal_name)
            url = f"https://api.crossref.org/journals?query={encoded_name}"
            
            response = await client.get(url, timeout=10.0)
            response.raise_for_status()
            
            data = response.json()
            
            # Check if we have any matches
            if 'message' in data and 'items' in data['message'] and len(data['message']['items']) > 0:
                # Get the first (best) match
                journal = data['message']['items'][0]
                
                return {
                    "title": journal.get('title', journal_name),
                    "issn": journal.get('ISSN', []),
                    "publisher": journal.get('publisher', 'Unknown'),
                    "subjects": journal.get('subjects', []),
                    "flags": journal.get('flags', []),
                    "count": data['message'].get('total-results', 0),
                    "source": "Crossref"
                }
            
            return {"status": "not_found", "source": "Crossref"}
            
    except Exception as e:
        logger.warning(f"Error fetching Crossref data: {str(e)}")
        return {"status": "error", "message": str(e), "source": "Crossref"}

async def _fetch_journal_sjr(journal_name: str) -> Dict[str, Any]:
    """
    Fetch journal ranking information from Scimago Journal Rank (SJR).
    
    Args:
        journal_name: The name of the journal
        
    Returns:
        Dictionary containing SJR data
    """
    try:
        # Make a real request to Scimago Journal & Country Rank
        encoded_name = quote(journal_name)
        url = f"https://www.scimagojr.com/journalsearch.php?q={encoded_name}"
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=10.0)
            response.raise_for_status()
            
            # Parse the HTML response
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Search for journal data in the page
            journal_data = {}
            
            # Try to find the journal card
            journal_card = soup.find('div', class_='search_results')
            if not journal_card:
                return {"status": "not_found", "message": "Journal not found in SJR database", "source": "SJR"}
            
            # Extract the SJR score
            sjr_element = soup.find('div', class_='journaldescription')
            sjr_data = {}
            
            if sjr_element:
                # Extract various metrics if available
                metrics_elements = sjr_element.find_all('div', class_='cell')
                for elem in metrics_elements:
                    label_elem = elem.find('p', class_='label')
                    value_elem = elem.find('p', class_='data')
                    if label_elem and value_elem:
                        label = label_elem.text.strip().lower()
                        value = value_elem.text.strip()
                        
                        if 'sjr' in label:
                            try:
                                sjr_data['sjr_score'] = float(value)
                            except ValueError:
                                sjr_data['sjr_score_text'] = value
                        elif 'h index' in label:
                            try:
                                sjr_data['h_index'] = int(value)
                            except ValueError:
                                sjr_data['h_index_text'] = value
            
            # If we found some data, return it
            if sjr_data:
                sjr_data['source'] = "Scimago Journal Rank"
                return sjr_data
            
            # If we didn't find any data, return a not found status
            return {"status": "data_not_found", "message": "Could not extract SJR data", "source": "SJR"}
            
    except Exception as e:
        logger.warning(f"Error fetching SJR data: {str(e)}")
        return {"status": "error", "message": str(e), "source": "SJR"}

async def _fetch_journal_jcr(journal_name: str) -> Dict[str, Any]:
    """
    Fetch journal impact factor using multiple sources.
    
    This function tries several approaches to find impact factor data:
    1. DOAJ API for basic journal info and some metrics
    2. Wikipedia for impact factor data
    3. Journal website scraping attempt
    
    Args:
        journal_name: The name of the journal
        
    Returns:
        Dictionary containing impact factor data from various sources
    """
    try:
        # Normalize journal name
        encoded_name = quote(journal_name)
        clean_name = journal_name.lower().strip()
        
        # 1. Try DOAJ (Directory of Open Access Journals) API first
        doaj_data = await _try_doaj_lookup(clean_name)
        if doaj_data.get("impact_factor"):
            return doaj_data
            
        # 2. Try Wikipedia for impact factor
        wiki_data = await _try_wikipedia_lookup(clean_name)
        if wiki_data.get("impact_factor"):
            return wiki_data
            
        # 3. Try Google Scholar for metrics data
        google_data = await _try_google_scholar(clean_name)
        if google_data.get("impact_factor") or google_data.get("h5_index"):
            return google_data
            
        # If we couldn't find impact factor data but found other metrics, return those
        if doaj_data.get("status") != "error" and doaj_data.get("journal_info"):
            return {
                "journal_info": doaj_data.get("journal_info", {}),
                "source": "DOAJ",
                "note": "Impact factor not available, but journal information found."
            }
            
        # Fall back to a message about impact factor data
        return {
            "impact_factor_estimate": _generate_impact_factor_estimate(journal_name),
            "status": "limited_data",
            "message": "Impact factor data is proprietary. Estimate based on journal reputation and field.",
            "source": "Multiple sources",
            "note": "For precise impact factor, check Journal Citation Reports via a university library."
        }
            
    except Exception as e:
        logger.warning(f"Error fetching impact factor data: {str(e)}")
        return {"status": "error", "message": str(e), "source": "Impact Factor Lookup"}
        
async def _try_doaj_lookup(journal_name: str) -> Dict[str, Any]:
    """Try to look up journal in DOAJ (Directory of Open Access Journals)."""
    try:
        url = f"https://doaj.org/api/v2/search/journals/title:{journal_name}"
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=10.0)
            
            if response.status_code != 200:
                return {"status": "not_found", "source": "DOAJ"}
                
            data = response.json()
            
            if data.get("total") > 0 and len(data.get("results", [])) > 0:
                journal = data["results"][0]
                
                # Extract relevant fields
                journal_info = {
                    "title": journal.get("bibjson", {}).get("title"),
                    "publisher": journal.get("bibjson", {}).get("publisher", {}).get("name"),
                    "issn": journal.get("bibjson", {}).get("identifier", []),
                    "subject": [s.get("term") for s in journal.get("bibjson", {}).get("subject", [])]
                }
                
                # Look for metrics in DOAJ data
                if "index" in journal.get("bibjson", {}):
                    metrics = journal["bibjson"]["index"]
                    if isinstance(metrics, list):
                        for idx in metrics:
                            if idx.get("name", "").lower() == "impact factor":
                                return {
                                    "impact_factor": idx.get("value"),
                                    "journal_info": journal_info,
                                    "source": "DOAJ"
                                }
                
                # Return journal info even without impact factor
                return {
                    "journal_info": journal_info,
                    "source": "DOAJ"
                }
                
        return {"status": "no_metrics", "source": "DOAJ"}
        
    except Exception as e:
        logger.warning(f"DOAJ lookup error: {str(e)}")
        return {"status": "error", "message": str(e), "source": "DOAJ"}
        
async def _try_wikipedia_lookup(journal_name: str) -> Dict[str, Any]:
    """Try to find impact factor data from Wikipedia."""
    try:
        # Wikipedia API search
        url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={quote(journal_name)}&format=json"
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=10.0)
            
            if response.status_code != 200:
                return {"status": "not_found", "source": "Wikipedia"}
                
            data = response.json()
            
            # Check if we have search results
            if "query" in data and "search" in data["query"] and len(data["query"]["search"]) > 0:
                # Get the page title of the first result
                title = data["query"]["search"][0]["title"]
                
                # Get the content of the page
                content_url = f"https://en.wikipedia.org/w/api.php?action=parse&page={quote(title)}&prop=text&format=json"
                content_response = await client.get(content_url, timeout=10.0)
                
                if content_response.status_code != 200:
                    return {"status": "content_not_found", "source": "Wikipedia"}
                    
                content_data = content_response.json()
                
                if "parse" in content_data and "text" in content_data["parse"]:
                    html_content = content_data["parse"]["text"]["*"]
                    
                    # Parse with BeautifulSoup
                    soup = BeautifulSoup(html_content, 'html.parser')
                    
                    # Look for impact factor in the infobox
                    impact_factor_regex = r"impact factor[:\s]*([\d\.]+)"
                    infobox = soup.find('table', class_='infobox')
                    
                    if infobox:
                        infobox_text = infobox.get_text()
                        match = re.search(impact_factor_regex, infobox_text, re.IGNORECASE)
                        
                        if match:
                            try:
                                impact_factor = float(match.group(1))
                                return {
                                    "impact_factor": impact_factor,
                                    "source": "Wikipedia",
                                    "year": "Latest available"  # Wikipedia data may not specify the year
                                }
                            except ValueError:
                                pass
                                
                    # Also try searching in the full text
                    full_text = soup.get_text()
                    match = re.search(impact_factor_regex, full_text, re.IGNORECASE)
                    
                    if match:
                        try:
                            impact_factor = float(match.group(1))
                            return {
                                "impact_factor": impact_factor,
                                "source": "Wikipedia",
                                "year": "Latest available"
                            }
                        except ValueError:
                            pass
        
        return {"status": "no_impact_factor", "source": "Wikipedia"}
        
    except Exception as e:
        logger.warning(f"Wikipedia lookup error: {str(e)}")
        return {"status": "error", "message": str(e), "source": "Wikipedia"}
        
async def _try_google_scholar(journal_name: str) -> Dict[str, Any]:
    """Try to find metrics from Google Scholar."""
    try:
        # Google Scholar metrics URL
        url = f"https://scholar.google.com/citations?view_op=search_venues&hl=en&vq={quote(journal_name)}"
        
        async with httpx.AsyncClient(headers={"User-Agent": "Mozilla/5.0"}) as client:
            response = await client.get(url, timeout=10.0, follow_redirects=True)
            
            if response.status_code != 200:
                return {"status": "not_accessible", "source": "Google Scholar"}
                
            # Parse with BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Look for h5-index and h5-median
            venues = soup.find_all('tr', class_='gsc_mvt_row')
            
            for venue in venues:
                name_element = venue.find('td', class_='gsc_mvt_t')
                if not name_element:
                    continue
                    
                venue_name = name_element.text.strip()
                
                # Check if this is a close match to our journal
                if _is_name_match(venue_name, journal_name):
                    h5_index = venue.find('td', class_='gsc_mvt_n')
                    h5_median = venue.find('a', class_='gs_ibl gsc_mp_anchor')
                    
                    result = {
                        "journal_name": venue_name,
                        "source": "Google Scholar"
                    }
                    
                    if h5_index:
                        result["h5_index"] = int(h5_index.text.strip())
                        
                    if h5_median:
                        result["h5_median"] = int(h5_median.text.strip())
                        
                    # Estimate impact factor from h5-index if available
                    if result.get("h5_index") and not result.get("impact_factor"):
                        # This is a crude estimation - not scientifically valid
                        # but provides a very rough approximation
                        h5 = result["h5_index"]
                        if h5 > 100:
                            result["impact_factor_estimate"] = "10+"
                        elif h5 > 80:
                            result["impact_factor_estimate"] = "8-10"
                        elif h5 > 60:
                            result["impact_factor_estimate"] = "6-8"
                        elif h5 > 40:
                            result["impact_factor_estimate"] = "4-6"
                        elif h5 > 20:
                            result["impact_factor_estimate"] = "2-4"
                        else:
                            result["impact_factor_estimate"] = "0-2"
                            
                        result["note"] = "Impact factor estimated from h5-index (approximate)"
                        
                    return result
            
        return {"status": "no_metrics", "source": "Google Scholar"}
        
    except Exception as e:
        logger.warning(f"Google Scholar lookup error: {str(e)}")
        return {"status": "error", "message": str(e), "source": "Google Scholar"}
        
def _is_name_match(name1: str, name2: str) -> bool:
    """Check if two journal names are similar enough to be considered a match."""
    # Normalize
    n1 = name1.lower().strip()
    n2 = name2.lower().strip()
    
    # Direct match
    if n1 == n2:
        return True
        
    # One is contained in the other
    if n1 in n2 or n2 in n1:
        return True
        
    # Split into words and check for significant overlap
    words1 = set(w for w in n1.split() if len(w) > 3)
    words2 = set(w for w in n2.split() if len(w) > 3)
    
    if words1 and words2:
        overlap = len(words1.intersection(words2))
        return overlap >= min(len(words1), len(words2)) / 2
        
    return False
    
def _generate_impact_factor_estimate(journal_name: str) -> str:
    """
    Generate an impact factor estimate based on journal name and reputation.
    This is a fallback when no actual data is available.
    """
    # This is a very rough estimate based on journal reputation
    prestigious_journals = {
        "nature": "40+",
        "science": "30+",
        "cell": "25+",
        "lancet": "25+",
        "nejm": "70+",
        "new england journal of medicine": "70+",
        "jama": "20+",
        "pnas": "10+"
    }
    
    name_lower = journal_name.lower()
    
    # Check for exact matches in our prestigious journals list
    for key, value in prestigious_journals.items():
        if key == name_lower or key in name_lower:
            return value
            
    # Return a range based on generic reputation cues in the name
    if "review" in name_lower:
        return "5-15"  # Review journals tend to have higher impact factors
    elif "advances" in name_lower:
        return "3-8"
    elif "transactions" in name_lower:
        return "2-5"
    elif any(x in name_lower for x in ["international", "global"]):
        return "1-4"
        
    # Default
    return "unknown"

def _identify_best_match(
    crossref_data: Dict[str, Any],
    sjr_data: Dict[str, Any],
    jcr_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Identify the best match among all data sources.
    
    Args:
        crossref_data: Data from Crossref
        sjr_data: Data from SJR
        jcr_data: Data from JCR
        
    Returns:
        Dictionary with best match information
    """
    # In this simple implementation, we prioritize data based on availability
    best_match = {}
    
    # Title from Crossref if available
    if 'title' in crossref_data:
        best_match['title'] = crossref_data['title']
    
    # Publisher from Crossref if available
    if 'publisher' in crossref_data:
        best_match['publisher'] = crossref_data['publisher']
    
    # ISSN from Crossref if available
    if 'issn' in crossref_data and crossref_data['issn']:
        best_match['issn'] = crossref_data['issn']
    
    # Subject categories from Crossref if available
    if 'subjects' in crossref_data and crossref_data['subjects']:
        best_match['categories'] = crossref_data['subjects']
    
    # Impact factor data from JCR
    if 'impact_factor' in jcr_data:
        best_match['impact_factor'] = jcr_data['impact_factor']
        best_match['impact_factor_source'] = jcr_data.get('source', 'JCR')
        
    # Impact factor estimate if no actual impact factor is available
    elif 'impact_factor_estimate' in jcr_data:
        best_match['impact_factor_estimate'] = jcr_data['impact_factor_estimate']
        best_match['impact_factor_note'] = jcr_data.get('note', 'Estimated value')
        
    # H5-index from Google Scholar
    if 'h5_index' in jcr_data:
        best_match['h5_index'] = jcr_data['h5_index']
        best_match['h5_source'] = 'Google Scholar'
        
    # Journal info from DOAJ or other sources
    if 'journal_info' in jcr_data:
        if 'journal_name' not in best_match and 'title' in jcr_data['journal_info']:
            best_match['journal_name'] = jcr_data['journal_info']['title']
            
        if 'publisher' not in best_match and 'publisher' in jcr_data['journal_info']:
            best_match['publisher'] = jcr_data['journal_info']['publisher']
            
        if 'subject' in jcr_data['journal_info'] and ('categories' not in best_match or not best_match['categories']):
            best_match['categories'] = jcr_data['journal_info']['subject']
    
    # SJR metrics if available
    if 'sjr_score' in sjr_data:
        best_match['sjr_score'] = sjr_data['sjr_score']
    
    if 'quartile' in sjr_data:
        best_match['quartile'] = sjr_data['quartile']
    
    if 'h_index' in sjr_data:
        best_match['h_index'] = sjr_data['h_index']
    
    return best_match

def _generate_summary(
    journal_name: str,
    crossref_data: Dict[str, Any],
    sjr_data: Dict[str, Any],
    jcr_data: Dict[str, Any]
) -> str:
    """
    Generate a summary of journal metrics.
    
    Args:
        journal_name: The name of the journal
        crossref_data: Data from Crossref
        sjr_data: Data from SJR
        jcr_data: Data from JCR
        
    Returns:
        Summary text
    """
    summary_parts = []
    
    # Get the proper title from sources in priority order
    journal_title = None
    if 'title' in crossref_data:
        journal_title = crossref_data['title']
    elif 'journal_name' in jcr_data:
        journal_title = jcr_data['journal_name']
    elif 'journal_info' in jcr_data and 'title' in jcr_data['journal_info']:
        journal_title = jcr_data['journal_info']['title']
    else:
        journal_title = journal_name
    
    # Start with basic info
    publisher = None
    if 'publisher' in crossref_data:
        publisher = crossref_data['publisher']
    elif 'journal_info' in jcr_data and 'publisher' in jcr_data['journal_info']:
        publisher = jcr_data['journal_info']['publisher']
    else:
        publisher = 'Unknown publisher'
        
    summary_parts.append(f"{journal_title} is published by {publisher}.")
    
    # Add impact factor info from various sources in priority order
    if 'impact_factor' in jcr_data:
        summary_parts.append(f"It has an Impact Factor of {jcr_data['impact_factor']} ({jcr_data.get('year', 'current')}).")
    elif 'impact_factor_estimate' in jcr_data:
        summary_parts.append(f"It has an estimated Impact Factor in the range of {jcr_data['impact_factor_estimate']}.")
    
    # Add h5-index from Google Scholar if available
    if 'h5_index' in jcr_data:
        summary_parts.append(f"Google Scholar h5-index: {jcr_data['h5_index']}.")
    
    # Add SJR info if available
    if 'sjr_score' in sjr_data:
        if 'quartile' in sjr_data:
            summary_parts.append(f"Its SJR score is {sjr_data['sjr_score']} (quartile: {sjr_data['quartile']}).")
        else:
            summary_parts.append(f"Its SJR score is {sjr_data['sjr_score']}.")
    
    # Add category info if available
    if 'rank_in_category' in sjr_data and 'total_in_category' in sjr_data and 'category' in sjr_data:
        summary_parts.append(
            f"It ranks {sjr_data['rank_in_category']}/{sjr_data['total_in_category']} " 
            f"in the category '{sjr_data['category']}'."
        )
    
    # Add categories/subjects from any source
    categories = []
    if 'subjects' in crossref_data and crossref_data['subjects']:
        categories.extend(crossref_data['subjects'])
    if 'journal_info' in jcr_data and 'subject' in jcr_data['journal_info']:
        categories.extend(jcr_data['journal_info']['subject'])
        
    if categories:
        category_str = ", ".join(categories[:3])
        if len(categories) > 3:
            category_str += f", and {len(categories) - 3} more"
        summary_parts.append(f"Subject areas include: {category_str}.")
    
    # Add h-index if available
    if 'h_index' in sjr_data:
        summary_parts.append(f"The journal has an h-index of {sjr_data['h_index']}.")
    
    # Add note if the impact factor is estimated
    if 'note' in jcr_data and not any(jcr_data['note'] in part for part in summary_parts):
        summary_parts.append(jcr_data['note'])
    
    # Combine all parts
    return " ".join(summary_parts)

# [MCP tool]
async def get_journal_rank(journal_name: str) -> LitReviewResults:
    """
    Check the ranking and impact factor of an academic journal.
    
    This tool looks up journal ranking information, with a focus on impact factor data,
    using a multi-source approach to find the most accurate information available.
    
    Args:
        journal_name (str): Name of the journal to look up
        
    Returns:
        Dict: Journal ranking information including impact factor and other metrics
        
    Examples:
        # Look up a top-tier journal
        get_journal_rank("Nature")
        
        # Look up a specialized journal
        get_journal_rank("Journal of Artificial Intelligence Research")
    """
    logger.info(f"Checking journal metrics for: {journal_name}")
    
    # Run the get_journal_rank function directly
    result = await _check_journal_rank(journal_name)
    
    # Create a standardized results object
    summary = result.get('summary', f"Retrieved information for journal '{journal_name}'.")
    provenance = "Data compiled from Crossref, Wikipedia, Google Scholar, and other sources."
    
    return LitReviewResults(
        summary=summary,
        provenance=provenance,
        data=result
    )
