#!/usr/bin/env python
"""
    deepverify.tools.lit_review.read_url
"""

import os
import sys
import httpx
import asyncio
from pydantic import BaseModel
from rich import print as rprint

from deepverify import config
from deepverify.cache import disk_cache

# --
# Output object

class ScrapeResult(BaseModel):
    title       : str
    description : str
    url         : str
    content     : str
    
    # def to_txt(self):
    #     # [TODO] fancier formatting?
    #     return f"<scrape_result>\n<title>{self.title}</title>\n<description>{self.description}</description>\n<url>{self.url}</url>\n<content>\n{self.content}\n</content>\n</scrape_result>"


# --
# Functions

# [MCP tool]
@disk_cache(cache_dir=config.CACHE_DIR / 'tools/lit_review/read_url', verbose=False)
async def read_url(url: str) -> ScrapeResult:
    """
    Fetch a URL and return the contents in Markdown format.  
    This can handle HTML, PDF, etc.
    
    Args:
        url (str): The URL to read
        
    Returns:
        ScrapeResult:
            title (str): title
            description (str): description
            url (str): url
            content (str): contents of the URL in Markdown format
    """
    
    _verbose = True
    
    API_KEY = os.environ.get("JINA_API_KEY")
    if not API_KEY:
        raise Exception("JINA_API_KEY is not set")
    
    url = f"https://r.jina.ai/{url}"

    headers = {
        "Accept"          : "application/json",
        "Authorization"   : f"Bearer {API_KEY}",
        "X-Return-Format" : "markdown",
        "X-Token-Budget"  : "1000000", #  very high limit
        # <<
        # EXPERIMENTAL
        # "X-Engine"        : "browser",
        "X-Md-Link-Style" : "discarded",
        "X-Retain-Images" : "none"
        # >>
    }
    
    try:
        async with httpx.AsyncClient(timeout=None) as client:
            if _verbose:
                rprint(f"[bright_black]read_url: fetching : {url}[/bright_black]", file=sys.stderr)
            
            res = await client.get(url, headers=headers)
            if _verbose:
                rprint(f"[bright_black]read_url: fetched  : {url}[/bright_black]", file=sys.stderr)
            
            if res.status_code != 200:
                rprint(f"[red]ERROR | scrape_jina: status_code != 200 - {res.status_code}[/red]", file=sys.stderr)
                raise Exception(f"ERROR | scrape_jina: status_code != 200 - {res.status_code}")
            
            data = res.json().get("data", None)
            if not data:
                rprint(f"[red]WARNING | scrape_jina: results is None[/red]", file=sys.stderr)
                raise Exception("ERROR | scrape_jina: results is None")
            
            return ScrapeResult(
                title       = data["title"],
                description = data["description"],
                url         = data["url"],
                content     = data["content"],
            )# .to_txt()
    
    except Exception as e:
        rprint(f"[red]ERROR | scrape_jina: {e}[/red]", file=sys.stderr)
        raise e

# --
# Test

if __name__ == "__main__":
    # out = asyncio.run(read_url("https://nyt.com"))
    out = asyncio.run(read_url("https://arxiv.org/pdf/2505.15742"))
    rprint(out)