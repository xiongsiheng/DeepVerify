import io
import asyncio
import sys
from pydantic import BaseModel
from pypdf import PdfReader
from rich import print as rprint
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError, Error

from deepverify import config
from deepverify.cache import disk_cache

class PdfReadResult(BaseModel):
    url: str
    content: str
    num_pages: int

@disk_cache(cache_dir=config.CACHE_DIR / 'tools/lit_review/read_pdf', verbose=False)
async def _read_pdf(url: str, _verbose: bool = True) -> PdfReadResult:
    """
    Internal implementation to fetch a PDF from a URL using a browser and return its text content.
    """
    if _verbose:
        rprint(f"[bright_black]read_pdf: launching browser for: {url}[/bright_black]", file=sys.stderr)

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        pdf_buffer = None
        
        try:
            # Primary method: handle direct download on navigation
            async with page.expect_download(timeout=30000) as download_info:
                try:
                    await page.goto(url, timeout=30000)
                except Error as e:
                    # Playwright throws this error when navigation is interrupted by a download.
                    # We can safely ignore it and proceed to wait for the download.
                    if "Download is starting" not in str(e):
                        raise
            
            download = await download_info.value
            # The Download object provides a path to the temporary file
            path = await download.path()
            if not path:
                raise Exception("Playwright download path was not found.")
            # Read the bytes from the temporary file path
            pdf_buffer = io.BytesIO(path.read_bytes())
            if _verbose:
                rprint(f"[bright_black]read_pdf: downloaded PDF to buffer[/bright_black]", file=sys.stderr)

        except PlaywrightTimeoutError:
            # Fallback method: if no download starts, the page might have an embedded viewer.
            # In this case, expect_download times out. We can then try to "print" the page to PDF.
            rprint("[yellow]WARNING | read_pdf: Download did not start. Trying to render page as PDF.[/yellow]", file=sys.stderr)
            await page.goto(url, timeout=60000) # Re-navigate just in case
            pdf_bytes = await page.pdf()
            pdf_buffer = io.BytesIO(pdf_bytes)
        
        except Exception as e:
            rprint(f"[red]ERROR | read_pdf: An unexpected error occurred - {e}[/red]", file=sys.stderr)
            await browser.close()
            raise e

        await browser.close()
        
        if not pdf_buffer:
             raise Exception("Could not retrieve PDF.")
        
        # Process the PDF from the buffer
        reader = PdfReader(pdf_buffer)
        text_content = ""
        for pdf_page in reader.pages:
            text_content += pdf_page.extract_text() or ""
        
        return PdfReadResult(
            url=url,
            content=text_content,
            num_pages=len(reader.pages)
        )

# [MCP tool]
async def read_pdf(url: str) -> PdfReadResult:
    """
    Fetch a PDF from a URL using a browser and return its text content.

    Args:
        url (str): The URL of the PDF to read.

    Returns:
        PdfReadResult:
            url (str): The URL of the PDF.
            content (str): The extracted text content of the PDF.
            num_pages (int): The number of pages in the PDF.
    """
    return await _read_pdf(url=url, _verbose=True)

# --
# Test

if __name__ == "__main__":
    test_url = "https://www.mdpi.com/2076-3417/13/15/8615/pdf?version=1690374320"
    
    if len(sys.argv) > 1:
        test_url = sys.argv[1]
        
    rprint(f"Testing read_pdf with URL: {test_url}")
    result = asyncio.run(read_pdf(test_url))
    rprint(result)
