"""
    deepverify.mcp.server
    
    Re-implementation of the `jataware-zero` endpoints.  
    Not 100% the same:
    
    (+ have been implemented, - are going to be omitted, TODO we need to do)
        + decompose
        + generate_queries
        + search
        + question_answer
        + evidence_extraction
        + evidence_filter
        + mars
        + gxbi
        + google_scholar_search
        + check_journal_rank
        + crossref_lookup

        [TODO] run (run_python) - run python script
        [TODO] *_domain_tools
        
        - write_note    - not generally used - left in `archytas` agent
        - read_note     - not generally used - left in `archytas` agent
        - pdf_ocr       - replaced w/ `read_url` (JINA reader)
        - check_h_index - dropped for now - complex and not used by agent
        - all background job stuff - async is hard
"""

import uvicorn
from fastapi import FastAPI
from fastmcp import FastMCP

# <<
# [TODO] Is this right?
import fastmcp
fastmcp.settings.stateless_http = True
# >>

from deepverify import config

from deepverify.tools.lit_review.generate_queries import generate_queries
from deepverify.tools.lit_review.get_journal_rank import get_journal_rank

from deepverify.tools.lit_review.search_crossref import search_crossref
from deepverify.tools.lit_review.search_google_scholar import search_google_scholar
from deepverify.tools.lit_review.read_url import read_url
from deepverify.tools.lit_review.search_google import search_google
from deepverify.tools.lit_review.read_pdf import read_pdf

from deepverify.tools.reasoning.decompose import decompose
from deepverify.tools.reasoning.question_answer import question_answer
from deepverify.tools.reasoning.evidence_extraction import evidence_extraction
from deepverify.tools.reasoning.evidence_filter import evidence_filter
from deepverify.tools.reasoning.mars import mars
from deepverify.tools.reasoning.run_python import run_python

mcp = FastMCP("deepverify")

# --
# Register tools

# [TODO] nicer naming / better organization for these?

mcp.tool(decompose,              name="decompose")
mcp.tool(evidence_extraction,    name="evidence_extraction")
mcp.tool(question_answer,        name="question_answer")
mcp.tool(evidence_filter,        name="evidence_filter")
mcp.tool(mars,                   name="mars")
mcp.tool(run_python,             name="run_python") # Insecure

mcp.tool(read_url,               name="read_url")
mcp.tool(generate_queries,       name="generate_queries")
mcp.tool(get_journal_rank,       name="get_journal_rank")
mcp.tool(search_crossref,        name="search_crossref")
mcp.tool(search_google_scholar,  name="search_google_scholar")
mcp.tool(search_google,          name="search_google")
mcp.tool(read_pdf,               name="read_pdf")

mcp_http_app = mcp.http_app()
app = FastAPI(lifespan=mcp_http_app.lifespan)
app.mount("/", mcp_http_app)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=config.MCP_PORT)
    
    # [TODO] - can wrap in a REST API as well using code in scratch/rest_wrapper.py, if need be
    # app = asyncio.run(create_endpoints(app))
    # mcp.run(transport="http", host="0.0.0.0", port=8890)
