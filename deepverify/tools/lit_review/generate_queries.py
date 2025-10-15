"""
    deepverify.tools.lit_review.generate_queries
"""

from deepverify.tools.promptrunner.promptrunner import PromptRunner

from deepverify import config
from deepverify import prompts
from deepverify.utils import json_loads_robust

# Query Generator
query_generator = PromptRunner(
    name       = 'query_generator',
    template   = prompts.query_generator,
    before     = lambda claim: {
        'CLAIM' : claim,
    },
    after      = lambda output_str: json_loads_robust(output_str),
    llm_kwargs = config.cheap_llm_kwargs,
    cache_dir  = './.cache/promptrunner',
    no_console = True,
)


# [MCP tool]
async def generate_queries(claim: str, seed: int = 0) -> list[str]:
    """
    Generate search queries from a claim.
    
    Args:
        claim (str): The claim to generate search queries for.
        seed (int): Random seed for reproducibility. Defaults to 0.
        
    Returns:
        list[str]: A list of generated search queries
    """
    result = await query_generator.arun(claim=claim, _cache_idx=seed)            
    return result["queries"]
