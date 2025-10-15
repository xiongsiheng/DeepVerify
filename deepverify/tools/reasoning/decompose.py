"""
    deepverify.tools.reasoning.decompose
"""

from typing import Dict
from pydantic import BaseModel

from deepverify.tools.promptrunner.promptrunner import PromptRunner

from deepverify.datamodels import ProofTree
from deepverify import config
from deepverify import prompts
from deepverify.cache import disk_cache

# --

decomposition_generator = PromptRunner(
    name            = "decomposition_generator",
    system          = prompts.decompose,
    template        = "CLAIM: {CLAIM}",
    response_format = ProofTree,
    before = lambda claim: {
        'CLAIM'            : claim,
        'FEASIBILITY_MODE' : 'feasibility', # Options: ['feasibility', 'infeasibility']
    },
    llm_kwargs = config.cheap_llm_kwargs,
    cache_dir  = './.cache/deepverify/tools/reasoning/decompose',
    no_console = True,
)


# [MCP tool]
async def decompose(claim: str, seed: int = 0) -> ProofTree:
    """
    Decompose a scientific claim into a logical proof tree.
    
    Args:
        claim (str): The scientific claim to decompose
        seed (int): Random seed for reproducibility. Defaults to 0.
        
    Returns:
        Dict: A dictionary containing the decomposition result with a proof tree structure
    """
    
    return await decomposition_generator.arun(claim=claim, _cache_idx=seed)
