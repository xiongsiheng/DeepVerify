"""
    deepverify.tools.reasoning.evidence_filter
"""

from deepverify.tools.promptrunner.promptrunner import PromptRunner
from deepverify import config
from deepverify import prompts
from deepverify.utils import json_loads_robust

# Evidence Extraction
_evidence_filter = PromptRunner(
    name       = 'evidence_filter',
    template   = prompts.evidence_filter,
    before     = lambda claim, subclaims, evidence, topk: {
        'CLAIM'     : claim,
        'SUBCLAIMS' : ' - ' + '\n - '.join(subclaims),
        'EVIDENCE'  : ' - ' + '\n - '.join(evidence), # should add IDS here
        'TOPK'      : topk
    },
    after      = lambda output_str: json_loads_robust(output_str),
    llm_kwargs = config.rich_llm_kwargs,
    cache_dir  = './.cache/promptrunner',
    no_console = True,
)

# [MCP tool]
async def evidence_filter(claim: str, subclaims: list[str], evidence: list[str], topk: int = 64, seed: int = 0) -> list[str]:
    """
    Filter and rank evidence based on relevance.
    
    Args:
        claim (str): The main claim
        subclaims (List[str]): List of subclaims derived from the main claim
        evidence (List[str]): List of evidence documents to filter
        topk (int): Number of top evidence documents to return. Defaults to 64.
        seed (int): Random seed for reproducibility. Defaults to 0.
        
    Returns:
        Dict: A dictionary containing filtered and ranked evidence
    """
    out = await _evidence_filter.arun(
        claim       = claim, 
        subclaims   = subclaims, 
        evidence    = evidence,
        topk        = topk,
        _cache_idx  = seed
    )
    
    return out['evidence']
