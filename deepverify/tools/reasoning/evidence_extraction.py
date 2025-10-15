"""
    deepverify.tools.reasoning.evidence_extraction
"""

from deepverify.tools.promptrunner.promptrunner import PromptRunner

from deepverify import config
from deepverify import prompts
from deepverify.utils import json_loads_robust

evidence_extractor = PromptRunner(
    name       = 'evidence_extractor',
    template   = prompts.evidence_extractor,
    before     = lambda claim, subclaims, evidence: {
        'CLAIM'     : claim,
        'SUBCLAIMS' : ' - ' + '\n - '.join(sorted(set(subclaims))),
        'EVIDENCE'  : evidence,
    },
    after      = lambda output_str: json_loads_robust(output_str),
    llm_kwargs = config.cheap_llm_kwargs,
    cache_dir  = './.cache/promptrunner',
    no_console = True,
)

# [MCP tool]
async def evidence_extraction(claim: str, subclaims: list[str], evidence: str, seed: int = 0) -> list[str]:
    """
    Extract relevant evidence from a document.
    
    Args:
        claim (str): The main claim
        subclaims (List[str]): List of subclaims derived from the main claim
        evidence (str): Raw evidence document to extract from
        seed (int): Random seed for reproducibility. Defaults to 0.
        
    Returns:
        Dict: A dictionary containing extracted evidence relevant to the claim and subclaims
    """      
    out = await evidence_extractor.arun(
        claim       = claim, 
        subclaims   = subclaims, 
        evidence    = evidence,
        _cache_idx  = seed
    )
    
    return out['extractions']