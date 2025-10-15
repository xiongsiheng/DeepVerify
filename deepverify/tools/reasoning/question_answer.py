"""
    deepverify.tools.lit_review.question_answer
"""

import re
from pydantic import BaseModel

from deepverify.tools.promptrunner.promptrunner import PromptRunner

from deepverify import config
from deepverify import prompts
from deepverify.utils import json_loads_robust

# --
# Data Models

class Citation(BaseModel):
    citation_id: str
    id_raw: str
    evidence_text: str
    rationale: str

class QuestionAnswerResponse(BaseModel):
    claim: str
    citations: list[Citation]
    explanation: str
    feasibility_score: int
    technological_readiness: int


def fix_unclosed_cite_tags(text):
    """
    Fix instances where a new <cite> tag is opened before the previous one is closed.
    Specifically, it looks for patterns like:
        <cite ids=[...]> ... <cite ids=[...]>
    and replaces them with:
        <cite ids=[...]></cite> ... <cite ids=[...]>
    """
    # The lookahead checks that after the <cite> tag there is some text (that does not include </cite>)
    # and then another <cite> tag.
    pattern = r'(<cite ids=\[[\d\.,\s]+\]>)(?=(?:(?!</cite>).)*?(<cite ids=\[[\d\.,\s]+\]>|$))'
    return re.sub(pattern, r'\1[CITE]</cite>', text, flags=re.DOTALL)

# Question Answering
def _question_answerer_after(output_str, __claim, __idx):
    out = json_loads_robust(output_str)
    
    if out is None:
        out = None
    
    if not isinstance(out, dict):
        out = None
    
    if 'feasibility_score' not in out:
        out['feasibility_score'] = -999
    
    if 'technological_readiness' not in out:
        out['technological_readiness'] = -999
    
    # Cleaning common LLM formatting errors
    out['explanation'] = fix_unclosed_cite_tags(out['explanation'])
    out['explanation'] = re.sub(
        r'<cite ids=\[([\d,\s]+)\]>(.*?)</cite>',
        lambda m: f'<cite ids=[{",".join([f"{__idx}.{cid.strip()}" for cid in m.group(1).split(",")])}]>{m.group(2)}</cite>',
        out['explanation']
    )
    
    # Fixing citation format
    for cit in out['citations']:
        cit['citation_id'] = f'{__idx}.{cit["citation_id"]}'
    
    out['claim'] = __claim
    return out

_question_answerer = PromptRunner(
    name     = 'question_answerer',
    template = prompts.question_answerer,
    before   = lambda claim, evidence, idx: {
        'CLAIM'    : claim,
        'EVIDENCE' : '\n --- \n'.join(evidence),
        '__claim'  : claim,
        '__idx'    : idx
    },
    after      = _question_answerer_after,
    llm_kwargs = config.rich_llm_kwargs,
    cache_dir  = config.CACHE_DIR / 'promptrunner',
    no_console = True,
)

# [MCP tool]
async def question_answer(question: str, evidence: list[str], idx: str, seed: int = 0) -> QuestionAnswerResponse:
    """
    Answer a question based on provided evidence using an LLM.
    
    Args:
        question (str): The question to answer
        evidence (List[str]): List of evidence documents/papers
        idx (str): Unique identifier for the question/claim
        seed (int): Random seed for reproducibility. Defaults to 0.
        
    Returns:
        Dict: A dictionary containing the answer and supporting information
    """
    
    # [TODO] - what is the schema for `evidence`?
    
    out =await _question_answerer.arun(
        claim      = question, 
        evidence   = sorted(set(evidence)),
        idx        = idx, 
        _cache_idx = seed
    )
    return QuestionAnswerResponse(**out)
