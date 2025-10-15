"""
    deepverify.tools.reasoning.mars
"""

import os
import uuid
import json
import asyncio
from pydantic import BaseModel

from deepverify import config
from deepverify.cache import disk_cache
from typing import Optional

from mars.mars.simple import run_swap

class MarsClaimResponse(BaseModel):
    claim_idx: int
    final_label: Optional[str]
    claim: str
    traj: dict
    graph: dict

class MarsResponse(BaseModel):
    claims: list[MarsClaimResponse]


# Get MARS model configuration
mars_model = 'gpt-5-mini'


@disk_cache(cache_dir=config.CACHE_DIR / 'tools/reasoning/mars', verbose=True, ignore_fields=['output_dir'])
def _run_swap_multi_claim(claims, statements, output_dir, *args, **kwargs):
    print("run_swap_multi_claim: start")
    # Add API_model parameter from config if not already provided
    if 'API_model' not in kwargs:
        kwargs['API_model'] = mars_model
    final_labels = [run_swap(claim=claim, statements=statements, output_dir=output_dir, *args, **kwargs) for claim in claims]
    print("run_swap_multi_claim: done")
    
    print("run_swap_multi_claim: post-processing")
    output = []
    for claim_idx, claim in enumerate(claims):
        claim_path = f'{output_dir}/{claim_idx}.json'
        if not os.path.exists(claim_path):
            continue
            
        mars        = json.loads(open(claim_path, 'r').read())
        final_graph = mars['final_graph']
        mars_best   = [xx for xx in mars['rollout'].values() if xx['active']]
        best_prompt = mars_best[0]['prompt']
        
        traj = {}
        for line in best_prompt.splitlines():
            key, *value = line.strip().split(':')
            key   = key.replace('"', '').strip()
            value = ':'.join(value).strip()
            traj[key] = value
        
        output.append({
            "claim_idx"   : claim_idx,
            "final_label" : final_labels[claim_idx] if claim_idx < len(final_labels) else None,
            "claim"       : claim,
            "traj"        : traj,
            "graph"       : final_graph
        })
    
    return {
        "claims" : output
    }


# [MCP tool]
async def mars(claims: list[str], statements: list[str]) -> MarsResponse:
    """
    Run MARS (Multi-Agent Reasoning System) to evaluate claims using structured reasoning.

    MARS is a multi-agent system designed to evaluate the validity of claims based on a provided set of supporting statements or facts.
    It incrementally constructs an entailment graph where:
    - Nodes represent statements.
    - Edges represent entailment relations (e.g., "Statement A" and "Statement B" together entail "Statement C").

    The reasoning process involves three specialized agents:
    - **Policy Agent**: Selects actions, such as choosing relevant statements or setting subgoals.
    - **World Model**: Simulates the outcome of the selected actions by updating the entailment graph with new statements and inferred relations.
    - **Discriminator Agent**: Evaluates the current state of the graph and provides feedback to guide the policy agent.

    This process iterates until the claim is either proven, refuted, or determined to be undecidable within a preset step limit.

    Args:
        claims (List[str]): List of claims to evaluate
        statements (List[str]): List of statements/facts to use in reasoning
        
    Returns:
        Dict: A dictionary containing the MARS reasoning results

    Examples:
        ### Input 1:
        "Problem": "sent1: Materials scientists have developed a high-temperature, single-phase Cu-3Ta-0.5Li (atomic percent) material that features complexion-stabilized precipitates and remains structurally robust up to 800°C. sent2: This novel alloy also exhibits a yield strength exceeding 1 gigapascal at room temperature. sent3: The breakthrough adds to a growing body of research aimed at designing next-generation structural materials capable of enduring extreme thermal and mechanical stress. sent4: Similar efforts include recent developments in refractory high-entropy alloys and nanostructured ceramics for aerospace and nuclear applications. sent5: Advances like this could lead to stronger, lighter components for jet engines, fusion reactors, and next-gen power grids, which are key infrastructure areas where performance under high heat environments is critical. sent6: There are no known single-phase materials in the Cu-Ta-Li system. sent7: A referenced paper shows a core-shelled system rather than single-phase materials. \n\nclaim: Scientist created a high-temperature novel single phase consisting of Cu-3Ta-0.5Li atomic percent alloy with complexion-stabilized precipitates that is structurally robust up to 800 C, with a yielding strength in excess of 1 gigapascal at room temperature"

        ### Output 1:
        "Goal": "Determine the claim 'Scientist created a high-temperature novel single phase consisting of Cu-3Ta-0.5Li atomic percent alloy with complexion-stabilized precipitates that is structurally robust up to 800 C, with a yielding strength in excess of 1 gigapascal at room temperature' is 'very likely', 'likely', 'neutral', 'unlikely', or 'very unlikely'."
        "Initial state": "We have 7 sentences: 'Materials scientists developed a high-temperature, single-phase Cu-3Ta-0.5Li with complexion-stabilized precipitates, robust up to 800°C', 'The alloy has yield strength >1 GPa at room temperature', 'This is part of research into next-gen structural materials', 'Similar efforts include refractory HEAs and nanostructured ceramics', 'Such advances could impact high-heat infrastructure', 'There are no known single-phase materials in Cu-Ta-Li', 'A referenced paper reports a core-shelled (not single-phase) system'."
        "Initial graph": {"Statement": {"s1": "Single-phase Cu-3Ta-0.5Li with complexion-stabilized precipitates is robust up to 800°C.", "s2": "The alloy's yield strength exceeds 1 GPa at room temperature.", "s3": "The work is part of broader research on extreme environments.", "s4": "Similar efforts include refractory HEAs and nanostructured ceramics.", "s5": "Potential applications include jet engines, fusion reactors, and power grids.", "s6": "No known single-phase materials exist in the Cu-Ta-Li system.", "s7": "A paper shows a core-shelled system rather than a single phase."}, "Entailment": {"s1": "Evidence", "s2": "Evidence", "s3": "Evidence", "s4": "Evidence", "s5": "Evidence", "s6": "Evidence", "s7": "Evidence"}}
        "Action 1": "Combine s1 and s2 to align with the technical content of the claim."
        "State 1": "From s1 and s2, we infer that the article asserts a single-phase Cu-3Ta-0.5Li with complexion-stabilized precipitates is robust to 800°C and has >1 GPa yield at room temperature."
        "Graph 1": {"Statement": {"s8": "Article-asserted properties match the claim (single phase Cu-3Ta-0.5Li, robust to 800°C, >1 GPa yield)."}, "Entailment": {"s8": ["s1", "s2"]}}
        "Action 2": "Synthesize s6 and s7 to evaluate whether the 'single-phase Cu-3Ta-0.5Li' assertion is contradicted by external evidence."
        "State 2": "s6 and s7 jointly indicate the system is not known to form a single phase; reported structures are core-shelled, contradicting the claim's single-phase premise."
        "Graph 2": {"Statement": {"s9": "External evidence contradicts the single-phase aspect of the claim."}, "Entailment": {"s9": ["s6", "s7"]}}
        "Action 3": "Compare the article-asserted match (s8) with the contradiction (s9)."
        "State 3": "Because the single-phase requirement is central to the claim and is contradicted by external evidence, the claim is overall not credible."
        "Graph 3": {"Statement": {"s10": "Given conflicting evidence, the claim is very unlikely to be true."}, "Entailment": {"s10": ["s8", "s9"]}}
        "Action 4": "Compare s10 with the claim to produce the final rating."
        "State 4": "s10 directly assesses the claim's plausibility as very unlikely."
        "Final answer": "very unlikely"


        ### Input 2:
        "Problem": "sent1: scientists report a fiber battery made from an anode wire and a cathode wire twisted together in a parallel configuration sent2: in geometry, parallel means two lines (or curves) that remain everywhere equidistant and never intersect sent3: twisted together means the two wires are helically wound around each other (a twisted pair)\n\nclaim: a fiber battery can be made with an anode wire and a cathode wire in a parallel configuration with the anode and cathode wires twisted together."

        ### Output 2:
        "Goal": "Determine the claim 'A fiber battery can be made with an anode wire and a cathode wire in a parallel configuration with the anode and cathode wires twisted together' is 'very likely', 'likely', 'neutral', 'unlikely', or 'very unlikely'."
        "Initial state": "We have 3 sentences: (1) a reported design uses an anode wire and a cathode wire twisted together in a parallel configuration, (2) 'parallel' means everywhere equidistant and non-intersecting, (3) 'twisted together' means helically wound."
        "Initial graph": {"Statement": {"s1": "A reported design uses an anode wire and a cathode wire twisted together in a parallel configuration.", "s2": "In geometry, 'parallel' means everywhere equidistant and never intersecting.", "s3": "'Twisted together' means the wires are helically wound."}, "Entailment": {"s1": "Evidence", "s2": "Evidence", "s3": "Evidence"}}
        "Action 1": "Combine s2 and s3 to assess geometric compatibility."
        "State 1": "From s2 and s3, helically wound (twisted) wires cannot be geometrically parallel."
        "Graph 1": {"Statement": {"s4": "Helically wound wires are not geometrically parallel."}, "Entailment": {"s4": ["s2", "s3"]}}
        "Action 2": "Compare the reported description (s1) with the incompatibility (s4)."
        "State 2": "Since s1 requires the wires to be both twisted and parallel, and s4 says that is impossible, mark the description as self-contradictory."
        "Graph 2": {"Statement": {"s5": "The description is self-contradictory."}, "Entailment": {"s5": ["s1", "s4"]}}
        "Action 3": "Map the contradiction (s5) to a likelihood judgment of the claim."
        "State 3": "Claims containing an internal contradiction are very unlikely."
        "Final answer": "very unlikely"
    """
    
    output_dir = f"./tmp/mars/{uuid.uuid4()}"
    os.makedirs(output_dir, exist_ok=True)
    
    loop = asyncio.get_running_loop()
    out  = await loop.run_in_executor(
        None, 
        lambda: _run_swap_multi_claim(
            claims       = claims, 
            statements   = sorted(set(statements)), 
            output_dir   = output_dir,
            fast_mode    = True,
            num_rollouts = 8,
            max_steps    = 1,
            API_model    = mars_model,  # Use model from config
            use_thinking_model = True,
            # callback=lambda x: print(x)
        )
    )

    return MarsResponse(**out)