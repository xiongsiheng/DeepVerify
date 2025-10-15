#!/usr/bin/env python3


import json
import asyncio
import argparse
import datetime
import traceback
import numpy as np
from pathlib import Path
from rich import print as rprint
from rich.panel import Panel
from rich.console import Console
from tenacity import retry, stop_after_attempt

from scify_formats import Problem
from .jdr import JDRAgent
from .jdr_oai import JDROAIAgent

@retry(stop=stop_after_attempt(3))
async def _run_one(sem, model_name, tool_whitelist, all_tools, no_tools, claim, output_dir):
    if 'openai/' in model_name:
        print(f'using JDROAIAgent for {model_name}')
        agent_cls  = JDROAIAgent
        model_name = model_name.replace('openai/', '')
    else:
        print(f'using JDRAgent for {model_name}')
        agent_cls = JDRAgent
        
    async with sem:
        try:
            rprint(f'[yellow]launching - {claim.problem_id}[/yellow]')
            
            (output_dir / claim.problem_id).mkdir(parents=True, exist_ok=True)
            trace_path = output_dir / claim.problem_id / "trace.jl"

            agent = agent_cls(
                model_name     = model_name, 
                tool_whitelist = tool_whitelist,
                all_tools      = all_tools,
                no_tools       = no_tools,
            )
        
            trace = await agent.arun(claim.claim, verbose=False, max_iters=32)
            rprint(f'[green]success   - {claim.problem_id}[/green]')
            trace_path.write_text('\n'.join([json.dumps(message) for message in trace]))
            
            return trace
            
        except Exception as e:
            error_trace = traceback.format_exc()
            console = Console()
            console.print(Panel(f"failure - {claim.problem_id}\n--\n" + error_trace, title="Error Trace", border_style="red"))
            raise e


async def main(args):
    all_claim_data = [json.loads(line.strip()) for line in open(args.input_file, 'r').readlines()]
    all_claims     = [Problem(**claim_data) for claim_data in all_claim_data]
    assert all(claim.model_dump() == claim_data for claim, claim_data in zip(all_claims, all_claim_data))
    
    complete = [p.parent.name for p in args.output_dir.glob('*/trace.jl')]
    if len(complete) > 0:
        rprint(f'[bold]ptest_suite: skipping {len(complete)} successful claims ...[/bold]')
        all_claims = [claim for claim in all_claims if (claim.problem_id not in complete)]
    
    rprint(f'[bold]ptest_suite: launching {len(all_claims)} claims ...[/bold]')

    # Shuffle claims
    all_claims = [all_claims[i] for i in np.random.permutation(len(all_claims))]
    
    sem = asyncio.Semaphore(args.max_concurrent)
    tasks = [
        _run_one(
            sem            = sem, 
            model_name     = args.model_name, 
            tool_whitelist = args.tool_whitelist,
            all_tools      = args.all_tools,
            no_tools       = args.no_tools,
            claim          = claim, 
            output_dir     = args.output_dir
        ) for claim in all_claims
    ]
    
    async for task in asyncio.as_completed(tasks):
        _ = await task


def parse_args():
    parser = argparse.ArgumentParser(description="Process claims from a JSONL file")
    
    # Required arguments
    parser.add_argument("--input_file",  type=str, help="Input JSONL file containing claims")
    parser.add_argument("--output_dir",  type=str, help="Output JSONL file for results")
    parser.add_argument("--run_id",      type=str)
    
    parser.add_argument("--model_name",     type=str, default="gemini/gemini-2.5-flash")
    parser.add_argument("--max_concurrent", type=int, default=8, help="Number of concurrent tasks")
    parser.add_argument("--tool_whitelist", nargs="*", help="Tools to enable")
    parser.add_argument("--all_tools",      action="store_true", help="Use all tools")
    parser.add_argument("--no_tools",       action="store_true", help="Use no tools")
    
    args = parser.parse_args()
    
    assert (args.tool_whitelist is not None) or (args.all_tools)
    
    args.run_id     = args.run_id or f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    args.output_dir = Path(args.output_dir) / args.run_id
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    return args

if __name__ == "__main__":
    args = parse_args()
    
    asyncio.run(main(args))