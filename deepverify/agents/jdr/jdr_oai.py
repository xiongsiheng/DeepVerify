#!/usr/bin/env python
"""
    deepverify.agents.jdr
    
    (Baseline) Tool-calling agent
"""

import os
os.environ["DEFER_PYDANTIC_BUILD"] = "0"

from openai import AsyncOpenAI

import json
import asyncio
from rich.console import Console
from rich import print as rprint
from langchain_core.utils.function_calling import convert_to_openai_tool

from deepverify.cache import disk_cache_fn
from deepverify import config
from deepverify.mcp.utils import get_tools_dict

# from .pretty import print_msg, print_tool_result

__here__      = os.path.dirname(os.path.abspath(__file__))
SYSTEM_PROMPT = os.path.join(__here__, "simple_prompt.md")
# SYSTEM_PROMPT = os.path.join(__here__, "notools_prompt.md")

DO_CACHE = False

# --
# Helpers

def _clean_tool(tool):
    out = convert_to_openai_tool(tool)['function']
    out['type'] = 'function'
    return out


async def _mcp_tool_call(tool_dict, tool_call):
    _name = tool_call.name
    _args = json.loads(tool_call.arguments)
    
    if _name not in tool_dict:
        raise ValueError(f"_mcp_tool_call - ERROR - invalid tool_call={tool_call}")
    
    if DO_CACHE:
        _tool_call_fn = disk_cache_fn(tool_dict[_name].arun, fn_name=f"mcp--{_name}", cache_dir=config.CACHE_DIR / 'mcp' / _name)
    else:
        _tool_call_fn = tool_dict[_name].arun
    
    _content = await _tool_call_fn(_args)
    assert isinstance(_content, str)
    
    return {
        "type"     : "function_call_output",
        "call_id"  : tool_call.call_id,
        "output"   : _content
    }

# --
# Agent


class JDROAIAgent:
    
    def __init__(self, model_name, tool_whitelist=None, all_tools=False, no_tools=False):
        self.model_name     = model_name
        self.client         = AsyncOpenAI()
        self.system_prompt  = open(SYSTEM_PROMPT).read()
        
        self.tool_dict      = None
        self.tool_whitelist = tool_whitelist
        self.all_tools      = all_tools
        self.no_tools       = no_tools
        
        if all_tools:
            assert tool_whitelist is None, "tool_whitelist is not None when all_tools is True"
        if no_tools:
            assert tool_whitelist is None, "tool_whitelist is not None when no_tools is True"
        
        if tool_whitelist is not None:
            assert not all_tools, "all_tools is True when tool_whitelist is not None"
            assert not no_tools,  "no_tools is True when tool_whitelist is not None"

    
    async def arun(self, query, max_iters=100, verbose=True):
        if self.no_tools:
            tool_sigs      = []
        else:
            self.tool_dict = await get_tools_dict(whitelist=self.tool_whitelist, all_tools=self.all_tools)
            tool_sigs      = [_clean_tool(t) for t in self.tool_dict.values()]
        
        if verbose:
            console = Console()
        
        messages = [
            {"role" : "system", "content" : self.system_prompt},
            {"role" : "user",   "content" : query},
        ]
        
        if verbose:
            for msg in messages:
                # print_msg(msg, console=console)
                rprint(msg) # [TODO] pretty printing

        for _ in range(max_iters):
            response = await self.client.responses.create(
                model    = self.model_name,
                input    = messages,
                tools    = tool_sigs,
            )
            
            messages += response.output
            if verbose:
                for item in response.output:
                    rprint(item) # [TODO] pretty printing
            
            has_tool_call = any([item.type == "function_call" for item in response.output])
            if has_tool_call:
                tool_result_msgs = await asyncio.gather(*[
                    _mcp_tool_call(self.tool_dict, item) for item in response.output if (item.type == "function_call")
                ])
                
                if verbose:
                    for tool_result_msg in tool_result_msgs:
                        rprint(tool_result_msg) # [TODO] pretty printing
                
                messages += tool_result_msgs
            else:
                break
        
        return [m if isinstance(m, dict) else m.model_dump() for m in messages]


# --
# CLI for testing

if __name__ == "__main__":
    import argparse
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt-5-mini")
    parser.add_argument("--claim",      type=str, required=True)
    args = parser.parse_args()
    
    agent = JDROAIAgent(model_name=args.model_name)
    trace = asyncio.run(agent.arun(args.claim))
    rprint(trace)
