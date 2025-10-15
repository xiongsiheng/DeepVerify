#!/usr/bin/env python
"""
    deepverify.agents.jdr
    
    (Baseline) Tool-calling agent
"""

import os
os.environ["DEFER_PYDANTIC_BUILD"] = "0"

import json
import asyncio
from litellm import acompletion
from rich.console import Console
from rich import print as rprint
from langchain_core.utils.function_calling import convert_to_openai_tool

from deepverify.cache import disk_cache_fn
from deepverify import config
from deepverify.mcp.utils import get_tools_dict

from .pretty import print_msg, print_tool_result

# import litellm ; litellm._turn_on_debug()

__here__      = os.path.dirname(os.path.abspath(__file__))
SYSTEM_PROMPT = os.path.join(__here__, "simple_prompt.md")
# SYSTEM_PROMPT = os.path.join(__here__, "notools_prompt.md")

DO_CACHE = False

# --
# Helpers

def _clean_message(message):
    BAD = ['reasoning_content', 'provider_specific_fields']
    return {k:v for k,v in message.items() if k not in BAD}

def _clean_tool(tool):
    return convert_to_openai_tool(tool) # this is the format that litellm wants ... kindof annoying

async def _mcp_tool_call(tool_dict, tool_call):
    _name = tool_call.function.name
    _args = json.loads(tool_call.function.arguments)
    
    if _name not in tool_dict:
        raise ValueError(f"_mcp_tool_call - ERROR - invalid tool_call={tool_call}")
    
    if DO_CACHE:
        _tool_call_fn = disk_cache_fn(tool_dict[_name].arun, fn_name=f"mcp--{_name}", cache_dir=config.CACHE_DIR / 'mcp' / _name)
    else:
        _tool_call_fn = tool_dict[_name].arun
    
    _content = await _tool_call_fn(_args)
    assert isinstance(_content, str)
    
    return {
        "role"          : "tool",
        "name"          : tool_call.function.name,
        "tool_call_id"  : tool_call.id,
        "content"       : _content
    }

# --
# Agent

class JDRAgent:

    MODEL_CONFIGS = {
        "gemini/gemini-2.5-flash": {
            "model"            : "gemini/gemini-2.5-flash",
            "reasoning_effort" : "medium",
        },
        "gemini/gemini-2.5-pro": {
            "model"            : "gemini/gemini-2.5-pro",
        },
        "openai/gpt-5": {
            "model"            : "openai/gpt-5",
        },
        "openai/gpt-5-mini": {
            "model"            : "openai/gpt-5-mini",
        }

    }
    
    def __init__(self, model_name, tool_whitelist=None, all_tools=False, no_tools=False):
        # Default tool whitelist for JDR agent - core reasoning and search tools
        self.model_config   = self.MODEL_CONFIGS[model_name]
        self.system_prompt  = open(SYSTEM_PROMPT).read()
        self._acompletion   = disk_cache_fn(acompletion, fn_name="litellm-acompletion") if DO_CACHE else acompletion
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
                print_msg(msg, console=console)

        for _ in range(max_iters):
            out = await self._acompletion(
                **self.model_config,
                messages = messages,
                tools    = tool_sigs,
            )
            message = out.choices[0].message
            
            if verbose:
                print_msg(message, console=console)
            
            # --
            # Tool call
            
            if message.tool_calls:
                messages.append(message)
                
                tool_result_msgs = await asyncio.gather(*[
                    _mcp_tool_call(self.tool_dict, tool_call) for tool_call in message.tool_calls
                ])
                
                if verbose:
                    for tool_result_msg in tool_result_msgs:
                        print_tool_result(tool_result_msg, console=console)
                
                messages += tool_result_msgs
            else:
                messages.append({
                    "role"              : message.role,
                    "content"           : message.content,
                    "reasoning_content" : message.reasoning_content if hasattr(message, 'reasoning_content') else None,
                })
                break
        
        return [m if isinstance(m, dict) else m.model_dump() for m in messages]


# --
# CLI for testing

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gemini/gemini-2.5-flash")
    parser.add_argument("--claim",      type=str, required=True)
    parser.add_argument("--tool_whitelist", type=str, nargs='*', help="List of tools to whitelist")
    parser.add_argument("--all_tools", action="store_true", help="Allow all tools (not recommended)")
    parser.add_argument("--no_tools", action="store_true", help="Run without any tools")
    args = parser.parse_args()

    # some default tools, for reference
    # tool_whitelist = [
    #     "decompose",
    #     "generate_queries",
    #     "question_answer",
    #     "evidence_extraction",
    #     "evidence_filter",
    #     "mars"
    # ]
    
    agent = JDRAgent(
        model_name=args.model_name,
        tool_whitelist=args.tool_whitelist,
        all_tools=args.all_tools,
        no_tools=args.no_tools
    )
    trace = asyncio.run(agent.arun(args.claim))
    rprint(trace)
