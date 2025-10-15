#!/usr/bin/env python
"""
    jdr.pretty
    
    Utilities for pretty-printing traces
"""

import json
from rich.panel import Panel
from rich.text import Text
from rich.console import Console
from pydantic import BaseModel
from typing import Optional

from typing import Dict, Any, List

class DummyFunction(BaseModel):
    name: str
    arguments: str

class DummyToolCall(BaseModel):
    index: int
    id: str
    type: str
    function: DummyFunction

class DummyMessage(BaseModel):
    role: str
    content: Optional[str] = None
    reasoning_content: Optional[str] = None
    tool_calls: Optional[List[DummyToolCall]] = None

def print_msg(message, console=None):
    if isinstance(message, dict):
        message = DummyMessage(**message)
    
    if console is None:
        console = Console()
    
    # Create a container for all message content
    reasoning_text = Text()
    if hasattr(message, 'reasoning_content') and message.reasoning_content:
        reasoning_text.append("Reasoning:\n", style="bold green")
        reasoning_text.append(message.reasoning_content, style="green")
        reasoning_text.append("\n", style="green")
    
    content_text = Text()
    if hasattr(message, 'content') and message.content:
        content_text.append("Content:\n", style="bold blue")
        content_text.append(message.content, style="blue")
        content_text.append("\n", style="blue")
    
    tool_call_text = Text()
    if hasattr(message, 'tool_calls') and message.tool_calls:
        for tool_call in message.tool_calls:
            tool_call_text.append(f"Tool Call:\n - {tool_call.function.name}(**{tool_call.function.arguments})\n", style="bold yellow")
    
    # Display the message in a nice panel
    console.print(Panel(
        reasoning_text + content_text + tool_call_text,
        title=f"[bold]{message.role} message[/bold]",
        border_style="white"
    ))


def print_tool_result(tool_result_msg, console=None, max_chars=1000):    
    if console is None:
        console = Console()
    
    _content = tool_result_msg['content']
    
    text = Text()
    
    if max_chars > 0:
        text.append(_content[:max_chars], style="white")
        if len(_content) > max_chars:
            text.append("\n\n ...... (truncated for printing) ......", style="red")
    else:
        text.append(_content, style="white")
    
    console.print(Panel(
        text,
        title="[bold]tool result[/bold]",
        border_style="white"
    ))



    