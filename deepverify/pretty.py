import json
from rich import print as rprint
from rich.panel import Panel
from rich.text import Text

from mcp.types import Tool as MCPTool
from langchain_core.tools import StructuredTool as LCStructuredTool


def _rprint_tool_mcp(tool):
    content = Text()
    
    # Tool name
    content.append("Name:\n", style="bold yellow")
    content.append(f"{tool.name}\n\n", style="white")
    
    # Tool description
    content.append("Description:\n", style="bold yellow")
    if tool.description:
        content.append(f"{tool.description}\n\n", style="white")
    else:
        content.append("[!! No docstring in tool !!]\n\n", style="red")
    
    # Args schema
    content.append("inputSchema:\n", style="bold yellow")
    content.append(f"{json.dumps(tool.inputSchema, indent=4)}\n\n", style="white")

    content.append("outputSchema:\n", style="bold yellow")
    content.append(f"{json.dumps(tool.outputSchema, indent=4)}\n\n", style="white")
    
    rprint(Panel(
        content,
        title=f"[bold yellow]Tool: {tool.name}[/bold yellow]",
        border_style="yellow"
    ))

def _rprint_tool_langchain(tool):
    content = Text()
    
    # Tool name
    content.append("Name:\n", style="bold yellow")
    content.append(f"{tool.name}\n\n", style="white")
    
    # Tool description
    content.append("Description:\n", style="bold yellow")
    if tool.description:
        content.append(f"{tool.description}\n\n", style="white")
    else:
        content.append("[!! No docstring in tool !!]\n\n", style="red")
    
    # Args schema
    content.append("Schema:\n", style="bold yellow")
    content.append(f"{json.dumps(tool.args_schema, indent=4)}", style="white")
    
    rprint(Panel(
        content,
        title=f"[bold yellow]Tool: {tool.name}[/bold yellow]",
        border_style="yellow"
    ))

def rprint_tool(tool):
    if isinstance(tool, MCPTool):
        _rprint_tool_mcp(tool)
    elif isinstance(tool, LCStructuredTool):
        _rprint_tool_langchain(tool)
    else:
        raise ValueError(f"Unsupported tool type: {type(tool)}")