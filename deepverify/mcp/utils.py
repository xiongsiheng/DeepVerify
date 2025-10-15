from deepverify import config
from typing import Optional

from langchain_mcp_adapters.client import MultiServerMCPClient
from rich import print as rprint
from fastmcp.client import Client

async def get_tools_dict(whitelist:Optional[list[str]] = None, all_tools: bool = False, mcp_url=config.MCP_URL, format="langchain"):
    """
        Get a dictionary of tools from MCP servers, in LangChain format

        Args:
            whitelist: list of tool names to return
            all_tools: If True, include all tools in the dictionary. This is not recommended for production.
            mcp_url: The URL of the main MCP server.
            include_domain_tools: Whether to include domain tools MCP server (port 8889)
    """
    if all_tools:
        rprint('[yellow]WARNING - all_tools=True - this is not recommended for production.  Consider using a tools whitelist. [/yellow]')

    if whitelist is None:
        assert all_tools, "whitelist is None and all_tools is False"

    if whitelist is not None and len(whitelist) == 0:
        assert all_tools, "whitelist is empty and all_tools is False"

    if format == "langchain":
        # Configure multiple MCP servers
        servers = {
            "main": {
                "transport": "streamable_http",
                "url": mcp_url
            }
        }

        client = MultiServerMCPClient(servers)
        
        tools = await client.get_tools()
    
    elif format == "mcp":
        print('!!!')
        client = Client(mcp_url)
        async with client:
            tools = await client.list_tools()
    
    else:
        raise ValueError(f"Invalid format: {format}")
    
    tools_dict = {tool.name: tool for tool in tools}
    
    if not all_tools:
        tools_dict = {tool_name: tools_dict[tool_name] for tool_name in whitelist}
    
    return tools_dict
