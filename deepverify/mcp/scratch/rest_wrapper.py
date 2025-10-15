#!/usr/bin/env python
"""
    rest_wrapper.py
    
    Wraps a FastMCP server in FastAPI REST endpoints
"""

from typing import Dict, Any, Type
from pydantic import BaseModel, create_model
from fastapi import HTTPException


def create_pydantic_model_from_schema(tool_name: str, schema: Dict[str, Any]) -> Type[BaseModel]:
    """Create a Pydantic model from a JSON schema"""
    fields = {}
    
    if 'properties' in schema:
        for field_name, field_info in schema['properties'].items():
            print(field_name, field_info)
            
            field_type = Any  # Default type
            
            # Map JSON schema types to Python types
            if field_info.get('type') == 'string':
                field_type = str
            elif field_info.get('type') == 'integer':
                field_type = int
            elif field_info.get('type') == 'number':
                field_type = float
            elif field_info.get('type') == 'boolean':
                field_type = bool
            elif field_info.get('type') == 'array':
                field_type = list
            elif field_info.get('type') == 'object':
                field_type = dict
            
            # Check if field is required
            is_required = field_name in schema.get('required', [])
            
            # Create Field with metadata
            field_kwargs = {}
            
            # Add description if present
            if 'description' in field_info:
                field_kwargs['description'] = field_info['description']
            
            # Add title if present
            if 'title' in field_info:
                field_kwargs['title'] = field_info['title']
            
            # Handle default value
            if 'default' in field_info:
                default_value = field_info['default']
            elif not is_required:
                default_value = None
            else:
                default_value = ...
            
            # Create the field with Field() if we have metadata, otherwise use simple tuple
            if field_kwargs:
                from pydantic import Field
                if default_value is ...:
                    fields[field_name] = (field_type, Field(**field_kwargs))
                else:
                    fields[field_name] = (field_type, Field(default=default_value, **field_kwargs))
            else:
                fields[field_name] = (field_type, default_value)
    
    return create_model(
        f"{tool_name.title()}_Request", 
        **fields
    )


def make_tool_handler(_tool_name, _tool, _request_model):
    async def fn(request: _request_model):
        try:
            # Convert the Pydantic model to a dict for the tool
            params = request.dict() if hasattr(request, 'dict') else {}
            result = await _tool.run(params)
            return result.structured_content
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    fn.__name__ = _tool_name
    return fn


async def create_endpoints(app, mcp):
    tools = await mcp.get_tools()
    
    for tool_name, tool in tools.items():
        model_from_schema = create_pydantic_model_from_schema(tool_name, tool.parameters)
        handler = make_tool_handler(
            tool_name, 
            tool, 
            model_from_schema
        )
        app.post(f"/tools/{tool_name}")(handler)
    
    return app
