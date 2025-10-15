import os
from pathlib import Path

MCP_PORT = os.environ.get('DeepVerify_OS_MCP_PORT', 8890)
MCP_URL  = f"http://localhost:{MCP_PORT}/mcp"


# --
# Cache Dir

CACHE_DIR = Path(os.environ.get('DeepVerify_CACHE_DIR', './.cache/deepverify'))
CACHE_ENABLE = str(os.environ.get('DeepVerify_CACHE_ENABLE', 'true')).lower() == 'true'


# --
# LLM Configs

cheap_provider = 'gemini'
cheap_model    = 'gemini-2.0-flash'

rich_provider  = 'gemini'
rich_model     = 'gemini-2.0-flash-thinking-exp-01-21'

cheap_llm_kwargs = {
    'model'       : f'{cheap_provider}/{cheap_model}',
    'temperature' : 1.0,
}

rich_llm_kwargs = {
    'model'              : f'{rich_provider}/{rich_model}',
    'temperature'        : 0.7,
    'top_p'              : 0.95,
    'top_k'              : 64,
    'max_output_tokens'  : 65536,
    'response_mime_type' : 'text/plain'
}
