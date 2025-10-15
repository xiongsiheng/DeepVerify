import os
from pathlib import Path

# Get the directory containing this __init__.py file
_prompts_dir = Path(__file__).parent

# Read all .md files in this directory and expose them as variables
for md_file in _prompts_dir.glob("*.md"):
    # Get the variable name from the filename (without .md extension)
    var_name = md_file.stem
    
    # Read the file contents
    with open(md_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Add the variable to the module's globals
    globals()[var_name] = content
