#!/usr/bin/env python3
"""
    datamodels.py
"""

from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel
from enum import Enum
from typing import Union, Optional, List, Dict, Any, Tuple

# --
# Proof trees

class Node(BaseModel):
    x1_id        : str
    x2_parent    : str
    x3_statement : str

class ProofTree(BaseModel):
    x1_chain_of_thought : str
    x2_nodes            : list[Node]
    
    def pprint(self, console=None):
        """
        Pretty print a proof tree with indentation showing parent-child relationships.
        
        Args:
            self: A ProofTree object containing nodes
            console: Rich console for output
        """
        
        if console is None:
            console = Console()
        
        # Create a mapping of parent IDs to their children
        parent_to_children = {}
        for node in self.x2_nodes:
            parent_id = node.x2_parent
            if parent_id not in parent_to_children:
                parent_to_children[parent_id] = []
            parent_to_children[parent_id].append(node)
        
        # Build the tree content as a string
        content = []
        
        # Add the chain of thought
        content.append(f"[green]Chain of Thought:[/green]\n{self.x1_chain_of_thought}\n")
        
        # Add the tree structure
        content.append("[bold]Proof Tree Structure:[/bold]")
        
        # Print the root node first
        root_nodes = [n for n in self.x2_nodes if n.x2_parent == 'null']
        if root_nodes:
            root = root_nodes[0]
            content.append(f"[bold blue]{root.x3_statement}[/bold blue]")
            
            # Recursive function to add children with indentation
            def format_children(parent_id, indent=1):
                if parent_id not in parent_to_children:
                    return []
                
                result = []
                for child in parent_to_children[parent_id]:
                    result.append(f"{'  ' * indent}└─ [blue]{child.x3_statement}[/blue]")
                    result.extend(format_children(child.x1_id, indent + 1))
                return result
            
            # Add all children of the root
            content.extend(format_children(root.x1_id))
        
        # Display everything in a panel
        console.print(Panel("\n".join(content), title="Proof Tree", border_style="cyan", expand=False))
        console.print()


# --
# Literature Review Models

class LitReviewResults(BaseModel):
    """
    Results model for literature review tools.
    """
    summary: str = Field(..., description="Summary of the operation results",
                       json_schema_extra={"example": "Successfully retrieved 5 articles matching the query from Google Scholar."})
    provenance: str = Field(..., description="Source information for the results",
                          json_schema_extra={"example": "Data retrieved from Google Scholar on 2023-06-15T14:30:45Z"})
    data: Dict[str, Any] = Field(..., description="The actual result data, varies by operation type")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "summary": "Successfully retrieved 5 articles matching the query from Google Scholar.",
                "provenance": "Data retrieved from Google Scholar on 2023-06-15T14:30:45Z",
                "data": {
                    "query": "machine learning explainability",
                    "num_results": 5,
                    "results": [
                        {
                            "title": "Explaining machine learning models",
                            "authors": ["Smith, J.", "Jones, K."],
                            "year": 2022,
                            "venue": "Journal of ML Research",
                            "citations": 45
                        }
                    ]
                }
            }
        }
    }
