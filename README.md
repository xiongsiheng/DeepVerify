# DeepVerify: Evidence-Based Expert-Level Scientific Claim Verification

This repository contains the code for our agentic system for scientific claim verification.

## Overview

<p align="center">
  <img src='https://raw.githubusercontent.com/xiongsiheng/DeepVerify/main/misc/Framework.png' width=850>
</p>

**DeepVerify** equips state-of-the-art language models with search and reasoning tools to perform **evidence-based expert-Level scientific claim verification**. Given an expert-Level scientific claim, the system predicts its veracity using both retrieved literature and the model’s internal knowledge.

Built on the **MCP server framework**, DeepVerify is **simple, modular, and extensible**, allowing to easily integrate custom tools and build your own specialized research agents.

For structured reasoning, DeepVerify incorporates [MARS](https://github.com/xiongsiheng/DeepVerify/tree/main/mars) **(Multi-Agent Reasoning System)**, developed upon our previous work [SWAP](https://github.com/xiongsiheng/SWAP) **(Structure-aware Planning)**. MARS leverages entailment graphs to guide multi-step reasoning, improving the accuracy and transparency of the verification process.

<p align="center">
  <img src='https://raw.githubusercontent.com/xiongsiheng/DeepVerify/main/misc/Example.png' width=650>
</p>

Models trained with our policy training framework [DeepControl](https://github.com/xiongsiheng/DeepControl/tree/main) **(Adaptive Information Control)** can be deployed locally and directly integrated into the DeepVerify agent pipeline.


## Table of Contents
- [Quickstart](#quickstart)
- [Installation](#installation)
- [Launch MCP Server](#launch-mcp-server)
- [Agents](#agents)
- [API Keys](#api-keys)
- [Notes](#notes)
- [Citation](#citation)

## Quickstart

If you just want to get the system running end-to-end:

```bash
# 1. Install dependencies
pixi install

# 2. Enter the pixi environment
pixi shell

# 3. Create your environment file
cp env.sample .env
# Edit .env with your API keys

# 4. Load environment variables
set -a && source .env && set +a

# 5. Start the MCP server
pixi run mcp-server-dev

# 6. In another shell, run an agent
python -m deepverify.agents.basic.workflow_fixed --claim "Ultra-high performance concrete with a compressive strength greater than 45,000 psi exists."
```

## Installation
```bash
pixi install
```

We use [pixi](https://pixi.sh/latest/) for Python environment management. You can probably install the project with other tools using `pyproject.toml`, but the documented workflow assumes `pixi`.

## Launch MCP Server
```bash
# Load pixi environment
pixi shell

# Set environment variables
set -a && source .env && set +a

# dev (single worker, auto-restart on code change)
pixi run mcp-server-dev

# or

# prod (multiple workers)
pixi run mcp-server
```

## Agents

### Tool Whitelisting

All agents support tool whitelisting to control which tools are available. This is **strongly recommended** as providing too many tools can hurt performance.

```bash
# Load pixi environment
pixi shell

# Load environment variables
set -a && source .env && set +a

# DeepResearch Agent - Tool whitelist examples
python -m deepverify.agents.deepresearch.deepresearch --claim "Ultra-high performance concrete with a compressive strength greater than 45,000 psi exists." --no_tools
python -m deepverify.agents.deepresearch.deepresearch --claim "Ultra-high performance concrete with a compressive strength greater than 45,000 psi exists." --tool_whitelist generate_queries question_answer
python -m deepverify.agents.deepresearch.deepresearch --claim "Ultra-high performance concrete with a compressive strength greater than 45,000 psi exists." --all_tools
```

### Basic Usage

```bash
# Load pixi environment
pixi shell

# Set environment variables
set -a && source .env && set +a

# Run the fixed workflow
python -m deepverify.agents.basic.workflow_fixed --claim "Ultra-high performance concrete with a compressive strength greater than 45,000 psi exists."

# Run the multi-agent reasoning system (MARS)
python -m deepverify.agents.basic.workflow_mars --claim "Ultra-high performance concrete with a compressive strength greater than 45,000 psi exists."

# Run ReAct agent
python -m deepverify.agents.basic.react_agent --claim "Ultra-high performance concrete with a compressive strength greater than 45,000 psi exists."  --feasibility_assessment
```

The `workflow_fixed` and `workflow_mars` modules are staged workflows. The `react_agent` module uses a ReAct-style tool-calling agent.

## API Keys
You need several API keys to run the full system. If you only want to smoke test the code path, you can disable or comment out tools, but the system will not behave as intended.

Copy `env.sample` to `.env` and fill in your API keys:
```bash
cp env.sample .env
# Edit .env with your API keys
```

### Required Environment Variables

At minimum, you should expect to need LLM credentials plus the retrieval/tooling credentials used by the tools you enable.

```bash
# LLM keys
GEMINI_API_KEY=
OPENAI_API_KEY=
ANTHROPIC_API_KEY=

# SERPAPI (for `search_google_scholar` + `search_google` tool)
SERPAPI_API_KEY=

# JINA Reader (for `read_url` tool)
JINA_API_KEY=

# LangSmith for tracing and debugging LangGraph agents
LANGCHAIN_TRACING_V2="true"
LANGCHAIN_API_KEY=
LANGCHAIN_PROJECT="DeepVerify Research Agent"
```

`LANGCHAIN_*` is optional unless you want tracing/debugging.

## Notes
- **Tool Selection** - evidence suggests that giving an agent more tools does not make it better.  If you're testing agents, you should whitelist tool names, rather than using everything available on the MCP server.  This is especially important because some of the tools are (partially) redundant (`read_url` and `read_pdf`, for instance).



## Citation

```bibtex
@article{xiong2026scaling,
  title={Scaling Search-Augmented LLM Reasoning via Adaptive Information Control},
  author={Xiong, Siheng and Gungordu, Oguzhan and Johnson, Blair and Kerce, James C and Fekri, Faramarz},
  journal={arXiv preprint arXiv:2602.01672},
  year={2026}
}
```

```bibtex
@inproceedings{xiong-etal-2025-deliberate,
    title = "Deliberate Reasoning in Language Models as Structure-Aware Planning with an Accurate World Model",
    author = "Xiong, Siheng  and
      Payani, Ali  and
      Yang, Yuan  and
      Fekri, Faramarz",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-long.1540/",
    doi = "10.18653/v1/2025.acl-long.1540",
    pages = "31900--31931",
    ISBN = "979-8-89176-251-0"
}
```