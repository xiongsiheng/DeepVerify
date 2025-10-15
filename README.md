# DeepVerify: Evidence-Based Scientific Claim Verification

**DeepVerify** empowers state-of-the-art language models with search and reasoning tools to perform **evidence-based scientific claim verification**. Given a scientific claim, the system predicts its veracity using both retrieved literature and the model’s internal knowledge.

Built on the **MCP server framework**, DeepVerify is **simple, modular, and extensible**, allowing to easily integrate custom tools and build your own specialized research agents.

<p align="center">
  <img src='https://raw.githubusercontent.com/xiongsiheng/DeepVerify/main/misc/MARS.png' width=650>
</p>

We also introduce **MARS** (Multi-Agent Reasoning System), developed upon our previous work [**SWAP**](https://github.com/xiongsiheng/SWAP). MARS leverages **entailment graphs** to guide multi-step reasoning, improving the accuracy and transparency of the verification process.

<p align="center">
  <img src='https://raw.githubusercontent.com/xiongsiheng/DeepVerify/main/misc/DeepControl.png' width=650>
</p>

Our ongoing project, **DeepControl: Enhancing Research Agents via Process-Level Verification**, extends this vision. DeepControl is a plug-in framework that verifies both the **reasoning process** and **tool usage** of deep research agents. By inspecting intermediate workflows and providing adaptive feedback, it enhances reliability and scientific rigor.
To enable **adaptive and self-improving control**, DeepControl is trained via **reinforcement fine-tuning**, enabling it to learn *when* and *how* to intervene during complex reasoning and tool-invocation sequences.

## Table of Contents
- [Installation](#installation)
- [Launch MCP Server](#launch-mcp-server)
- [Agents](#agents)
- [API Keys](#api-keys)
- [Notes](#notes)

## Installation
```bash
pixi install
```

We use [pixi](https://pixi.sh/latest/) for python environment management.  You can probablly install using other tools + the `pyproject.toml`.

## Launch MCP Server
```bash
# Load pixi environment
pixi shell

# Set environment variables
set -a && source env

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
# JDR Agent - Tool whitelist examples
python -m deepverify.agents.jdr.jdr --claim "Ultra-high performance concrete with a compressive strength greater than 45,000 psi exists." --no_tools
python -m deepverify.agents.jdr.jdr --claim "Ultra-high performance concrete with a compressive strength greater than 45,000 psi exists." --tool_whitelist generate_queries question_answer
python -m deepverify.agents.jdr.jdr --claim "Ultra-high performance concrete with a compressive strength greater than 45,000 psi exists." --all_tools
```

### Basic Usage

```bash
# Load pixi environment
pixi shell

# Set environment variables
set -a && source .env

# Run the hard-coded pipeline
python -m deepverify.agents.dspy_basic.pipeline --claim "Ultra-high performance concrete with a compressive strength greater than 45,000 psi exists."

# Run the multi-agent reasoning system (MARS)
python -m deepverify.agents.dspy_basic.mars --claim "Ultra-high performance concrete with a compressive strength greater than 45,000 psi exists."

# Run ReAct agent
python -m deepverify.agents.dspy_basic.mcp_react_agent --claim "Ultra-high performance concrete with a compressive strength greater than 45,000 psi exists."  --feasibility_assessment
```

## API Keys
You need a variety of API keys to run this - if you don't have them but you want to test, you can comment out those tools, though it definitely won't work as intended.

Copy `env.sample` to `.env` and fill in your API keys:
```bash
cp env.sample .env
# Edit .env with your API keys
```

### Required Environment Variables

```bash
# LLM keys
export GEMINI_API_KEY=
export OPENAI_API_KEY=
export ANTHROPIC_API_KEY=

# SERPAPI (for `search_google_scholar` + `search_google` tool)
export SERPAPI_API_KEY=

# JINA Reader (for `read_url` tool)
export JINA_API_KEY=

# LangSmith for tracing and debugging LangGraph agents
export LANGCHAIN_TRACING_V2="true"
export LANGCHAIN_API_KEY="YOUR_LANGCHAIN_API_KEY"
export LANGCHAIN_PROJECT="XFarScape Research Agent"
```

## Notes
- **Tool Selection** - evidence suggests that giving an agent more tools does not make it better.  If you're testing agents, you should whitelist tool names, rather than using everything available on the MCP server.  This is especially important because some of the tools are (partially) redundant (`read_url` and `read_pdf`, for instance).

## Contact
If you have any inquiries, please feel free to raise an issue or reach out to sxiong45@gatech.edu.

## Citation
```
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
