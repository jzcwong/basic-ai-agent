# Introduction
This repo contains two demo scenarios, both built directly on the Anthropic API with no agent framework:

| Script | What it shows |
|---|---|
| `agent.py` | A simple agent with a local tool call and memory |
| `falcon-agent.py` | An agent that connects to a locally running `falcon-mcp` server and calls its tools to interact with the CrowdStrike Falcon platform |

## Pre-requisites (both scenarios)

The virtual environment is created **outside this directory** so it isn't synced to Google Drive:

```bash
uv venv --python 3.12 ~/.venvs/basic-ai-agent
source ~/.venvs/basic-ai-agent/bin/activate
```

```bash
export ANTHROPIC_API_KEY="your-key-here"
```

---

## Scenario 1: `agent.py` (local tool call + memory)

A minimal agent that answers order status questions.

This scenario is based on https://github.com/balapriyac/ai-agent-from-scratch and the companion article https://machinelearningmastery.com/how-and-why-to-build-an-ai-agent-from-scratch-in-python/

### Key components
- Uses the Anthropic Claude Haiku 4.5 model
- **Tool call:** a local Python function (`get_order_status`) with a JSON schema the model can read. It looks up orders in an in-memory dictionary (try order IDs `4471` and `4472`)
- **Memory:** `self.messages` keeps the conversation across turns, so follow-up questions work
- **Agent loop:** the model calls tools, gets their results back, and repeats until it can answer

### Install
```bash
uv pip install anthropic
```

### Run
```bash
python agent.py
```

---

## Scenario 2: `falcon-agent.py` (Falcon via falcon-mcp)

Instead of local tools, this agent connects to the [falcon-mcp](https://github.com/CrowdStrike/falcon-mcp) server. It loads the server's Falcon tools (detections, incidents, hosts, intel, etc.) and calls them as needed to answer security questions with live data from the Falcon platform.

### Key components
- **MCP client:** connects to `falcon-mcp` and turns the server's tools into Anthropic tool definitions automatically
- **Memory:** keeps conversation history, trimmed to the last 10 user turns
- **Safety:** tool output is treated as untrusted, large responses are truncated, and you can list tools that need y/N approval in `REQUIRES_APPROVAL`

### Install
```bash
uv pip install mcp anthropic
```

### Run the falcon-mcp server locally
The server needs Falcon API credentials:

```bash
export FALCON_CLIENT_ID="your-client-id"
export FALCON_CLIENT_SECRET="your-client-secret"
export FALCON_BASE_URL="https://api.crowdstrike.com"   # change for your cloud region

uvx falcon-mcp --transport streamable-http --host 127.0.0.1 --port 8000
```

### Run the agent
In a second terminal (with the venv activated and `ANTHROPIC_API_KEY` set):

```bash
python falcon-agent.py
```

By default it connects to `http://127.0.0.1:8000/mcp` over streamable-http. Other options:

```bash
python falcon-agent.py --transport stdio      # agent launches falcon-mcp itself via uvx (needs FALCON_* env vars)
python falcon-agent.py --transport sse --url http://127.0.0.1:8000/sse
```
