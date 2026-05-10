# Research Agent

Project 2 in the AI learning journey: a plain-Python research agent built layer by layer.

This repo also keeps a personal learning record in [LEARNING_LOG.md](./LEARNING_LOG.md): decisions, failed attempts, provider surprises, and lessons learned along the way.

## Layer 1: Model Adapter

This layer only answers one question:

```text
Can the app call different LLM providers through the same interface?
```

Current providers:

- OpenRouter with `minimax/minimax-m2.5:free`
- Ollama with `gemma4:e4b`

The shared interface is:

```python
call_llm(messages, provider)
```

## Setup

Copy the example environment file:

```bash
cp .env.example .env
```

For OpenRouter, add an API key:

```text
OPENROUTER_API_KEY=...
```

For Ollama, pull the local model:

```bash
ollama pull gemma4:e4b
```

Make sure Ollama is running:

```bash
ollama serve
```

## Run The Audition

Local Gemma 4:

```bash
python3 main.py audition --provider ollama
```

OpenRouter MiniMax:

```bash
python3 main.py audition --provider openrouter
```

OpenRouter with a specific free model:

```bash
python3 main.py audition --provider openrouter --model minimax/minimax-m2.5:free
```

The audition asks the model to return a first research-agent action as JSON. This lets us check whether the model can follow the action format before we build tools or an agent loop.

## Layer 2A: Search Tool

Run a DuckDuckGo search:

```bash
python3 main.py search "RAG vs long-context LLMs tradeoffs"
```

The `search_web` tool returns structured results:

```python
[
    {
        "title": "...",
        "url": "...",
        "snippet": "..."
    }
]
```

## Layer 2B: Page Reader Tool

Read a page and extract text:

```bash
python3 main.py read-page "https://www.llamaindex.ai/blog/towards-long-context-rag"
```

The `read_page` tool returns:

```python
{
    "url": "...",
    "title": "...",
    "text": "..."
}
```
