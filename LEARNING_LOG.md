# Research Agent Learning Log

This project is not only about producing a working research agent. It is also a record of the learning journey: what I tried, what failed, what surprised me, and how my mental model of modern AI systems changed along the way.

The final app may be simple. The valuable part is the trail of decisions.

## 2026-05-10 - Starting Project 2

### Why This Project

Project 1, the receipt OCR app, taught me the pattern:

```text
input -> one model call -> structured output
```

This project is meant to teach the next pattern:

```text
goal -> model chooses action -> tool runs -> observation returns -> repeat
```

That makes this project a small version of the architecture behind modern AI tools like ChatGPT, Codex, and Claude Code. Those systems are much bigger and more polished, but the core building blocks are recognizable:

- tool calling
- orchestration
- reflection
- memory, later
- state and context management

The goal is to understand the physics of agentic systems before using higher-level frameworks.

### Decision: Plain Python First

I intentionally chose plain Python instead of LangChain, LlamaIndex, CrewAI, or another agent framework.

The point is not that frameworks are bad. The point is to see the machinery directly:

- what JSON gets sent to the model
- how messages are structured
- how tools will be represented as Python functions
- how a model response becomes an action
- how observations get fed back into the loop
- where errors, rate limits, bad JSON, and provider issues actually happen

This felt important because modern LLM tools can generate a generic working version in one shot. The learning value comes from understanding the layers myself.

### Layer 1: Model Adapter

The first layer is only about calling models through one interface:

```python
call_llm(messages, provider, model=None)
```

Current providers:

- OpenRouter for cloud/free models
- Ollama for local models

The code is deliberately low-level. Instead of using an SDK like `openai`, it uses Python's standard library:

```python
urllib.request
json
```

This was refreshing because it made the model call feel less magical. At the bottom, an LLM API call is just:

```text
JSON payload -> HTTP POST -> JSON response
```

The agentic part does not come from the API call. It comes from the loop we build around it.

### Google Gemini Billing Surprise

Originally, Gemini 2.0 Flash seemed like a natural choice because it has a free tier. But when I tried Google AI Studio earlier, Google wanted a billing verification or prepay flow around 2000 JPY / 10 USD.

Lesson:

```text
"Free tier" does not always mean frictionless.
```

Even if usage is free or cheap, provider setup, billing verification, regional rules, and account state can become part of the engineering decision.

For this project, I decided not to depend on Gemini direct API for now.

### OpenRouter: Qwen Worked Before, But Free Routes Are Unstable

From Project 1, I had a good experience with:

```text
Qwen2.5-VL-72B-Instruct via OpenRouter
```

It performed very well for receipt OCR, especially compared with local Gemma 3 12B.

For this project, I first wanted to try a free Qwen text/instruct model:

```text
qwen/qwen3-next-80b-a3b-instruct:free
```

But the free route repeatedly returned upstream rate limits:

```text
HTTP 429
temporarily rate-limited upstream
provider_name: Venice
```

At one point, the old receipt project API key also returned:

```text
API key USD spend limit exceeded
```

After checking OpenRouter, I realized the old `receipt-ai` key had usage and a spending cap, while the new `research-agent` key was fresh. Creating a separate key for this project was the right move.

Lesson:

```text
Provider abstraction is not only about switching models.
It also protects the app from account limits, upstream provider limits, and routing weirdness.
```

### MiniMax M2.5 Became The OpenRouter Default

After Qwen free was rate-limited, I tried:

```text
minimax/minimax-m2.5:free
```

It passed the first audition prompt:

```json
{
  "action": "search_web",
  "query": "RAG vs long-context LLMs tradeoffs comparison"
}
```

This made MiniMax a better default OpenRouter model for now:

```text
OpenRouter default: minimax/minimax-m2.5:free
```

Lesson:

```text
For this project, model selection is empirical.
The question is not "which model is best in general?"
The question is "which model follows our agent action format reliably enough?"
```

### Local Gemma 4 On MacBook Air

I wanted a truly zero-money local path too, so I tried:

```text
gemma4:e4b via Ollama
```

The model downloaded successfully and ran on my MacBook Air with 16GB RAM.

Important observations from the Ollama logs:

- model run used about 9.8 GiB total memory
- default context was 4096
- the model loaded successfully
- it returned valid JSON for the audition prompt

The audition response:

```json
{
  "action": "search_web",
  "query": "tradeoffs between RAG and long-context LLMs"
}
```

Lesson:

```text
Gemma 4 e4b fits on my MacBook Air 16GB, but it is not lightweight.
For early layers, I should keep prompts and page text modest.
```

### Current Model Setup

Working defaults:

```text
Cloud/free:
OpenRouter + minimax/minimax-m2.5:free

Local:
Ollama + gemma4:e4b
```

The model layer now supports overriding a model at runtime:

```bash
python3 main.py --provider openrouter --model minimax/minimax-m2.5:free
python3 main.py --provider ollama --model gemma4:e4b
```

This reinforces an important architecture idea:

```text
The orchestrator should not care which model provider is underneath.
```

### Git And Secrets

I created a `.gitignore` before committing anything, because `.env` contains the real OpenRouter API key.

Safe pattern:

```text
.env.example = safe template to commit
.env = real local secrets, ignored by git
```

Lesson:

```text
Secret handling is part of the project architecture, even for a learning repo.
```

### What I Learned In Layer 1

- An LLM API call is just HTTP and JSON.
- SDKs are convenient, but they hide details that are useful to learn early.
- "Free" hosted models can still have rate limits, provider routing issues, and account/key limits.
- Local models remove billing friction but introduce hardware, memory, speed, and quality tradeoffs.
- A provider-neutral model adapter is the first clean abstraction in this project.
- The right first test for an agent model is not deep reasoning. It is whether it can follow the action format reliably.

### Next Layer

The next layer should be tools, starting with:

```python
search_web(query) -> list[SearchResult]
```

No agent loop yet. First, the tool should work by itself.

## 2026-05-10 - Layer 2A: Search Tool

### Search Provider Choice

For the web search tool, I wanted something completely free:

```text
no paid plan
no billing setup
no deposit
ideally no API key
```

I considered a few options:

- Tavily: clean agent-focused API, but requires an account and API key
- Brave Search API: good official API, but free plan can require a credit card
- Google Custom Search JSON API: official, but tied to Google Cloud setup and billing-adjacent quotas
- DuckDuckGo via `ddgs`: no API key, no account, no billing

I chose DuckDuckGo through the `ddgs` Python package for the first implementation.

Lesson:

```text
For a learning agent, the first tool does not need to be the most production-ready tool.
It needs to make the concept visible.
```

### Tool Definition

The first real tool is:

```python
search_web(query: str, max_results: int = 5) -> list[SearchResult]
```

The result shape is:

```python
{
    "title": "...",
    "url": "...",
    "snippet": "..."
}
```

This is important because tools should return structured observations. The future orchestrator should not have to parse messy text if the tool can return clean data.

### First Search Test

I tested:

```bash
python3 main.py search "RAG vs long-context LLMs tradeoffs" --max-results 3
```

It returned useful results, including:

- a LlamaIndex article about long-context RAG
- an arXiv paper about long-context vs RAG
- another arXiv result about RAG system configuration

Lesson:

```text
The search tool works independently before any LLM is involved.
That is the right layering.
```

### Search In Modern AI Products

I also learned that when tools like ChatGPT, Claude, or Gemini show "searching the web," they are doing a more advanced version of this same pattern:

```text
decide search is needed -> generate query -> call search provider -> read sources -> synthesize -> cite
```

The difference is that production systems add many layers:

- query rewriting
- ranking and deduplication
- source quality checks
- caching
- fallbacks across providers
- citation extraction
- privacy and safety controls

This makes the simple `search_web` tool feel like a small but real version of the same building block.

### How Search Engines Work

I also paused to understand what a search API is actually doing.

The `ddgs` package is not itself a search engine. It does not crawl the web, store pages, or maintain an index. It is a Python interface that asks DuckDuckGo for results and gives them back to my code.

The real search engine layer usually has four parts:

```text
crawl -> index -> rank -> serve
```

- Crawling: automated programs visit public web pages and follow links.
- Indexing: the search engine stores processed information about pages in a huge searchable catalog.
- Ranking: when a query arrives, algorithms choose which indexed pages are most relevant.
- Serving: the engine returns results and snippets quickly.

The important realization:

```text
A search engine does not scan the live internet every time I search.
It searches its already-built index.
```

DuckDuckGo is also not exactly the same kind of search engine as Google. Google has its own massive crawler, index, and ranking systems. DuckDuckGo uses many sources, including its own crawler and indexes, but its traditional web links are largely sourced from Bing.

Lesson:

```text
Our search_web tool is not "web knowledge."
It is a doorway into an external retrieval system's index.
```

That means the quality of the future research agent depends on both sides:

- how good the model's generated search query is
- how good the external search provider's index and ranking are
- whether we read the actual pages after search
- whether we compare and cite sources properly

### Cloud Model Reliability Correction

After adding the search tool, I re-ran the MiniMax OpenRouter audition. It appeared to take a long time, but that was partly because I had stepped away and could not immediately approve the Codex network-access prompt.

Lesson:

```text
Do not over-interpret tool timing when the workflow includes human approval steps.
Free hosted models can still have rate limits and availability issues, but this specific wait was not strong evidence that MiniMax was slow.
```

## 2026-05-10 - Layer 2B: Page Reader Tool

### Tool Definition

The second tool is:

```python
read_page(url: str, max_chars: int = 6000) -> PageContent
```

The result shape is:

```python
{
    "url": "...",
    "title": "...",
    "text": "..."
}
```

This turns a search result URL into actual text evidence that the agent can reason over later.

### First Extraction Attempt

I tested the page reader on a LlamaIndex article:

```bash
python3 main.py read-page "https://www.llamaindex.ai/blog/towards-long-context-rag" --max-chars 1500
```

The first version technically worked, but the extracted text was polluted by navigation menus and marketing text.

Lesson:

```text
Fetching HTML is easy.
Extracting the useful article text is the real problem.
```

### First Improvement

I improved the extraction rule:

```text
prefer <article>
else prefer <main>
else use <body>
```

That made the output much closer to the article content.

This is an important agent-tool lesson:

```text
Tool outputs shape model quality.
If the page reader returns noisy text, the LLM has to spend context and attention on junk.
```

Later, I may want to compare this simple BeautifulSoup approach with more specialized extraction libraries such as `trafilatura`, `readability-lxml`, or Firecrawl. For now, the simple version is good enough because the goal is to understand the layer.
