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

## 2026-05-16 - Layer 3: Action Format

### What This Layer Adds

Now that the project can call models and run tools manually, I added the contract between the model and the Python code.

The model is told which actions exist:

```text
search_web
read_page
finish
```

And it must return one next action as JSON:

```json
{
  "thought": "brief reason for the action",
  "action": "search_web",
  "action_input": {
    "query": "..."
  }
}
```

This is not the full agent loop yet. The model chooses an action, but Python does not automatically execute it in this layer.

Lesson:

```text
Tool use starts as a protocol.
Before the model can use tools, it needs a language for requesting them.
```

### Files Added

I separated this layer into two files:

- `prompts.py`: builds the model messages that describe the available tools and JSON format
- `agent.py`: asks the model for one next action and validates the returned JSON

This keeps `main.py` from becoming the whole application.

### First Agent-Step Test

I tested:

```bash
python3 main.py agent-step "What are the tradeoffs between RAG and long-context LLMs?" --provider ollama
python3 main.py agent-step "What are the tradeoffs between RAG and long-context LLMs?" --provider openrouter
```

Both Gemma 4 local and MiniMax via OpenRouter returned valid JSON and chose:

```text
action: search_web
```

Lesson:

```text
The action format works across both local and cloud models.
That means the next layer can focus on orchestration instead of prompt basics.
```

### Python Environment Note

During this step, I hit a Python environment mismatch. The shell used Apple's system Python 3.9.6, which did not have the installed dependencies, while the project had been using pyenv Python 3.14.4.

I added:

```text
.python-version
```

with:

```text
3.14.4
```

Lesson:

```text
Even tiny AI projects need reproducible runtime setup.
The model/tool architecture can be correct, but the wrong Python interpreter still breaks everything.
```

## 2026-05-16 - Layer 4: Orchestrator Loop

### What This Layer Adds

Layer 4 turns the pieces into an actual loop:

```text
ask model for next action
parse action JSON
run the selected Python tool
store the observation
show the observation to the model on the next step
repeat
```

This is the first point where the project starts to feel like an agent instead of separate utilities.

Lesson:

```text
The LLM is not the agent by itself.
The loop around the LLM is what creates agentic behavior.
```

### State

The orchestrator stores each step:

```python
{
    "thought": "...",
    "action": "search_web",
    "action_input": {...},
    "observation": {...}
}
```

The next prompt includes the previous steps, so the model can choose what to do next based on what already happened.

Lesson:

```text
An agent needs short-term state inside a run.
Without state, every model call is isolated.
```

### First Local Run

I tested:

```bash
python3 main.py run "What are the tradeoffs between RAG and long-context LLMs?" --provider ollama --max-steps 2
```

Gemma 4 successfully:

1. chose `search_web`
2. received DuckDuckGo results
3. chose `read_page`
4. read one of the returned URLs

It did not finish in two steps, which was expected. Two steps only validated the mechanics.

Lesson:

```text
A small max_steps setting is useful for debugging the loop.
It proves the agent can move from search results to page reading before trying a full run.
```

### OpenRouter Free Model Limit

I also tried a longer MiniMax OpenRouter run with `max_steps=4`.

It failed with:

```text
HTTP 429
minimax/minimax-m2.5:free is temporarily rate-limited upstream
```

Lesson:

```text
Agent loops multiply provider calls.
A model that works for one step may hit free-tier limits during multi-step runs.
```

This makes the local Gemma 4 path even more useful for learning the loop without provider interruptions.

### Next Improvement

The current `run` command prints the final JSON only after the loop ends. For longer runs, this feels like a black box.

A future improvement should stream progress:

```text
Step 1: search_web(...)
Step 2: read_page(...)
Step 3: ...
```

That would make the orchestrator easier to debug and easier to learn from.

## 2026-05-16 - Progress Output And Tool Errors

### Progress Output

I added progress output to the orchestrator so a run now prints each step as it happens:

```text
Step 1: asking model for next action...
Step 1: search_web
Thought: ...
Input: ...
Observation: ...
```

Lesson:

```text
Agent loops should be observable.
If the loop is a black box, debugging and learning both get harder.
```

### First Tool Error

The progress output immediately revealed a realistic problem: the model chose a Medium URL from the search results, and `read_page` received:

```text
403 Forbidden
```

This is normal web reality. Some sites block scripted requests, require JavaScript, have bot protection, or behave differently from normal browser traffic.

I changed the orchestrator so tool errors become observations instead of crashing the whole run:

```python
{
    "error": "...",
    "error_type": "HTTPError"
}
```

Lesson:

```text
In an agent loop, tool failure is information.
The model can use that observation to choose a different URL on the next step.
```

## 2026-05-16 - Finish Behavior And Markdown Report

### Stopping Rule

I made the prompt step-aware:

```text
Current step: N of max_steps
Remaining steps after this action: M
```

The model is now told to ask at every step whether it has enough evidence to finish.

Lesson:

```text
Stopping is a behavior that has to be designed.
Otherwise the agent can keep searching or reading forever.
```

### Markdown Report

I added a report writer that saves the final answer when the model chooses `finish`:

```bash
python3 main.py run "..." --output outputs/report.md
```

The report contains:

```text
Question
Answer
Sources
```

This turns the loop from a trace into a useful artifact.

### First Report Quality Issue

The first full run successfully produced a Markdown report, but it exposed a quality issue:

- the model tried to read a Medium page
- Medium returned `403 Forbidden`
- the model then read one successful page
- the model chose `finish`

That means the model treated an attempted page read as part of its evidence, even though one read failed.

I tightened the prompt:

```text
Read at least two successful, relevant pages before finishing.
A failed read_page observation with an error does not count as evidence.
```

Lesson:

```text
The orchestrator can record errors, but the prompt must teach the model how to interpret them.
```

### JSON Mode

During a rerun, Gemma 4 gathered enough evidence but returned malformed JSON when trying to finish with a long answer.

I enabled Ollama JSON mode:

```python
"format": "json"
```

This fixed the malformed final answer and allowed the agent to save a Markdown report successfully.

Lesson:

```text
Structured-output mode is valuable once answers get long.
Prompting for JSON helps, but provider-level JSON mode is more reliable when available.
```

### Remaining Quality Issue

The successful report used one full page read plus search result snippets. The model still described the evidence as if it had enough source coverage.

Lesson:

```text
Prompt instructions help, but they are not the same as programmatic guarantees.
If I truly require two successful page reads, the orchestrator should enforce that in code.
```

## 2026-05-16 - Hard Source Enforcement

### Prompt Policy vs Code Policy

I added a hard `min_sources` rule in the orchestrator:

```bash
--min-sources 2
```

The agent is not allowed to accept a `finish` action until it has enough successful `read_page` observations.

Then I discovered that "successful" also needed a definition. A page could return a tiny amount of irrelevant text and still technically count as a read. I added:

```bash
--min-source-chars 500
```

Lesson:

```text
Soft policy: ask the model to read enough sources.
Hard policy: count successful source reads in Python.
```

### Black Hole Test

I tested a different topic:

```bash
python3 main.py run "What is a black hole and what are the latest important discoveries about black holes?" --provider ollama --max-steps 7 --min-sources 2 --min-source-chars 500 --output outputs/black_holes_strict.md
```

The first strict attempt revealed another model behavior problem: after reading enough black-hole sources, the model drifted into an unrelated Roman Empire search. I improved the prompt by adding:

```text
Successful source reads: N of M required.
If the minimum source count is met, choose finish now.
Stay on the original research question.
```

The final run behaved correctly:

1. searched for black-hole information
2. read a NASA article about a rapidly growing black hole
3. read a Space.com black-hole coverage page
4. finished and saved a Markdown report

Lesson:

```text
Giving the model explicit state counters helps it obey the orchestrator's intent.
Agents need both memory of what happened and clear signals about what that memory means.
```

## 2026-05-16 - Source Metadata

### Making Observations Less Opaque

After the black-hole run worked, I noticed that "two sources" is still a pretty weak idea. A page can be long but unfocused, or useful but truncated, or come from a domain I may want to treat differently later.

I added simple metadata to `read_page`:

```python
{
    "domain": "...",
    "char_count": 6000,
    "source_char_count": 18420,
    "was_truncated": True,
}
```

The agent now uses `char_count` when deciding whether a page counts as a substantial source.

Lesson:

```text
Agent observations should be structured records, not just text dumps.
The more measured facts a tool returns, the less the model has to infer from messy context.
```

This is also a step toward source quality. Later, the agent can use the same kind of metadata for things like source type, dates, relevance scores, and extracted evidence notes.

## 2026-05-16 - Evidence Extraction

### Turning Pages Into Claims

The next improvement was to stop treating a page read as only a large block of text. After every successful `read_page`, the orchestrator now asks the model to extract compact evidence notes:

```python
{
    "relevance": "high",
    "summary": "what this page contributes",
    "notes": [
        {
            "claim": "a factual point relevant to the question",
            "supporting_text": "a short phrase from the page"
        }
    ]
}
```

This is not exposed as a new tool action. The model still chooses between:

```text
search_web
read_page
finish
```

The evidence extraction happens inside Python after a page is read.

Lesson:

```text
Not every model call has to be an agent action.
Some model calls are background processing steps that make the agent state cleaner.
```

### Why This Matters

Before this layer, the final answer depended on the model re-reading large page observations in the prompt. Now each page also carries a smaller evidence record that the final synthesis can use.

Lesson:

```text
Good agent memory is not just a transcript.
It is a progressively refined working state.
```

The Markdown report now includes an Evidence Notes section so I can inspect what the agent thought each source contributed.

During the first test, the model mostly followed the evidence schema but let one supporting quote run long. I added a small Python guard that trims `supporting_text` to 20 words.

Lesson:

```text
Formatting constraints belong in code when they matter.
The prompt can request the shape, but Python should enforce the parts that are easy to check.
```

## 2026-05-16 - Separate Answer Synthesis

### Finish Is Now A Control Signal

Originally, the model chose `finish` and wrote the final answer in the same JSON action:

```python
{
    "action": "finish",
    "action_input": {
        "answer": "..."
    }
}
```

That worked, but it mixed two jobs:

```text
decide whether research is done
write the final answer
```

I changed `finish` into a control signal:

```python
{
    "action": "finish",
    "action_input": {
        "reason": "the evidence is enough"
    }
}
```

When Python accepts that signal, it calls a separate synthesis prompt using the extracted evidence notes.

Lesson:

```text
Agent workflows get easier to debug when decisions and final generation are separate steps.
```

### Synthesis From Evidence Notes

The new synthesis step returns:

```python
{
    "answer": "...",
    "confidence": "high",
    "limitations": "..."
}
```

This means the final report is not just whatever the action-selection model decided to write in the moment. It is a dedicated generation step grounded in the cleaner evidence state.

Lesson:

```text
A good agent state can become the input to specialized model calls.
The orchestrator is not only a loop runner; it decides which model job happens when.
```

## 2026-05-16 - Source Quality Scoring

### Not All Sources Should Count Equally

The Chandra test made it obvious that two successful reads can still be very different. An official Chandra or NASA-related page should carry more weight than a secondary blog-style summary, even if both pages produce useful evidence notes.

I added another background model call after evidence extraction:

```python
{
    "source_type": "primary",
    "credibility": "high",
    "relevance": "high",
    "weight": 5,
    "reason": "..."
}
```

This score is stored in the page observation as:

```python
observation["source_quality"]
```

Lesson:

```text
Retrieval is not only about finding information.
An agent also needs to decide how much each source deserves to influence the answer.
```

### Source Quality Becomes Part Of Synthesis

The final synthesis prompt now receives source quality next to the evidence notes. It is instructed to give more weight to higher-quality and more relevant sources.

Lesson:

```text
Structured source judgment is a bridge between raw retrieval and trustworthy synthesis.
Without it, every readable page looks equally important to the model context.
```

The report now includes a Source Quality section so I can inspect and disagree with the agent's source judgments.

### First Test Exposed A Schema Slip

The first source-quality smoke test failed before reaching the scoring step because the local model returned only:

```json
{
  "thought": "..."
}
```

It forgot the required `action` and `action_input` fields.

I added a one-shot repair path: if action parsing fails, Python sends the invalid response and parser error back to the model and asks for corrected JSON.

In this test, the local model repeated the same incomplete JSON even after the repair prompt. I added one deterministic fallback: if search results exist and the model forgot the action fields, Python reads the first unread search result.

Lesson:

```text
Structured output is a protocol, but model responses can still drift.
A practical orchestrator needs small repair loops around schema boundaries.
When repair fails, deterministic fallbacks can keep simple workflows moving.
```

## 2026-05-16 - Search Planning

### Better Retrieval Starts Before Search

The agent was previously asking for one broad search query and then choosing from whatever DuckDuckGo returned. That worked, but it made retrieval too accidental.

I added a search-planning step before the action loop:

```python
{
    "queries": [
        "site:nasa.gov Chandra black hole growth",
        "site:chandra.harvard.edu Chandra black hole growth",
        "Chandra black hole growth Astrophysical Journal"
    ],
    "preferred_source_types": ["primary", "academic", "government"],
    "rationale": "..."
}
```

Python runs the planned queries, merges the results, deduplicates URLs, and stores the combined results as the first observation.

Lesson:

```text
Search quality is part of agent design.
If retrieval starts with a weak query, every later layer has to compensate.
```

### Planning Is Not The Same As Acting

The search plan is another background model call. It does not give the agent a new public action. The visible agent actions stay simple:

```text
search_web
read_page
finish
```

Lesson:

```text
Modern agent workflows often use hidden planning steps to improve tool calls.
The user-facing action space can stay simple while the orchestrator gets smarter.
```

## 2026-05-16 - Answer Reflection

### Grounding And Completeness Are Different Checks

I added a reflection step after answer synthesis. It checks two things:

```text
Is the answer grounded in the evidence notes?
Does the answer miss an important likely angle?
```

This matters because the model's own knowledge can be useful, but I do not want it to silently add unsupported facts to the final answer.

The reflection output looks like:

```python
{
    "grounded": True,
    "complete": True,
    "recommended_action": "accept",
    "issues": [],
    "missing_angles": [],
    "follow_up_queries": [],
    "revision_advice": ""
}
```

Lesson:

```text
The model's latent knowledge is safest when it proposes checks and searches, not uncited final claims.
```

### Accept, Revise, Or Search More

The reflector can recommend:

```text
accept
revise
search_more
```

For this layer, I wired in `accept` and one `revise` pass. If the reflector recommends `search_more`, the report records the missing angle and follow-up queries, but the agent does not yet expand the retrieval loop automatically.

Lesson:

```text
Reflection is a control signal.
The orchestrator decides which reflection recommendations are allowed to change the workflow.
```

## 2026-05-16 - Persistent Run Traces

### Saving The Black Box Recorder

The Markdown report is easy to read, but it does not preserve everything the agent did. I added a JSON trace file next to every Markdown report:

```text
outputs/example.md
outputs/example.md.trace.json
```

The trace includes:

```text
metadata
search plan
all actions
all observations
evidence notes
source quality scores
reflection result
final answer
```

Lesson:

```text
Agent development needs durable traces.
Without saved traces, every debugging session depends on whatever was visible in the terminal.
```

### Failed Runs Are Also Data

The CLI saves the trace even when the agent does not finish. This is intentional. A failed run can show where the loop got stuck, which source failed, or which model response broke the schema.

Lesson:

```text
Failures are part of the learning artifact.
For an agent project, failed traces are often more informative than successful reports.
```

## 2026-05-17 - Rapid-MLX Runtime Preflight

### Why Runtime Choice Is A Real Layer

The next recommended experiment was not another agent feature. It was a runtime comparison:

```text
Ollama baseline -> Rapid-MLX local server -> same JSON prompt -> compare behavior
```

This matters because the agent is now making many model calls per run:

- search planning
- next-action decisions
- evidence extraction
- source quality scoring
- answer synthesis
- reflection
- optional revision

So runtime latency and heat are no longer background details. They shape whether the loop feels usable.

### Rapid-MLX Docs Check

Rapid-MLX presents itself as an Apple Silicon local inference server with an OpenAI-compatible API. The documented local endpoint is:

```text
http://localhost:8000/v1/chat/completions
```

For a 16 GB MacBook Air, the docs recommend starting small with:

```bash
rapid-mlx serve qwen3.5-4b --port 8000
```

The app can then call the server with:

```text
base URL: http://localhost:8000/v1
model: default
```

### What Was Verified Today

Rapid-MLX was not installed locally at the start:

```bash
command -v rapid-mlx
```

returned no path.

After installing with Homebrew:

```bash
brew install raullenchai/rapid-mlx/rapid-mlx
```

Rapid-MLX reported:

```text
rapid-mlx 0.6.51
```

The install pulled a substantial dependency stack and installed:

```text
/opt/homebrew/Cellar/rapid-mlx/0.6.51
441.2 MB
```

`rapid-mlx doctor` failed inside the normal sandbox because it tried to create a diagnostic report under the Homebrew Cellar. Re-running it outside the sandbox passed:

```text
[metal] OK
[imports] OK
[cli] OK
[model_load] SKIP (download required)
Result: PASS
```

Ollama was installed and the baseline audition still worked:

```bash
/Users/khoand/.pyenv/versions/3.14.4/bin/python3 main.py audition --provider ollama
```

It returned valid action JSON:

```json
{
  "action": "search_web",
  "query": "tradeoffs between RAG and long-context LLMs"
}
```

With wall-clock timing, the Ollama audition took about:

```text
10.94 seconds
```

### Rapid-MLX 4B Attempt

I started:

```bash
rapid-mlx serve qwen3.5-4b --port 8000
```

The server downloaded the model and reached:

```text
Ready: http://localhost:8000/v1
```

But it also warned:

```text
Memory pressure warning
Model on disk: 2.9 GB
Estimated working set: 4.3 GB
Currently used by OS: 9.9 GB
Total system RAM: 16.0 GB
Projected utilization: 88%
```

The tiny audition request then crashed the Rapid-MLX server with:

```text
[METAL] Command buffer execution failed: Insufficient Memory
```

Lesson:

```text
"Fits on 16 GB" is not the same as "comfortable on this 16 GB machine right now."
The active OS memory state matters, and local runtime experiments need to watch failure modes beyond model quality.
```

### Smaller Rapid-MLX Follow-Up

`rapid-mlx models` listed smaller aliases, including:

```text
gemma3-1b
llama3-1b
bonsai-1.7b
smollm3-3b
```

I changed the project's Rapid-MLX default from `default` to:

```text
gemma3-1b
```

That first smaller attempt also taught something: `gemma3-1b` is listed as a small model, but Rapid-MLX tried to load it through its multimodal path and failed because the Homebrew install did not include the optional `mlx-vlm` dependency.

It also reported that the OS was now using about 14.3 GB of 16 GB RAM after the previous failed experiment, which made even the 1B model startup look risky.

I changed the project default again to the text-only alias:

```text
llama3-1b
```

That makes the next experiment a safer test of the OpenAI-compatible provider path without installing the vision extra.

`llama3-1b` started successfully with lower GPU memory utilization:

```bash
rapid-mlx serve llama3-1b --port 8000 --gpu-memory-utilization 0.75
```

It still warned about memory pressure because the OS was already using about 14.3 GB of 16 GB, but it completed startup.

### Rapid-MLX 1B Results

The strict audition against Rapid-MLX took about:

```text
0.96 seconds
```

It returned the right JSON content, but wrapped it in a Markdown code fence:

```text
JSON check: failed
```

Then I tested the real action schema with:

```bash
/usr/bin/time -p /Users/khoand/.pyenv/versions/3.14.4/bin/python3 main.py agent-step "What are the tradeoffs between RAG and long-context LLMs?" --provider rapid_mlx
```

That took about:

```text
1.12 seconds
```

The model returned almost-correct action JSON, but omitted the final closing brace. This means `llama3-1b` is much faster than the Ollama/Gemma 4 baseline for the tiny prompt, but it is weaker at strict schema following.

I improved the deterministic fallback so malformed near-JSON can still salvage simple quoted fields such as:

```text
thought
query
```

The parsed fallback action became:

```json
{
  "thought": "Tradeoffs between RAG and long-context LLMs Fallback: start with a web search.",
  "action": "search_web",
  "action_input": {
    "query": "tradeoffs between RAG and long-context LLMs"
  }
}
```

Lesson:

```text
Fast local inference is not enough by itself.
For an agent loop, schema reliability is part of model quality, and small models may need stronger repair and fallback paths.
```

### Code Change And Removal

I temporarily added a third model provider:

```text
rapid_mlx
```

It uses the same OpenAI-compatible response shape as OpenRouter, but points at the local Rapid-MLX server and uses a dummy bearer token because local OpenAI-compatible servers usually only need the header shape, not a real API key.

I also expanded connection error handling so a local server crash or closed socket is reported as a normal model-call failure instead of a raw Python traceback.

The temporary configuration was:

```text
RAPID_MLX_BASE_URL=http://localhost:8000/v1
RAPID_MLX_MODEL=llama3-1b
```

Lesson:

```text
An OpenAI-compatible local server is a useful provider boundary.
If the server speaks the same chat-completions shape, the orchestrator does not need to know whether the model is cloud, Ollama, or MLX.
```

After discussing the results, I removed Rapid-MLX from the active repo surface:

- removed the `rapid_mlx` provider from `llm.py`
- removed Rapid-MLX environment variables from `config.py` and `.env.example`
- removed Rapid-MLX setup and run commands from `README.md`
- updated the handoff so the next action is back to the agent feature roadmap

Final decision:

```text
Rapid-MLX has potential, but it is not useful for this repo right now.
Keep the experiment in the learning log, but keep the app focused on Ollama and OpenRouter.
```

Lesson:

```text
Removing an experimental branch is progress when the experiment has answered its question.
The repo should preserve lessons, not carry every path that seemed promising.
```

## 2026-05-17 - Automatic Search-More Reflection Loop

### What Changed

Before this layer, answer reflection could recommend:

```text
search_more
```

but Python only recorded that recommendation. The agent did not act on it.

Now the orchestrator allows one bounded reflection-triggered retrieval round:

```text
synthesize answer
-> reflect
-> if search_more:
   run follow-up searches from follow_up_queries
   read up to two new sources
   extract evidence notes
   score source quality
   synthesize again
   reflect again
```

This keeps reflection as a control signal, but gives the orchestrator permission to act on it once.

### Implementation Notes

The new helper is:

```python
_run_reflection_search_more(...)
```

It appends normal trace steps:

```text
search_web
read_page
read_page
```

That means the expanded retrieval round shows up in the same `steps` list as the rest of the run. The Markdown report and JSON trace do not need a separate data format to explain what happened.

The loop is intentionally bounded:

- use at most three follow-up queries
- read at most two new URLs
- do at most one search-more retrieval round
- reflect once more after resynthesis

If the second reflection still recommends `search_more`, the recommendation remains recorded, but the agent does not recurse.

### Verification

I tested the branch with a monkeypatched in-process run that forced the first reflection to return `search_more`.

The expected sequence happened:

```text
plan_search
finish
search_web
read_page
read_page
```

The synthesizer ran twice and the reflector ran twice:

```text
synthesize: 2
reflect: 2
```

I also ran the normal Chandra smoke test:

```bash
/Users/khoand/.pyenv/versions/3.14.4/bin/python3 main.py run "What did NASA Chandra recently find about black hole growth?" --provider ollama --max-steps 4 --min-sources 1 --min-source-chars 500 --output outputs/chandra_search_more_smoke.md
```

That live run completed successfully in about 167 seconds and saved both:

```text
outputs/chandra_search_more_smoke.md
outputs/chandra_search_more_smoke.md.trace.json
```

The live reflection recommended:

```text
accept
```

So the real smoke test confirmed the normal path still works, while the monkeypatched test confirmed the new `search_more` branch.

### Permanent Test Harness

I added the first unit test file:

```text
tests/test_agent_search_more.py
```

It uses `unittest` and `unittest.mock` to force the first reflection to return `search_more`. The test asserts that:

- the follow-up search is appended to the trace
- exactly two follow-up pages are read
- synthesis runs twice
- reflection runs twice
- the final reflection result is used

I also moved the `DDGS` import inside `search_web`, so importing `agent.py` for mocked unit tests does not load the DuckDuckGo dependency stack unless the real search tool is actually called.

Lesson:

```text
Agent autonomy should be granted in small budgets.
The reflector can ask for more evidence, but Python decides how much extra retrieval is allowed.
```

## 2026-05-17 - Layer 11: Trace Memory

### Why Memory Starts With Traces

The next recommended layer was memory and state management. I chose not to start by giving the live agent hidden long-term memory inside prompts.

Instead, I made the existing trace files the first durable memory substrate:

```text
outputs/*.trace.json -> memory/index.json
```

This feels like the right learning step because the trace already contains the facts that matter:

- what question was asked
- which sources were read successfully
- which URLs failed
- what claims were extracted
- how source quality was scored
- whether the final answer was accepted

Lesson:

```text
Agent memory should start as inspectable state, not invisible vibes in a prompt.
```

### Memory Tool

I added a new command:

```bash
python3 main.py memory
```

It reads saved `.trace.json` files and writes:

```text
memory/index.json
```

The memory index records:

```text
runs
sources
domains
failures
```

The useful parts are source and domain memory. The agent can now remember that a domain like `www.nasa.gov` produced high-quality evidence in earlier runs, while blocked pages can be remembered as failures.

Lesson:

```text
Memory is not only conversation history.
For a research agent, tool outcomes and source behavior are memory too.
```

### What Is Still Ephemeral

The live action loop has not been changed yet. The current run still keeps short-term state in `steps`, and `memory/index.json` is only an inspectable artifact.

This separation is intentional:

```text
short-term state: steps inside one run
long-term state: distilled trace memory across runs
```

Lesson:

```text
Before memory influences behavior, it should be possible to read, test, and disagree with it.
```

### Next Memory Question

The next step is to decide how memory should influence the agent:

- search planning could see useful domains from prior traces
- URL selection could avoid previously failed URLs
- source quality scoring could compare a source with past domain behavior
- user preferences could be stored separately from source/tool memory

For now, the safest next experiment is read-only memory context for search planning. That would let the model learn from prior runs without letting memory directly override evidence or final synthesis.

## 2026-05-17 - Memory-Guided Search Planning

### Feeding Memory Back Conservatively

After making the memory index inspectable, I connected it to the first safe place in the live loop:

```text
memory/index.json -> compact memory context -> search planning prompt
```

This means memory can now influence where the agent looks, but not what the final answer claims.

The search planner can see:

```text
useful domains from previous runs
previously failed URLs
```

For example, if previous traces show that `www.nasa.gov` and `www.cfa.harvard.edu` were high-quality sources, the planner may choose targeted searches using those domains. If a URL failed before, the planner is told not to target that exact URL when possible.

Lesson:

```text
Long-term memory is safest when it first improves retrieval strategy, not final truth.
```

### The Evidence Boundary

The prompt explicitly says:

```text
Memory is only retrieval guidance.
Do not treat old remembered claims as current evidence.
```

This is important because old memory can be stale, incomplete, or wrong. The current answer still has to be synthesized from fresh page reads and evidence notes in the current run.

Lesson:

```text
Memory can help the agent decide where to look.
Evidence from the current run should decide what it says.
```

### What This Teaches About Production Systems

This small layer makes a production pattern visible:

```text
past interactions -> distilled memory -> retrieved relevant memory -> prompt context
```

The hard part is not merely storing everything. The hard part is deciding:

- what to remember
- when to retrieve it
- how much to place in the prompt
- what authority the memory should have

For this project, source and tool memory now have low authority. They can guide search planning, but they cannot bypass the research loop.

I also added a `--no-memory` flag for `main.py run` so memory-guided planning can be turned off during clean experiments.

Lesson:

```text
Memory should be inspectable and controllable.
If it changes behavior, there should be a simple way to compare with memory disabled.
```

## 2026-05-17 - Semantic Memory Selection

### Why Not Word Overlap

A naive next step would have been keyword matching:

```text
new question words overlap old memory words -> include memory
```

That is easy to implement, but it is not very reliable. A question about "Chandra observations of early-universe quasars" can be related to earlier "black hole growth" memory even if the exact words differ. Conversely, two questions can share words and still need different sources.

I added a model-based memory selector instead:

```text
memory/index.json
-> compact memory candidates
-> semantic selector model call
-> selected memory context
-> search planning prompt
```

The selector returns:

```json
{
  "useful_domains": ["..."],
  "failed_urls": ["..."],
  "rationale": "..."
}
```

Lesson:

```text
Memory retrieval is itself an intelligence problem.
The system should select memory by meaning, not just surface words.
```

### Keeping The Selector Bounded

The selector still does not see raw traces. Python first builds compact candidates from the index:

```text
domains
sources
failures
```

Each candidate contains only small fields such as domain, title, previous questions, claims, source type, and quality weight. If the selector returns invalid JSON or fails, the agent falls back to no memory rather than injecting unrelated context.

Lesson:

```text
When memory selection fails, the safest fallback is less memory, not random memory.
```

## 2026-05-17 - JSON Vector Memory

### Replacing Selector Cost With Embedding Retrieval

The model-based semantic selector is smarter than keyword matching, but it adds another chat model call before every research run. I added a cheaper retrieval path:

```text
memory/index.json
-> compact memory items
-> local Ollama embeddings
-> memory/vectors.json
```

At runtime:

```text
new question
-> embedding vector
-> brute-force cosine against memory/vectors.json
-> top matches
-> search-planning memory context
```

This means the agent can retrieve semantically related memory without asking a chat model to read all candidates every time.

Lesson:

```text
Embeddings turn memory selection into vector math.
The model cost moves from every run to index-building and one cheap query embedding.
```

### Why JSON First

I intentionally used a JSON vector file instead of a vector database:

```text
memory/vectors.json
```

Each item stores:

```text
id
kind
text
metadata
embedding
```

The retrieval code loads the file, compares the question vector with every item vector, sorts by cosine similarity, and keeps the top matches.

Lesson:

```text
Brute-force vector search is the clearest way to learn the idea.
A vector database is an optimization and persistence layer, not a different concept.
```

### Fallback Order

The runtime memory path is now:

```text
if memory/vectors.json exists:
    use embedding retrieval
else if memory/index.json exists:
    use model-based semantic selector
else:
    use no memory
```

The old selector remains useful as a fallback and comparison point.

Lesson:

```text
Memory systems benefit from graceful degradation.
If the vector index is missing, the agent can still use the simpler semantic selector.
```

### First Embed Attempt

I tested:

```bash
python3 main.py memory --embed
```

The normal memory index saved successfully, but vector building could not reach Ollama:

```text
Could not reach http://localhost:11434/api/embeddings
```

I changed the CLI to report this as an expected setup issue instead of printing a traceback. To build vectors, Ollama must be running and the embedding model must be available:

```bash
ollama pull nomic-embed-text
ollama serve
```

Lesson:

```text
Vector memory adds another model dependency.
Even when embeddings are local and cheap, the runtime still needs clear setup and failure messages.
```

## 2026-05-31 - Layer 12: Evaluation Harness

### Closing The Loop

The final substantial layer is not another agent capability. It is an evaluation harness:

```bash
python3 main.py eval
```

This reads saved trace files:

```text
outputs/*.trace.json
```

and writes:

```text
outputs/evals/summary.json
outputs/evals/summary.md
```

The eval summary records:

- whether each run finished
- source count and domains
- average source quality weight
- reflection result
- whether memory was used
- duration
- a simple quality score

Lesson:

```text
An agent project should end with inspection, not just more features.
Evaluation turns traces into feedback about whether the system is improving.
```

### First Eval Result

The first eval pass found three completed Chandra traces:

```text
Traces evaluated: 3
Finished runs: 3
Memory-guided runs: 1
Average score: 9.8
```

All three used two high-quality sources from:

```text
www.cfa.harvard.edu
www.nasa.gov
```

One caveat: the vector-memory smoke trace was generated before the metadata label was fixed, so it still says `selection: semantic` even though the console output confirmed vector retrieval. Future traces should distinguish:

```text
none
llm_selector
vector
```

Lesson:

```text
Trace metadata is part of the product.
If it is inaccurate, evaluation becomes harder even when runtime behavior is correct.
```

### Why This Is A Good Stopping Point

The project now covers the full conceptual arc:

```text
model adapter
tools
action protocol
orchestration
evidence
source quality
synthesis
reflection
trace persistence
memory
vector retrieval
evaluation
```

There are many possible improvements left, but the core learning objective is complete: the machinery of a modern research agent is now visible in plain Python, with saved traces and a small eval loop to inspect behavior.
