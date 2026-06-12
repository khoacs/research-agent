# Research Agent

Project 2 in the AI learning journey: a plain-Python research agent built layer by layer.

A plain-Python research agent with web search, evidence extraction, source scoring, trace memory, and evals.

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

This project was built with Python 3.14.4 via pyenv:

```bash
pyenv local 3.14.4
python3 -m pip install -r requirements.txt
```

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

For local vector memory, pull the embedding model:

```bash
ollama pull nomic-embed-text
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
    "domain": "...",
    "title": "...",
    "text": "...",
    "char_count": 6000,
    "source_char_count": 18420,
    "was_truncated": true
}
```

## Layer 3: Action Format

Ask the model to choose one next research action without running the tool yet:

```bash
python3 main.py agent-step "What are the tradeoffs between RAG and long-context LLMs?"
```

The model must return:

```python
{
    "thought": "...",
    "action": "search_web | read_page | finish",
    "action_input": {
        "query | url | reason": "..."
    }
}
```

## Layer 4: Orchestrator

Run the simple research agent loop:

```bash
python3 main.py run "What are the tradeoffs between RAG and long-context LLMs?" --provider ollama --max-steps 5 --min-sources 2 --min-source-chars 500
```

Save the final answer as Markdown:

```bash
python3 main.py run "What are the tradeoffs between RAG and long-context LLMs?" --provider ollama --max-steps 5 --min-sources 2 --min-source-chars 500 --output outputs/rag_vs_long_context.md
```

When `--output` is provided, the CLI also saves a structured trace beside the report:

```text
outputs/rag_vs_long_context.md.trace.json
```

The orchestrator repeats:

```text
retrieve relevant long-term memory for search planning when available
plan targeted searches before the action loop
ask model for next action
run the selected tool
extract evidence notes after successful page reads
score source quality after successful page reads
add the observation to history
ask again
synthesize the final answer after finish is accepted
reflect on grounding and completeness
run one bounded follow-up retrieval round if reflection says search_more
resynthesize and reflect again after follow-up retrieval
revise once if reflection says the current evidence can fix the answer
save Markdown report and JSON trace when --output is provided
```

It stops when the model chooses `finish` or when `max_steps` is reached.

The command prints progress as each step runs, then prints the full JSON result at the end.

The orchestrator enforces `--min-sources` and `--min-source-chars` in code, so a model cannot finish until enough substantial pages were successfully read. The page reader returns source metadata such as domain, extracted character counts, and truncation status so later layers can reason about source quality without guessing from the text blob alone.

## Layer 5: Evidence Notes

After every successful `read_page` action, the orchestrator asks the model to extract compact evidence notes from that page:

```python
{
    "relevance": "high | medium | low",
    "summary": "...",
    "notes": [
        {
            "claim": "...",
            "supporting_text": "..."
        }
    ]
}
```

This is an internal post-processing step, not a new agent action. The action loop stays simple, while each page observation becomes more useful for the final synthesis.

## Layer 6: Answer Synthesis

The `finish` action is now only a control signal:

```python
{
    "thought": "the evidence is sufficient",
    "action": "finish",
    "action_input": {
        "reason": "..."
    }
}
```

After Python accepts the finish action, it calls a separate answer synthesis prompt that uses the extracted evidence notes and source list. The synthesizer returns:

```python
{
    "answer": "...",
    "confidence": "high | medium | low",
    "limitations": "..."
}
```

This separates deciding when to stop from writing the final answer.

## Layer 7: Source Quality

After evidence extraction, the orchestrator scores the source itself:

```python
{
    "source_type": "primary | academic | government | news | company | blog | reference | unknown",
    "credibility": "high | medium | low",
    "relevance": "high | medium | low",
    "weight": 1,
    "reason": "..."
}
```

The final synthesizer sees these scores and is instructed to give more weight to higher-quality, more relevant sources. Reports include a `Source Quality` section so the source judgments are inspectable.

## Layer 8: Search Planning

Before the normal action loop begins, the agent now asks the model for a targeted search plan:

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

Python runs the planned queries, merges the results, removes duplicate URLs, and stores the combined results as the first observation. The normal action loop then chooses which pages to read from this richer candidate set.

## Layer 9: Answer Reflection

After answer synthesis, the agent asks a verifier to check grounding and completeness:

```python
{
    "grounded": true,
    "complete": true,
    "recommended_action": "accept | revise | search_more",
    "issues": [
        {
            "claim": "...",
            "problem": "..."
        }
    ],
    "missing_angles": ["..."],
    "follow_up_queries": ["..."],
    "revision_advice": "..."
}
```

The reflector can use model knowledge to notice likely missing angles, but it cannot add unsupported facts to the final answer. If it recommends `revise`, the agent performs one revision using the existing evidence and verifier feedback. If it recommends `search_more`, the agent runs one bounded follow-up retrieval round using the reflector's `follow_up_queries`, reads up to two new sources, extracts evidence, scores source quality, synthesizes again, and reflects again.

## Layer 10: Persistent Traces

The Markdown report is for reading. The JSON trace is for debugging, learning, and future memory.

Each trace includes:

- run metadata such as provider, model, timestamps, limits, and duration
- every action and observation
- search plan and search results
- page text metadata
- extracted evidence
- source quality
- final synthesis and reflection result

The trace is saved even when the agent does not finish, so failed runs remain inspectable.

## Layer 11: Trace Memory

Build an inspectable memory index from saved traces:

```bash
python3 main.py memory
```

By default, this reads:

```text
outputs/*.trace.json
```

and saves:

```text
memory/index.json
```

Build a simple JSON vector index too:

```bash
python3 main.py memory --embed
```

This uses Ollama's local embedding endpoint and saves:

```text
memory/vectors.json
```

The memory index summarizes:

- past runs and whether they produced an answer
- unique sources the agent successfully read
- domain-level source history and average source weight
- tool failures such as blocked `read_page` URLs
- compact claims extracted from evidence notes

The live agent now prefers `memory/vectors.json` when it exists. It embeds the new question, runs brute-force cosine similarity across the JSON vector items, and passes only the closest memory into the search-planning prompt. If no vector index exists, it falls back to the model-based semantic selector over `memory/index.json`.

Memory is used only as retrieval guidance:

- useful domains may inspire targeted `site:` searches
- previously failed URLs can be avoided
- old remembered claims are not treated as evidence for a new answer

Final synthesis still uses evidence from the current run's page reads.

To run without memory-guided search planning:

```bash
python3 main.py run "..." --no-memory
```

## Layer 12: Evaluation Harness

Evaluate saved traces:

```bash
python3 main.py eval
```

By default, this reads:

```text
outputs/*.trace.json
```

and saves:

```text
outputs/evals/summary.json
outputs/evals/summary.md
```

The eval summary records:

- whether each run finished
- source counts and source domains
- average source quality weight
- reflection result
- whether memory was used
- duration and a simple quality score

This closes the learning loop: the project can now inspect whether agent runs are completing with grounded answers and useful sources, instead of only producing one-off reports.

## Tests

Run the unit tests:

```bash
python3 -m unittest discover -s tests
```

The current test suite includes mocked tests for the bounded reflection-triggered `search_more` retrieval loop, the trace-memory index, vector-memory retrieval, memory-guided search planning, and eval summaries.
