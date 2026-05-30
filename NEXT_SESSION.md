# Next Session Handoff

This file is a compact handoff for continuing the `research-agent` learning project in a fresh Codex session.

## Start Here

In the next session, ask Codex:

```text
Please read README.md, LEARNING_LOG.md, and NEXT_SESSION.md first.
Then help me continue the research-agent learning journey from the recommended next action.
```

## Project Goal

This is a plain-Python learning project for understanding modern agentic AI systems layer by layer.

The final app matters, but the personal learning trail matters just as much. Keep recording decisions, failures, surprises, and lessons in `LEARNING_LOG.md`.

## Current Architecture

The current pipeline is:

```text
model adapter
-> optional vector/trace memory retrieval
-> search planning
-> merged DuckDuckGo search
-> action loop
-> page reader
-> evidence extraction
-> source quality scoring
-> finish decision
-> answer synthesis
-> answer reflection
-> optional one-pass search_more retrieval
-> optional answer resynthesis and second reflection
-> optional one-pass answer revision
-> Markdown report
-> JSON trace
-> eval summary
```

Main files:

- `main.py`: CLI entrypoint and progress output
- `agent.py`: orchestrator, action parsing, evidence extraction, source quality, synthesis, reflection
- `prompts.py`: all prompt builders
- `tools.py`: `search_web`, `search_web_many`, `read_page`
- `llm.py`: raw HTTP model adapter for OpenRouter and Ollama
- `memory.py`: trace memory, vector memory, and cosine retrieval helpers
- `report.py`: Markdown report and JSON trace saving
- `evals.py`: offline evaluation of saved JSON traces
- `tests/`: mocked tests for orchestration branches, memory retrieval, vector search, prompts, and eval summaries
- `LEARNING_LOG.md`: personal learning journal
- `README.md`: project overview and layer documentation

## Current Model Setup

Local default:

```text
Ollama model: gemma4:e4b
```

Cloud default:

```text
OpenRouter model: minimax/minimax-m2.5:free
```

Notes:

- Gemma 4 runs locally on the MacBook Air 16GB, but it is slow as the pipeline gets more model calls.
- OpenRouter free models can be rate-limited.
- Qwen free route via OpenRouter previously hit provider limits.
- Google Gemini API was avoided because billing verification required a deposit/payment setup.
- Rapid-MLX was tested and removed from the active repo surface. It has potential, but on this machine it was either memory-risky or too weak at strict JSON for the current agent loop.

## Current Output Behavior

When using:

```bash
/Users/khoand/.pyenv/versions/3.14.4/bin/python3 main.py run "..." --provider ollama --output outputs/example.md
```

the CLI saves:

```text
outputs/example.md
outputs/example.md.trace.json
```

The trace is saved even if the agent does not finish.

Evaluate saved traces with:

```bash
/Users/khoand/.pyenv/versions/3.14.4/bin/python3 main.py eval
```

This saves:

```text
outputs/evals/summary.json
outputs/evals/summary.md
```

## Verification Command

Use this smoke test:

```bash
/Users/khoand/.pyenv/versions/3.14.4/bin/python3 main.py run "What did NASA Chandra recently find about black hole growth?" --provider ollama --max-steps 4 --min-sources 1 --min-source-chars 500 --output outputs/chandra_next_session_smoke.md
```

Expected behavior:

- plans targeted searches
- reads NASA and/or Harvard/Chandra sources
- extracts evidence notes
- scores source quality
- synthesizes an answer
- reflects on grounding and completeness
- saves a Markdown report and `.trace.json`

## Known Quirks

- Local Gemma may sometimes return incomplete action JSON. `agent.py` has a repair attempt and a deterministic fallback to read the first unread search result.
- Some pages return `403 Forbidden`; the orchestrator treats tool errors as observations.
- The reflection-triggered `search_more` retrieval is intentionally bounded to one round and up to two page reads.
- `outputs/` is ignored by git, so generated reports/traces are local artifacts unless the user explicitly wants to commit examples.
- `.env` is ignored and contains the OpenRouter key locally. Do not commit it.

## Recommended Next Action

Close out the project with a final checkpoint.

The project now includes memory-guided retrieval and an evaluation harness. Before closing:

1. Run the test suite:

```bash
/Users/khoand/.pyenv/versions/3.14.4/bin/python3 -m unittest discover -s tests
```

2. Run the eval summary:

```bash
/Users/khoand/.pyenv/versions/3.14.4/bin/python3 main.py eval
```

3. Review `outputs/evals/summary.md`.

4. Commit the final learning checkpoint.

Layer 11 treats saved JSON traces as the first durable memory substrate. It can build an inspectable memory index from previous runs:

```bash
/Users/khoand/.pyenv/versions/3.14.4/bin/python3 main.py memory
```

This reads `outputs/*.trace.json` and writes:

```text
memory/index.json
```

To also build the simple JSON vector index:

```bash
/Users/khoand/.pyenv/versions/3.14.4/bin/python3 main.py memory --embed
```

This uses Ollama embeddings and writes:

```text
memory/vectors.json
```

The index records past runs, unique successfully read sources, domain-level source history, evidence claims, and tool failures.

The live agent now prefers `memory/vectors.json` when present. It embeds the new question, runs brute-force cosine similarity across memory items, and passes only the closest memory into search planning. If no vector index exists, it falls back to the model-based selector over `memory/index.json`. This is deliberately conservative:

- memory can suggest useful domains for targeted searches
- memory can warn about previously failed URLs
- memory cannot replace fresh page reads as evidence

Use `--no-memory` on `main.py run` when you want a clean run that ignores the memory index.

The `search_more` branch now has a permanent mocked unit test:

```bash
/Users/khoand/.pyenv/versions/3.14.4/bin/python3 -m unittest discover -s tests
```

The memory index has a mocked unit test in `tests/test_memory_index.py`, vector retrieval has one in `tests/test_vector_memory.py`, the selector/runtime memory path has one in `tests/test_agent_memory_selection.py`, the memory-guided search prompt has one in `tests/test_search_memory_prompt.py`, and eval summaries have one in `tests/test_eval_summary.py`.

Optional future questions, outside the core learning arc:

- Should remembered failed URLs be skipped automatically after search results come back?
- Should vector-retrieved source items influence result ordering after search?
- Should the vector index be updated incrementally after every run instead of rebuilt by `main.py memory --embed`?
- How should user preferences be represented separately from source/tool memory?
- What should stay ephemeral inside one run?

## Git Checkpoint

The current checkpoint should include all layers through persistent JSON traces plus this handoff file.

Before continuing implementation, run:

```bash
git status --short
```

The next session should ideally start from a clean committed state.
