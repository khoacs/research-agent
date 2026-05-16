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
-> search planning
-> merged DuckDuckGo search
-> action loop
-> page reader
-> evidence extraction
-> source quality scoring
-> finish decision
-> answer synthesis
-> answer reflection
-> optional one-pass answer revision
-> Markdown report
-> JSON trace
```

Main files:

- `main.py`: CLI entrypoint and progress output
- `agent.py`: orchestrator, action parsing, evidence extraction, source quality, synthesis, reflection
- `prompts.py`: all prompt builders
- `tools.py`: `search_web`, `search_web_many`, `read_page`
- `llm.py`: raw HTTP model adapter for OpenRouter and Ollama
- `report.py`: Markdown report and JSON trace saving
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
- The user wants to try Rapid-MLX next as a possible faster Apple Silicon local runtime.

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
- The current reflection step can recommend `search_more`, but it only records that recommendation. It does not yet run another retrieval cycle.
- `outputs/` is ignored by git, so generated reports/traces are local artifacts unless the user explicitly wants to commit examples.
- `.env` is ignored and contains the OpenRouter key locally. Do not commit it.

## Recommended Next Action

Start with a Rapid-MLX local runtime experiment.

Why:

- The current Ollama/Gemma 4 path works, but full agent runs can take around 2-3 minutes.
- The MacBook Air 16GB gets warm during local inference.
- Rapid-MLX claims to be an Apple Silicon optimized, OpenAI-compatible local inference server.
- This is a good learning branch because runtime choice matters, not only model choice.

Suggested experiment:

```text
Ollama baseline
-> install/run Rapid-MLX separately
-> run the same tiny JSON prompt through both
-> compare latency, schema reliability, and heat/feel
-> if promising, add a rapid_mlx provider to llm.py
```

Keep Ollama as the stable baseline. Do not replace it immediately.

Things to check:

- Does Rapid-MLX support a model that fits MacBook Air 16GB comfortably?
- Does it expose an OpenAI-compatible `/v1/chat/completions` endpoint?
- Does it support JSON-structured output reliably enough for this agent?
- Does it actually feel faster/cooler on this machine for short agent calls?
- Does it require MLX-format models separate from Ollama model files?

If Rapid-MLX works well, update:

- `llm.py`: add `rapid_mlx` provider
- `config.py`: add Rapid-MLX base URL/model env vars
- `README.md`: document the provider
- `LEARNING_LOG.md`: record install friction, performance, and comparison with Ollama

After the Rapid-MLX experiment, continue with the automatic `search_more` reflection loop.

## Next Agent Feature

Implement the automatic `search_more` reflection loop.

Current behavior:

```text
synthesize answer
-> reflect
-> accept OR revise once OR record search_more recommendation
```

Next behavior:

```text
synthesize answer
-> reflect
-> if search_more:
   run follow-up queries
   read one or two new sources
   extract evidence
   score source quality
   synthesize again
   reflect again
```

Keep it bounded for learning:

- allow at most one reflection-triggered retrieval round
- use the reflector's `follow_up_queries`
- save the expanded trace
- document the lesson in `LEARNING_LOG.md`

## Git Checkpoint

The current checkpoint should include all layers through persistent JSON traces plus this handoff file.

Before continuing implementation, run:

```bash
git status --short
```

The next session should ideally start from a clean committed state.
