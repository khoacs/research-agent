from __future__ import annotations

import argparse
import json
from typing import Any

from agent import choose_next_action, run_agent
from llm import Provider, call_llm
from report import save_json_trace, save_markdown_report, trace_path_for_report
from tools import read_page, search_web


AUDITION_MESSAGES = [
    {
        "role": "system",
        "content": (
            "You are the model inside a simple research agent. "
            "Return only valid JSON. Do not wrap it in Markdown."
        ),
    },
    {
        "role": "user",
        "content": (
            "Question: What are the main tradeoffs between RAG and long-context LLMs?\n\n"
            "Choose the best first action from this list:\n"
            "- search_web\n"
            "- read_page\n"
            "- finish\n\n"
            "Return exactly this JSON shape:\n"
            '{ "action": "search_web", "query": "..." }'
        ),
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Research agent learning CLI.")
    subparsers = parser.add_subparsers(dest="command")

    audition_parser = subparsers.add_parser("audition", help="Audition a model backend.")
    audition_parser.add_argument(
        "--provider",
        choices=["openrouter", "ollama"],
        default="ollama",
        help="Model provider to call.",
    )
    audition_parser.add_argument(
        "--model",
        help="Override the default model for the selected provider.",
    )

    search_parser = subparsers.add_parser("search", help="Run the search_web tool.")
    search_parser.add_argument("query", help="Search query.")
    search_parser.add_argument(
        "--max-results",
        type=int,
        default=5,
        help="Maximum number of search results to return.",
    )

    read_parser = subparsers.add_parser("read-page", help="Run the read_page tool.")
    read_parser.add_argument("url", help="URL to read.")
    read_parser.add_argument(
        "--max-chars",
        type=int,
        default=6000,
        help="Maximum number of extracted text characters to return.",
    )

    agent_step_parser = subparsers.add_parser(
        "agent-step",
        help="Ask the model to choose one next research action.",
    )
    agent_step_parser.add_argument("question", help="Research question.")
    agent_step_parser.add_argument(
        "--provider",
        choices=["openrouter", "ollama"],
        default="ollama",
        help="Model provider to call.",
    )
    agent_step_parser.add_argument(
        "--model",
        help="Override the default model for the selected provider.",
    )

    run_parser = subparsers.add_parser(
        "run",
        help="Run the simple research agent loop.",
    )
    run_parser.add_argument("question", help="Research question.")
    run_parser.add_argument(
        "--provider",
        choices=["openrouter", "ollama"],
        default="ollama",
        help="Model provider to call.",
    )
    run_parser.add_argument(
        "--model",
        help="Override the default model for the selected provider.",
    )
    run_parser.add_argument(
        "--max-steps",
        type=int,
        default=5,
        help="Maximum number of agent loop steps.",
    )
    run_parser.add_argument(
        "--min-sources",
        type=int,
        default=2,
        help="Minimum successful page reads required before finish is accepted.",
    )
    run_parser.add_argument(
        "--min-source-chars",
        type=int,
        default=500,
        help="Minimum extracted text length for a page read to count as a source.",
    )
    run_parser.add_argument(
        "--output",
        help="Optional path to save the final answer as a Markdown report.",
    )

    args = parser.parse_args()

    if args.command == "search":
        results = search_web(args.query, max_results=args.max_results)
        print(json.dumps(results, indent=2, ensure_ascii=False))
        return

    if args.command == "read-page":
        page = read_page(args.url, max_chars=args.max_chars)
        print(json.dumps(page, indent=2, ensure_ascii=False))
        return

    if args.command == "agent-step":
        try:
            raw_response, action = choose_next_action(
                args.question,
                provider=args.provider,
                model=args.model,
            )
        except (RuntimeError, ValueError) as exc:
            print(f"Agent step failed: {exc}")
            return

        print("Raw model response:")
        print(raw_response)
        print()
        print("Parsed action:")
        print(json.dumps(action, indent=2, ensure_ascii=False))
        return

    if args.command == "run":
        try:
            result = run_agent(
                args.question,
                provider=args.provider,
                model=args.model,
                max_steps=args.max_steps,
                min_sources=args.min_sources,
                min_source_chars=args.min_source_chars,
                on_progress=print_progress,
            )
        except (RuntimeError, ValueError) as exc:
            print(f"Agent run failed: {exc}")
            return

        if args.output:
            if result["answer"]:
                path = save_markdown_report(result, args.output)
                print(f"\nSaved report: {path}")
            else:
                print("\nNo report saved because the agent did not finish.")
            trace_path = save_json_trace(result, trace_path_for_report(args.output))
            print(f"Saved trace: {trace_path}")

        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    if args.command is None:
        args.command = "audition"
        args.provider = "ollama"
        args.model = None

    try:
        raw_response = call_llm(
            AUDITION_MESSAGES,
            provider=args.provider,
            model=args.model,
        )
    except RuntimeError as exc:
        print(f"Model call failed: {exc}")
        return

    print("Raw model response:")
    print(raw_response)
    print()

    try:
        parsed = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        print(f"JSON check: failed ({exc})")
        return

    print("JSON check: passed")
    print(json.dumps(parsed, indent=2, ensure_ascii=False))


def print_progress(event: str, payload: dict[str, Any]) -> None:
    if event == "thinking":
        print(f"\nStep {payload['step']}: asking model for next action...")
        return

    if event == "planning_search":
        print("\nStep 1: planning targeted searches...")
        return

    if event == "search_plan":
        plan = payload["plan"]
        queries = "; ".join(plan.get("queries", []))
        print(f"Search plan: {queries}")
        print(f"Preferred sources: {', '.join(plan.get('preferred_source_types', []))}")
        return

    if event == "action":
        action = payload["action"]
        print(f"Step {payload['step']}: {action['action']}")
        print(f"Thought: {action['thought']}")
        print(f"Input: {json.dumps(action['action_input'], ensure_ascii=False)}")
        return

    if event == "observation":
        print(f"Observation: {_summarize_observation(payload['observation'])}")
        return

    if event == "extracting_evidence":
        print(f"Step {payload['step']}: extracting evidence notes...")
        return

    if event == "scoring_source":
        print(f"Step {payload['step']}: scoring source quality...")
        return

    if event == "synthesizing_answer":
        print("Synthesizing final answer from evidence notes...")
        return

    if event == "reflecting_answer":
        print("Reflecting on answer grounding and completeness...")
        return

    if event == "revising_answer":
        print("Revising answer from reflection feedback...")
        return

    if event == "reflection_search_more":
        print("Reflection requested more evidence. Running follow-up searches...")
        print(f"Follow-up queries: {'; '.join(payload['queries'])}")
        return


def _summarize_observation(observation: Any) -> str:
    if isinstance(observation, list):
        titles = [
            item.get("title", "Untitled")
            for item in observation
            if isinstance(item, dict)
        ]
        preview = "; ".join(titles[:3])
        suffix = "" if len(titles) <= 3 else f"; +{len(titles) - 3} more"
        return f"{len(observation)} results: {preview}{suffix}"

    if isinstance(observation, dict):
        if "error" in observation:
            return f"{observation.get('error_type')}: {observation.get('error')}"
        if "text" in observation:
            char_count = observation.get("char_count")
            if char_count is None:
                char_count = len(str(observation.get("text", "")))
            domain = observation.get("domain")
            domain_note = f" on {domain}" if domain else ""
            truncated_note = " (truncated)" if observation.get("was_truncated") else ""
            quality = observation.get("source_quality")
            quality_note = ""
            if isinstance(quality, dict):
                quality_note = (
                    f"; quality={quality.get('source_type')}/"
                    f"{quality.get('credibility')}, weight={quality.get('weight')}"
                )
            return (
                f"read {char_count} chars{domain_note} from "
                f"{observation.get('title') or observation.get('url')}"
                f"{truncated_note}{quality_note}"
            )
        if "answer" in observation:
            return "finish signal accepted"

    return str(observation)[:500]


if __name__ == "__main__":
    main()
