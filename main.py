from __future__ import annotations

import argparse
import json

from llm import Provider, call_llm
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

    args = parser.parse_args()

    if args.command == "search":
        results = search_web(args.query, max_results=args.max_results)
        print(json.dumps(results, indent=2, ensure_ascii=False))
        return

    if args.command == "read-page":
        page = read_page(args.url, max_chars=args.max_chars)
        print(json.dumps(page, indent=2, ensure_ascii=False))
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


if __name__ == "__main__":
    main()
