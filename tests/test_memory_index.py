from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from memory import (
    build_memory_index,
    build_search_memory_candidates,
    build_search_memory_context,
    find_trace_paths,
    render_memory_summary,
)


class MemoryIndexTest(unittest.TestCase):
    def test_builds_source_domain_and_failure_memory_from_traces(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_path = Path(tmpdir) / "example.md.trace.json"
            trace_path.write_text(
                json.dumps(
                    {
                        "question": "What happened?",
                        "answer": "Answer",
                        "confidence": "high",
                        "metadata": {
                            "provider": "ollama",
                            "model": None,
                            "started_at": "2026-05-17T00:00:00+00:00",
                            "duration_seconds": 12.5,
                        },
                        "steps": [
                            {
                                "thought": "read primary",
                                "action": "read_page",
                                "action_input": {"url": "https://example.com/a"},
                                "observation": {
                                    "url": "https://example.com/a",
                                    "domain": "example.com",
                                    "title": "Example Source",
                                    "text": "evidence",
                                    "char_count": 900,
                                    "evidence": {
                                        "notes": [
                                            {
                                                "claim": "A useful remembered claim",
                                                "supporting_text": "evidence",
                                            }
                                        ]
                                    },
                                    "source_quality": {
                                        "source_type": "primary",
                                        "credibility": "high",
                                        "relevance": "high",
                                        "weight": 5,
                                        "reason": "official",
                                    },
                                },
                            },
                            {
                                "thought": "blocked",
                                "action": "read_page",
                                "action_input": {"url": "https://blocked.test/page"},
                                "observation": {
                                    "error": "403 Forbidden",
                                    "error_type": "HTTPError",
                                },
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            memory = build_memory_index(find_trace_paths(tmpdir))

        self.assertEqual(memory["trace_count"], 1)
        self.assertEqual(memory["successful_run_count"], 1)
        self.assertEqual(memory["sources"][0]["url"], "https://example.com/a")
        self.assertEqual(memory["sources"][0]["best_weight"], 5)
        self.assertEqual(memory["sources"][0]["claims"], ["A useful remembered claim"])
        self.assertEqual(memory["domains"][0]["domain"], "example.com")
        self.assertEqual(memory["domains"][0]["successful_reads"], 1)
        self.assertEqual(memory["failures"][0]["domain"], "blocked.test")
        self.assertIn("Unique sources read: 1", render_memory_summary(memory))

        context = build_search_memory_context(memory)
        self.assertEqual(context["useful_domains"][0]["domain"], "example.com")
        self.assertEqual(context["failed_urls"][0]["url"], "https://blocked.test/page")

        candidates = build_search_memory_candidates(memory)
        self.assertEqual(candidates["domains"][0]["domain"], "example.com")
        self.assertEqual(candidates["sources"][0]["claims"], ["A useful remembered claim"])
        self.assertEqual(candidates["failures"][0]["url"], "https://blocked.test/page")


if __name__ == "__main__":
    unittest.main()
