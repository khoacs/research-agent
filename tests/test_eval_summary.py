from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evals import build_eval_summary, evaluate_trace, render_eval_markdown


class EvalSummaryTest(unittest.TestCase):
    def test_evaluates_trace_quality_metrics(self) -> None:
        trace = {
            "question": "What happened?",
            "answer": "A grounded answer.",
            "confidence": "high",
            "metadata": {
                "duration_seconds": 12.5,
                "memory": {
                    "used_for_search_planning": True,
                    "selection": "vector",
                },
            },
            "reflection": {
                "grounded": True,
                "complete": True,
                "recommended_action": "accept",
            },
            "steps": [
                {
                    "action": "read_page",
                    "observation": {
                        "url": "https://example.com/a",
                        "domain": "example.com",
                        "char_count": 900,
                        "source_quality": {"weight": 5},
                    },
                }
            ],
        }

        result = evaluate_trace(trace, "trace.json")

        self.assertTrue(result["finished"])
        self.assertEqual(result["successful_source_count"], 1)
        self.assertEqual(result["domains"], ["example.com"])
        self.assertEqual(result["average_source_weight"], 5.0)
        self.assertEqual(result["memory_selection"], "vector")
        self.assertGreater(result["score"], 0)

    def test_builds_summary_and_memory_comparison_from_trace_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_dir = Path(tmpdir)
            for name, memory_used, selection in [
                ("memory.trace.json", True, "vector"),
                ("no_memory.trace.json", False, "none"),
            ]:
                (trace_dir / name).write_text(
                    json.dumps(
                        {
                            "question": "Shared question?",
                            "answer": "answer",
                            "confidence": "medium",
                            "metadata": {
                                "duration_seconds": 10,
                                "memory": {
                                    "used_for_search_planning": memory_used,
                                    "selection": selection,
                                },
                            },
                            "reflection": {
                                "grounded": True,
                                "complete": True,
                                "recommended_action": "accept",
                            },
                            "steps": [],
                        }
                    ),
                    encoding="utf-8",
                )

            summary = build_eval_summary(sorted(trace_dir.glob("*.trace.json")))

        self.assertEqual(summary["trace_count"], 2)
        self.assertEqual(summary["finished_count"], 2)
        self.assertEqual(summary["memory_used_count"], 1)
        self.assertEqual(len(summary["comparisons"]), 1)
        self.assertIn("Research Agent Eval Summary", render_eval_markdown(summary))


if __name__ == "__main__":
    unittest.main()
