from __future__ import annotations

import json
import unittest
from unittest.mock import patch

import agent


class AgentMemorySelectionTest(unittest.TestCase):
    def test_semantic_selector_keeps_only_model_selected_memory(self) -> None:
        memory = {
            "domains": [
                {
                    "domain": "www.nasa.gov",
                    "successful_reads": 2,
                    "failed_reads": 0,
                    "average_weight": 5.0,
                    "source_types": ["government"],
                },
                {
                    "domain": "docs.python.org",
                    "successful_reads": 3,
                    "failed_reads": 0,
                    "average_weight": 5.0,
                    "source_types": ["reference"],
                },
            ],
            "sources": [
                {
                    "url": "https://www.nasa.gov/chandra",
                    "domain": "www.nasa.gov",
                    "title": "Chandra result",
                    "best_weight": 5,
                    "source_type": "government",
                    "questions": ["What did Chandra find about black hole growth?"],
                    "claims": ["Chandra observed rapid black hole growth."],
                }
            ],
            "failures": [
                {
                    "url": "https://blocked.example/chandra",
                    "domain": "blocked.example",
                    "error_type": "HTTPError",
                    "question": "What did Chandra find?",
                }
            ],
        }

        def fake_call_llm(*args, **kwargs):
            return json.dumps(
                {
                    "useful_domains": ["www.nasa.gov"],
                    "failed_urls": ["https://blocked.example/chandra"],
                    "rationale": "Chandra memory is relevant; Python docs are not.",
                }
            )

        with patch.object(agent, "call_llm", fake_call_llm):
            context = agent.select_search_memory_context(
                "What did Chandra recently find?",
                memory,
            )

        self.assertEqual(
            [domain["domain"] for domain in context["useful_domains"]],
            ["www.nasa.gov"],
        )
        self.assertEqual(
            [failure["url"] for failure in context["failed_urls"]],
            ["https://blocked.example/chandra"],
        )
        self.assertEqual(context["_selection"], "llm_selector")

    def test_selector_returns_empty_context_on_invalid_model_response(self) -> None:
        memory = {
            "domains": [
                {
                    "domain": "www.nasa.gov",
                    "successful_reads": 2,
                    "failed_reads": 0,
                    "average_weight": 5.0,
                    "source_types": ["government"],
                }
            ],
            "sources": [],
            "failures": [],
        }

        with patch.object(agent, "call_llm", lambda *args, **kwargs: "not json"):
            context = agent.select_search_memory_context("anything", memory)

        self.assertEqual(
            context,
            {"useful_domains": [], "failed_urls": [], "_selection": "none"},
        )

    def test_runtime_prefers_vector_memory_when_available(self) -> None:
        vector_memory = {
            "items": [
                {
                    "id": "domain:www.nasa.gov",
                    "kind": "domain",
                    "text": "NASA Chandra",
                    "metadata": {
                        "domain": "www.nasa.gov",
                        "successful_reads": 2,
                        "failed_reads": 0,
                        "average_weight": 5.0,
                        "source_types": ["government"],
                    },
                    "embedding": [1.0, 0.0],
                }
            ]
        }

        with (
            patch.object(agent, "_load_vector_memory", lambda: vector_memory),
            patch.object(agent, "embed_text", lambda text: [1.0, 0.0]),
            patch.object(agent, "_load_memory_index") as load_memory,
            patch.object(agent, "call_llm") as call_llm,
        ):
            context = agent.retrieve_search_memory_context(
                "What did Chandra find?",
            )

        self.assertEqual(context["useful_domains"][0]["domain"], "www.nasa.gov")
        self.assertEqual(context["_selection"], "vector")
        load_memory.assert_not_called()
        call_llm.assert_not_called()


if __name__ == "__main__":
    unittest.main()
