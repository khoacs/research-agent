from __future__ import annotations

import unittest

from memory import (
    build_search_memory_context_from_vector_matches,
    build_vector_memory,
    cosine_similarity,
    retrieve_vector_memory,
)


class VectorMemoryTest(unittest.TestCase):
    def test_retrieves_top_items_with_brute_force_cosine(self) -> None:
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
                    "successful_reads": 1,
                    "failed_reads": 0,
                    "average_weight": 5.0,
                    "source_types": ["reference"],
                },
            ],
            "sources": [],
            "failures": [
                {
                    "url": "https://blocked.example/chandra",
                    "domain": "blocked.example",
                    "error_type": "HTTPError",
                    "question": "What did Chandra find?",
                }
            ],
        }

        def fake_embed(text: str) -> list[float]:
            lowered = text.lower()
            if "nasa" in lowered or "chandra" in lowered:
                return [1.0, 0.0, 0.0]
            if "python" in lowered:
                return [0.0, 1.0, 0.0]
            return [0.0, 0.0, 1.0]

        vector_memory = build_vector_memory(
            memory,
            embed_fn=fake_embed,
            embedding_model="fake-embed",
        )

        matches = retrieve_vector_memory(
            vector_memory,
            query_embedding=[1.0, 0.0, 0.0],
            top_k=2,
            min_score=0.5,
        )

        self.assertEqual(matches[0]["id"], "domain:www.nasa.gov")
        self.assertEqual(matches[1]["id"], "failure:https://blocked.example/chandra")
        self.assertNotIn("embedding", matches[0])

        context = build_search_memory_context_from_vector_matches(matches)
        self.assertEqual(context["useful_domains"][0]["domain"], "www.nasa.gov")
        self.assertEqual(
            context["failed_urls"][0]["url"],
            "https://blocked.example/chandra",
        )

    def test_cosine_similarity_handles_zero_or_mismatched_vectors(self) -> None:
        self.assertEqual(cosine_similarity([1.0], [1.0, 0.0]), 0.0)
        self.assertEqual(cosine_similarity([0.0, 0.0], [1.0, 0.0]), 0.0)
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [1.0, 0.0]), 1.0)


if __name__ == "__main__":
    unittest.main()
