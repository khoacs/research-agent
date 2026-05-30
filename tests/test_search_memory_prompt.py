from __future__ import annotations

import unittest

from prompts import build_search_plan_messages


class SearchMemoryPromptTest(unittest.TestCase):
    def test_search_plan_prompt_includes_memory_as_retrieval_guidance(self) -> None:
        messages = build_search_plan_messages(
            "What did Chandra find?",
            memory_context={
                "useful_domains": [
                    {
                        "domain": "www.nasa.gov",
                        "successful_reads": 2,
                        "failed_reads": 0,
                        "average_weight": 5.0,
                        "source_types": ["government"],
                    }
                ],
                "failed_urls": [
                    {
                        "url": "https://example.com/blocked",
                        "domain": "example.com",
                        "error_type": "HTTPError",
                    }
                ],
            },
        )

        system_text = messages[0]["content"]
        user_text = messages[1]["content"]

        self.assertIn("Memory is only retrieval guidance", system_text)
        self.assertIn("www.nasa.gov", user_text)
        self.assertIn("https://example.com/blocked", user_text)
        self.assertIn("Previously failed URLs", user_text)


if __name__ == "__main__":
    unittest.main()
