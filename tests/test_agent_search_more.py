from __future__ import annotations

import unittest
from unittest.mock import patch

import agent


class SearchMoreReflectionLoopTest(unittest.TestCase):
    def test_search_more_runs_one_bounded_follow_up_round(self) -> None:
        calls = {"synthesize": 0, "reflect": 0}
        events: list[str] = []

        def fake_plan_search(question: str, provider: str = "ollama", model: str | None = None):
            return {
                "queries": ["initial query"],
                "preferred_source_types": ["primary"],
                "rationale": "initial plan",
            }

        def fake_choose_next_action(*args, **kwargs):
            return (
                "{}",
                {
                    "thought": "initial evidence is enough to finish",
                    "action": "finish",
                    "action_input": {"reason": "test finish"},
                },
            )

        def fake_search_web_many(*args, **kwargs):
            return [
                {
                    "title": "Initial",
                    "url": "https://example.com/initial",
                    "snippet": "initial",
                }
            ]

        def fake_run_tool(action: agent.AgentAction):
            if action["action"] == "search_web":
                return [
                    {"title": "A", "url": "https://example.com/a", "snippet": "a"},
                    {"title": "B", "url": "https://example.com/b", "snippet": "b"},
                    {"title": "C", "url": "https://example.com/c", "snippet": "c"},
                ]
            if action["action"] == "read_page":
                return {
                    "url": action["action_input"]["url"],
                    "domain": "example.com",
                    "title": "Example page",
                    "text": "evidence " * 100,
                    "char_count": 900,
                }
            if action["action"] == "finish":
                return {"reason": action["action_input"]["reason"]}

            raise AssertionError(f"Unexpected action: {action}")

        def fake_extract_evidence(
            question: str,
            page: dict,
            provider: str = "ollama",
            model: str | None = None,
        ):
            return {
                "relevance": "high",
                "summary": "useful",
                "notes": [{"claim": "claim", "supporting_text": "evidence"}],
            }

        def fake_score_source_quality(
            question: str,
            page: dict,
            provider: str = "ollama",
            model: str | None = None,
        ):
            return {
                "source_type": "primary",
                "credibility": "high",
                "relevance": "high",
                "weight": 5,
                "reason": "test source",
            }

        def fake_synthesize_answer(
            question: str,
            steps: list[agent.AgentStep],
            provider: str = "ollama",
            model: str | None = None,
        ):
            calls["synthesize"] += 1
            return {
                "answer": f"answer using {len(steps)} steps",
                "confidence": "medium",
                "limitations": "",
            }

        def fake_reflect_on_answer(
            question: str,
            synthesis: dict[str, str],
            steps: list[agent.AgentStep],
            provider: str = "ollama",
            model: str | None = None,
        ):
            calls["reflect"] += 1
            if calls["reflect"] == 1:
                return {
                    "grounded": True,
                    "complete": False,
                    "recommended_action": "search_more",
                    "issues": [],
                    "missing_angles": ["missing angle"],
                    "follow_up_queries": ["follow-up query"],
                    "revision_advice": "",
                }

            return {
                "grounded": True,
                "complete": True,
                "recommended_action": "accept",
                "issues": [],
                "missing_angles": [],
                "follow_up_queries": [],
                "revision_advice": "",
            }

        patches = [
            patch.object(agent, "plan_search", fake_plan_search),
            patch.object(agent, "search_web_many", fake_search_web_many),
            patch.object(agent, "choose_next_action", fake_choose_next_action),
            patch.object(agent, "run_tool", fake_run_tool),
            patch.object(agent, "extract_evidence", fake_extract_evidence),
            patch.object(agent, "score_source_quality", fake_score_source_quality),
            patch.object(agent, "synthesize_answer", fake_synthesize_answer),
            patch.object(agent, "reflect_on_answer", fake_reflect_on_answer),
        ]

        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
            patches[7],
        ):
            result = agent.run_agent(
                "test question",
                max_steps=2,
                min_sources=0,
                on_progress=lambda event, payload: events.append(event),
            )

        self.assertEqual(calls, {"synthesize": 2, "reflect": 2})
        self.assertEqual(
            [step["action"] for step in result["steps"]],
            ["plan_search", "finish", "search_web", "read_page", "read_page"],
        )
        self.assertEqual(result["answer"], "answer using 5 steps")
        self.assertEqual(result["reflection"]["recommended_action"], "accept")
        self.assertEqual(
            result["steps"][2]["action_input"],
            {"queries": ["follow-up query"]},
        )
        self.assertEqual(
            [step["action_input"]["url"] for step in result["steps"][3:]],
            ["https://example.com/a", "https://example.com/b"],
        )
        self.assertIn("reflection_search_more", events)
        self.assertEqual(events.count("synthesizing_answer"), 2)
        self.assertEqual(events.count("reflecting_answer"), 2)


if __name__ == "__main__":
    unittest.main()
