from __future__ import annotations

import json
import re
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Literal, TypedDict

from llm import Provider, call_llm
from llm import embed_text
from memory import (
    build_search_memory_candidates,
    build_search_memory_context_from_vector_matches,
    load_memory_index,
    load_vector_memory,
    retrieve_vector_memory,
)
from prompts import (
    build_agent_step_messages,
    build_answer_reflection_messages,
    build_answer_revision_messages,
    build_answer_synthesis_messages,
    build_evidence_extraction_messages,
    build_memory_selection_messages,
    build_search_plan_messages,
    build_source_quality_messages,
)
from tools import read_page, search_web, search_web_many


AgentActionName = Literal["plan_search", "search_web", "read_page", "finish"]


class AgentAction(TypedDict):
    thought: str
    action: AgentActionName
    action_input: dict


class AgentStep(TypedDict):
    thought: str
    action: AgentActionName
    action_input: dict
    observation: Any


class EvidenceNote(TypedDict):
    claim: str
    supporting_text: str


class EvidenceExtraction(TypedDict):
    relevance: Literal["high", "medium", "low"]
    summary: str
    notes: list[EvidenceNote]


class SourceQuality(TypedDict):
    source_type: Literal[
        "primary",
        "academic",
        "government",
        "news",
        "company",
        "blog",
        "reference",
        "unknown",
    ]
    credibility: Literal["high", "medium", "low"]
    relevance: Literal["high", "medium", "low"]
    weight: int
    reason: str


class SearchPlan(TypedDict):
    queries: list[str]
    preferred_source_types: list[str]
    rationale: str


class ReflectionIssue(TypedDict):
    claim: str
    problem: str


class AnswerReflection(TypedDict):
    grounded: bool
    complete: bool
    recommended_action: Literal["accept", "revise", "search_more"]
    issues: list[ReflectionIssue]
    missing_angles: list[str]
    follow_up_queries: list[str]
    revision_advice: str


class AgentResult(TypedDict):
    question: str
    answer: str | None
    confidence: Literal["high", "medium", "low"] | None
    limitations: str
    reflection: AnswerReflection | None
    metadata: dict[str, Any]
    steps: list[AgentStep]


ProgressCallback = Callable[[str, dict[str, Any]], None]
MAX_REFLECTION_SEARCH_READS = 2


def choose_next_action(
    question: str,
    provider: Provider = "ollama",
    model: str | None = None,
    steps: list[AgentStep] | None = None,
    current_step: int | None = None,
    max_steps: int | None = None,
    successful_source_count: int = 0,
    min_sources: int = 2,
) -> tuple[str, AgentAction]:
    messages = build_agent_step_messages(
        question,
        steps=steps,
        current_step=current_step,
        max_steps=max_steps,
        successful_source_count=successful_source_count,
        min_sources=min_sources,
    )
    raw_response = call_llm(messages, provider=provider, model=model)
    try:
        action = parse_agent_action(raw_response)
    except ValueError as exc:
        repaired_messages = messages + [
            {
                "role": "assistant",
                "content": raw_response,
            },
            {
                "role": "user",
                "content": (
                    f"Your previous response was invalid: {exc}\n"
                    "Return the corrected JSON object now. Include thought, action, "
                    "and action_input. Do not include any text outside the JSON."
                ),
            },
        ]
        raw_response = call_llm(repaired_messages, provider=provider, model=model)
        try:
            action = parse_agent_action(raw_response)
        except ValueError:
            action = _fallback_action(raw_response, steps or [])

    return raw_response, action


def run_agent(
    question: str,
    provider: Provider = "ollama",
    model: str | None = None,
    max_steps: int = 5,
    min_sources: int = 2,
    min_source_chars: int = 500,
    use_memory: bool = True,
    on_progress: ProgressCallback | None = None,
) -> AgentResult:
    steps: list[AgentStep] = []
    started_at = _utc_now()
    metadata: dict[str, Any] = {
        "provider": provider,
        "model": model,
        "max_steps": max_steps,
        "min_sources": min_sources,
        "min_source_chars": min_source_chars,
        "started_at": started_at,
    }
    memory_context = retrieve_search_memory_context(
        question,
        provider=provider,
        model=model,
        use_memory=use_memory,
        on_progress=on_progress,
    )
    if memory_context["useful_domains"] or memory_context["failed_urls"]:
        metadata["memory"] = {
            "used_for_search_planning": True,
            "selection": memory_context.get("_selection", "unknown"),
            "useful_domain_count": len(memory_context["useful_domains"]),
            "failed_url_count": len(memory_context["failed_urls"]),
        }
        _emit_progress(
            on_progress,
            "memory_context",
            {"memory": metadata["memory"]},
        )
    else:
        metadata["memory"] = {
            "used_for_search_planning": False,
            "selection": memory_context.get("_selection", "none"),
            "useful_domain_count": 0,
            "failed_url_count": 0,
        }

    if max_steps > 1:
        _emit_progress(on_progress, "planning_search", {"step": 1})
        search_plan = plan_search(
            question,
            provider=provider,
            model=model,
            memory_context=memory_context,
        )
        _emit_progress(on_progress, "search_plan", {"step": 1, "plan": search_plan})
        observation = search_web_many(
            search_plan["queries"],
            max_results_per_query=4,
            max_total=10,
        )
        _emit_progress(
            on_progress,
            "observation",
            {
                "step": 1,
                "action": "plan_search",
                "observation": observation,
            },
        )
        steps.append(
            {
                "thought": search_plan["rationale"],
                "action": "plan_search",
                "action_input": {
                    "queries": search_plan["queries"],
                    "preferred_source_types": search_plan["preferred_source_types"],
                },
                "observation": observation,
            }
        )

    for step_number in range(len(steps) + 1, max_steps + 1):
        _emit_progress(on_progress, "thinking", {"step": step_number})
        _, action = choose_next_action(
            question,
            provider=provider,
            model=model,
            steps=steps,
            current_step=step_number,
            max_steps=max_steps,
            successful_source_count=_successful_page_read_count(
                steps,
                min_source_chars,
            ),
            min_sources=min_sources,
        )
        _emit_progress(on_progress, "action", {"step": step_number, "action": action})
        if action["action"] == "finish" and not _can_finish(
            steps,
            min_sources,
            min_source_chars,
        ):
            observation = {
                "error": (
                    f"Cannot finish yet. Need at least {min_sources} successful "
                    f"read_page observations with at least {min_source_chars} chars, "
                    f"but only have "
                    f"{_successful_page_read_count(steps, min_source_chars)}."
                ),
                "error_type": "FinishRejected",
            }
        else:
            try:
                observation = run_tool(action)
                if _should_extract_evidence(action, observation):
                    _emit_progress(
                        on_progress,
                        "extracting_evidence",
                        {"step": step_number},
                    )
                    observation["evidence"] = extract_evidence(
                        question,
                        observation,
                        provider=provider,
                        model=model,
                    )
                    _emit_progress(
                        on_progress,
                        "scoring_source",
                        {"step": step_number},
                    )
                    observation["source_quality"] = score_source_quality(
                        question,
                        observation,
                        provider=provider,
                        model=model,
                    )
            except Exception as exc:
                observation = {
                    "error": str(exc),
                    "error_type": exc.__class__.__name__,
                }
        _emit_progress(
            on_progress,
            "observation",
            {
                "step": step_number,
                "action": action["action"],
                "observation": observation,
            },
        )
        steps.append(
            {
                "thought": action["thought"],
                "action": action["action"],
                "action_input": action["action_input"],
                "observation": observation,
            }
        )

        if action["action"] == "finish" and _can_finish(
            steps[:-1],
            min_sources,
            min_source_chars,
        ):
            synthesis_steps = steps[:-1]
            _emit_progress(on_progress, "synthesizing_answer", {"step": step_number})
            synthesis = synthesize_answer(
                question,
                synthesis_steps,
                provider=provider,
                model=model,
            )
            _emit_progress(on_progress, "reflecting_answer", {"step": step_number})
            reflection = reflect_on_answer(
                question,
                synthesis,
                synthesis_steps,
                provider=provider,
                model=model,
            )
            if reflection["recommended_action"] == "search_more":
                searched_more = _run_reflection_search_more(
                    question,
                    reflection,
                    steps,
                    provider=provider,
                    model=model,
                    on_progress=on_progress,
                )
                if searched_more:
                    synthesis_steps = steps
                    _emit_progress(
                        on_progress,
                        "synthesizing_answer",
                        {"step": "reflection"},
                    )
                    synthesis = synthesize_answer(
                        question,
                        synthesis_steps,
                        provider=provider,
                        model=model,
                    )
                    _emit_progress(
                        on_progress,
                        "reflecting_answer",
                        {"step": "reflection"},
                    )
                    reflection = reflect_on_answer(
                        question,
                        synthesis,
                        synthesis_steps,
                        provider=provider,
                        model=model,
                    )
            if reflection["recommended_action"] == "revise":
                _emit_progress(on_progress, "revising_answer", {"step": step_number})
                synthesis = revise_answer(
                    question,
                    synthesis,
                    reflection,
                    synthesis_steps,
                    provider=provider,
                    model=model,
                )
            return {
                "question": question,
                "answer": synthesis["answer"],
                "confidence": synthesis["confidence"],
                "limitations": synthesis["limitations"],
                "reflection": reflection,
                "metadata": _finish_metadata(metadata),
                "steps": steps,
            }

    return {
        "question": question,
        "answer": None,
        "confidence": None,
        "limitations": "",
        "reflection": None,
        "metadata": _finish_metadata(metadata),
        "steps": steps,
    }


def run_tool(action: AgentAction) -> Any:
    if action["action"] == "search_web":
        queries = action["action_input"].get("queries")
        if isinstance(queries, list):
            return search_web_many(
                [str(query) for query in queries],
                max_results_per_query=4,
                max_total=10,
            )

        return search_web(action["action_input"]["query"], max_results=5)

    if action["action"] == "read_page":
        return read_page(action["action_input"]["url"], max_chars=6000)

    if action["action"] == "finish":
        return {"reason": action["action_input"].get("reason", "")}

    raise ValueError(f"Unknown action: {action['action']}")


def _run_reflection_search_more(
    question: str,
    reflection: AnswerReflection,
    steps: list[AgentStep],
    provider: Provider,
    model: str | None,
    on_progress: ProgressCallback | None,
) -> bool:
    queries = reflection["follow_up_queries"][:3]
    if not queries:
        return False

    search_action: AgentAction = {
        "thought": "Reflection found a missing angle; running bounded follow-up search.",
        "action": "search_web",
        "action_input": {"queries": queries},
    }
    _emit_progress(
        on_progress,
        "reflection_search_more",
        {"queries": queries},
    )
    _emit_progress(
        on_progress,
        "action",
        {"step": "reflection", "action": search_action},
    )
    try:
        search_observation = run_tool(search_action)
    except Exception as exc:
        search_observation = {
            "error": str(exc),
            "error_type": exc.__class__.__name__,
        }
    _emit_progress(
        on_progress,
        "observation",
        {
            "step": "reflection",
            "action": "search_web",
            "observation": search_observation,
        },
    )
    steps.append(
        {
            "thought": search_action["thought"],
            "action": search_action["action"],
            "action_input": search_action["action_input"],
            "observation": search_observation,
        }
    )

    if not isinstance(search_observation, list):
        return True

    for url in _unread_urls_from_results(
        search_observation,
        steps,
        limit=MAX_REFLECTION_SEARCH_READS,
    ):
        read_action: AgentAction = {
            "thought": "Reflection follow-up: read a new source for the missing angle.",
            "action": "read_page",
            "action_input": {"url": url},
        }
        _emit_progress(
            on_progress,
            "action",
            {"step": "reflection", "action": read_action},
        )
        try:
            read_observation = run_tool(read_action)
            if _should_extract_evidence(read_action, read_observation):
                _emit_progress(
                    on_progress,
                    "extracting_evidence",
                    {"step": "reflection"},
                )
                read_observation["evidence"] = extract_evidence(
                    question,
                    read_observation,
                    provider=provider,
                    model=model,
                )
                _emit_progress(
                    on_progress,
                    "scoring_source",
                    {"step": "reflection"},
                )
                read_observation["source_quality"] = score_source_quality(
                    question,
                    read_observation,
                    provider=provider,
                    model=model,
                )
        except Exception as exc:
            read_observation = {
                "error": str(exc),
                "error_type": exc.__class__.__name__,
            }
        _emit_progress(
            on_progress,
            "observation",
            {
                "step": "reflection",
                "action": "read_page",
                "observation": read_observation,
            },
        )
        steps.append(
            {
                "thought": read_action["thought"],
                "action": read_action["action"],
                "action_input": read_action["action_input"],
                "observation": read_observation,
            }
        )

    return True


def plan_search(
    question: str,
    provider: Provider = "ollama",
    model: str | None = None,
    memory_context: dict[str, Any] | None = None,
) -> SearchPlan:
    messages = build_search_plan_messages(question, memory_context=memory_context)
    raw_response = call_llm(messages, provider=provider, model=model)
    return parse_search_plan(raw_response, question)


def select_search_memory_context(
    question: str,
    memory: dict[str, Any] | None,
    provider: Provider = "ollama",
    model: str | None = None,
) -> dict[str, Any]:
    candidates = build_search_memory_candidates(memory)
    if not _has_memory_candidates(candidates):
        return _empty_memory_context()

    messages = build_memory_selection_messages(question, candidates)
    try:
        raw_response = call_llm(messages, provider=provider, model=model)
        context = parse_search_memory_selection(raw_response, candidates)
        context["_selection"] = "llm_selector"
        return context
    except (RuntimeError, ValueError):
        return _empty_memory_context()


def retrieve_search_memory_context(
    question: str,
    provider: Provider = "ollama",
    model: str | None = None,
    use_memory: bool = True,
    on_progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    if not use_memory:
        return _empty_memory_context()

    vector_memory = _load_vector_memory()
    if vector_memory is not None:
        _emit_progress(on_progress, "retrieving_vector_memory", {})
        try:
            matches = retrieve_vector_memory(
                vector_memory,
                embed_text(question),
            )
        except RuntimeError:
            matches = []
        context = build_search_memory_context_from_vector_matches(matches)
        if context["useful_domains"] or context["failed_urls"]:
            context["_selection"] = "vector"
            return context

    memory = _load_memory_index()
    if memory is not None:
        _emit_progress(on_progress, "selecting_memory", {})
    return select_search_memory_context(
        question,
        memory,
        provider=provider,
        model=model,
    )


def extract_evidence(
    question: str,
    page: dict[str, Any],
    provider: Provider = "ollama",
    model: str | None = None,
) -> EvidenceExtraction:
    messages = build_evidence_extraction_messages(question, page)
    raw_response = call_llm(messages, provider=provider, model=model)
    return parse_evidence_extraction(raw_response)


def synthesize_answer(
    question: str,
    steps: list[AgentStep],
    provider: Provider = "ollama",
    model: str | None = None,
) -> dict[str, str]:
    messages = build_answer_synthesis_messages(question, steps)
    raw_response = call_llm(messages, provider=provider, model=model)
    return parse_answer_synthesis(raw_response)


def reflect_on_answer(
    question: str,
    synthesis: dict[str, str],
    steps: list[AgentStep],
    provider: Provider = "ollama",
    model: str | None = None,
) -> AnswerReflection:
    messages = build_answer_reflection_messages(
        question,
        synthesis["answer"],
        synthesis["confidence"],
        synthesis["limitations"],
        steps,
    )
    raw_response = call_llm(messages, provider=provider, model=model)
    return parse_answer_reflection(raw_response)


def revise_answer(
    question: str,
    synthesis: dict[str, str],
    reflection: AnswerReflection,
    steps: list[AgentStep],
    provider: Provider = "ollama",
    model: str | None = None,
) -> dict[str, str]:
    messages = build_answer_revision_messages(
        question,
        synthesis["answer"],
        reflection,
        steps,
    )
    raw_response = call_llm(messages, provider=provider, model=model)
    return parse_answer_synthesis(raw_response)


def score_source_quality(
    question: str,
    page: dict[str, Any],
    provider: Provider = "ollama",
    model: str | None = None,
) -> SourceQuality:
    messages = build_source_quality_messages(question, page)
    raw_response = call_llm(messages, provider=provider, model=model)
    return parse_source_quality(raw_response)


def parse_answer_synthesis(raw_response: str) -> dict[str, str]:
    data = _parse_json_object(raw_response)

    answer = data.get("answer")
    confidence = data.get("confidence")
    limitations = data.get("limitations", "")

    if not isinstance(answer, str) or not answer.strip():
        raise ValueError(f"Answer synthesis requires a non-empty answer: {raw_response}")
    if confidence not in {"high", "medium", "low"}:
        raise ValueError(
            f"Answer synthesis confidence must be high, medium, or low: {raw_response}"
        )
    if not isinstance(limitations, str):
        raise ValueError(f"Answer synthesis limitations must be a string: {raw_response}")

    return {
        "answer": answer.strip(),
        "confidence": confidence,
        "limitations": limitations.strip(),
    }


def parse_search_plan(raw_response: str, question: str) -> SearchPlan:
    data = _parse_json_object(raw_response)

    queries = data.get("queries")
    preferred_source_types = data.get("preferred_source_types")
    rationale = data.get("rationale", "")

    if not isinstance(queries, list):
        queries = []
    parsed_queries = [
        query.strip()
        for query in queries
        if isinstance(query, str) and query.strip()
    ][:3]
    if not parsed_queries:
        parsed_queries = [question]

    if not isinstance(preferred_source_types, list):
        preferred_source_types = []
    parsed_source_types = [
        source_type.strip()
        for source_type in preferred_source_types
        if isinstance(source_type, str) and source_type.strip()
    ][:5]
    if not parsed_source_types:
        parsed_source_types = ["primary", "academic", "government"]

    if not isinstance(rationale, str) or not rationale.strip():
        rationale = "Generated targeted search queries for the research question."

    return {
        "queries": parsed_queries,
        "preferred_source_types": parsed_source_types,
        "rationale": rationale.strip(),
    }


def parse_search_memory_selection(
    raw_response: str,
    candidates: dict[str, Any],
) -> dict[str, Any]:
    data = _parse_json_object(raw_response)
    selected_domains = _string_list(data.get("useful_domains"), limit=5)
    selected_failed_urls = _string_list(data.get("failed_urls"), limit=5)

    domain_records = {
        domain.get("domain"): domain
        for domain in candidates.get("domains", [])
        if isinstance(domain, dict) and isinstance(domain.get("domain"), str)
    }
    failure_records = {
        failure.get("url"): failure
        for failure in candidates.get("failures", [])
        if isinstance(failure, dict) and isinstance(failure.get("url"), str)
    }

    useful_domains = []
    for domain in selected_domains:
        record = domain_records.get(domain)
        if record is not None:
            useful_domains.append(record)

    failed_urls = []
    for url in selected_failed_urls:
        record = failure_records.get(url)
        if record is not None:
            failed_urls.append(record)

    return {
        "useful_domains": useful_domains,
        "failed_urls": failed_urls,
    }


def parse_answer_reflection(raw_response: str) -> AnswerReflection:
    data = _parse_json_object(raw_response)

    grounded = data.get("grounded")
    complete = data.get("complete")
    recommended_action = data.get("recommended_action")
    issues = data.get("issues", [])
    missing_angles = data.get("missing_angles", [])
    follow_up_queries = data.get("follow_up_queries", [])
    revision_advice = data.get("revision_advice", "")

    if not isinstance(grounded, bool):
        raise ValueError(f"Reflection grounded must be a boolean: {raw_response}")
    if not isinstance(complete, bool):
        raise ValueError(f"Reflection complete must be a boolean: {raw_response}")
    if recommended_action not in {"accept", "revise", "search_more"}:
        raise ValueError(
            f"Reflection recommended_action must be accept, revise, or search_more: {raw_response}"
        )
    if not isinstance(revision_advice, str):
        raise ValueError(f"Reflection revision_advice must be a string: {raw_response}")

    parsed_issues: list[ReflectionIssue] = []
    if isinstance(issues, list):
        for issue in issues:
            if not isinstance(issue, dict):
                continue
            claim = issue.get("claim")
            problem = issue.get("problem")
            if isinstance(claim, str) and isinstance(problem, str):
                parsed_issues.append(
                    {
                        "claim": claim.strip(),
                        "problem": problem.strip(),
                    }
                )

    parsed_missing_angles = _string_list(missing_angles, limit=5)
    parsed_follow_up_queries = _string_list(follow_up_queries, limit=5)

    return {
        "grounded": grounded,
        "complete": complete,
        "recommended_action": recommended_action,
        "issues": parsed_issues,
        "missing_angles": parsed_missing_angles,
        "follow_up_queries": parsed_follow_up_queries,
        "revision_advice": revision_advice.strip(),
    }


def parse_source_quality(raw_response: str) -> SourceQuality:
    data = _parse_json_object(raw_response)

    source_type = data.get("source_type")
    credibility = data.get("credibility")
    relevance = data.get("relevance")
    weight = data.get("weight")
    reason = data.get("reason")

    source_types = {
        "primary",
        "academic",
        "government",
        "news",
        "company",
        "blog",
        "reference",
        "unknown",
    }
    quality_levels = {"high", "medium", "low"}

    if source_type not in source_types:
        raise ValueError(f"Unknown source_type in source quality: {raw_response}")
    if credibility not in quality_levels:
        raise ValueError(f"Source credibility must be high, medium, or low: {raw_response}")
    if relevance not in quality_levels:
        raise ValueError(f"Source relevance must be high, medium, or low: {raw_response}")
    if not isinstance(weight, int) or not 1 <= weight <= 5:
        raise ValueError(f"Source weight must be an integer from 1 to 5: {raw_response}")
    if not isinstance(reason, str):
        raise ValueError(f"Source quality reason must be a string: {raw_response}")

    return {
        "source_type": source_type,
        "credibility": credibility,
        "relevance": relevance,
        "weight": weight,
        "reason": reason.strip(),
    }


def parse_evidence_extraction(raw_response: str) -> EvidenceExtraction:
    data = _parse_json_object(raw_response)

    relevance = data.get("relevance")
    summary = data.get("summary")
    notes = data.get("notes")

    if relevance not in {"high", "medium", "low"}:
        raise ValueError(f"Evidence relevance must be high, medium, or low: {raw_response}")
    if not isinstance(summary, str):
        raise ValueError(f"Evidence summary must be a string: {raw_response}")
    if not isinstance(notes, list):
        raise ValueError(f"Evidence notes must be a list: {raw_response}")

    parsed_notes: list[EvidenceNote] = []
    for note in notes[:5]:
        if not isinstance(note, dict):
            continue
        claim = note.get("claim")
        supporting_text = note.get("supporting_text")
        if isinstance(claim, str) and isinstance(supporting_text, str):
            parsed_notes.append(
                {
                    "claim": claim.strip(),
                    "supporting_text": _limit_words(supporting_text.strip(), 20),
                }
            )

    return {
        "relevance": relevance,
        "summary": summary.strip(),
        "notes": parsed_notes,
    }


def _should_extract_evidence(action: AgentAction, observation: Any) -> bool:
    return (
        action["action"] == "read_page"
        and isinstance(observation, dict)
        and not observation.get("error")
        and bool(observation.get("text"))
    )


def _limit_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text

    return " ".join(words[:max_words]) + "..."


def _string_list(value: Any, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []

    return [
        item.strip()
        for item in value
        if isinstance(item, str) and item.strip()
    ][:limit]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _finish_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    finished_at = _utc_now()
    started_at = metadata.get("started_at")
    duration_seconds = None
    if isinstance(started_at, str):
        try:
            started = datetime.fromisoformat(started_at)
            finished = datetime.fromisoformat(finished_at)
            duration_seconds = round((finished - started).total_seconds(), 3)
        except ValueError:
            duration_seconds = None

    return {
        **metadata,
        "finished_at": finished_at,
        "duration_seconds": duration_seconds,
    }


def _load_memory_index() -> dict[str, Any] | None:
    try:
        return load_memory_index()
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _load_vector_memory() -> dict[str, Any] | None:
    try:
        return load_vector_memory()
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _has_memory_candidates(candidates: dict[str, Any]) -> bool:
    return any(
        isinstance(candidates.get(key), list) and bool(candidates[key])
        for key in ["domains", "sources", "failures"]
    )


def _empty_memory_context() -> dict[str, Any]:
    return {
        "useful_domains": [],
        "failed_urls": [],
        "_selection": "none",
    }


def _can_finish(
    steps: list[AgentStep],
    min_sources: int,
    min_source_chars: int,
) -> bool:
    return _successful_page_read_count(steps, min_source_chars) >= min_sources


def _successful_page_read_count(
    steps: list[AgentStep],
    min_source_chars: int,
) -> int:
    count = 0
    for step in steps:
        if step["action"] != "read_page":
            continue

        observation = step["observation"]
        if isinstance(observation, dict):
            char_count = int(
                observation.get("char_count")
                or len(str(observation.get("text", "")))
            )
        else:
            char_count = 0

        if (
            isinstance(observation, dict)
            and not observation.get("error")
            and char_count >= min_source_chars
        ):
            count += 1

    return count


def _emit_progress(
    on_progress: ProgressCallback | None,
    event: str,
    payload: dict[str, Any],
) -> None:
    if on_progress is not None:
        on_progress(event, payload)


def parse_agent_action(raw_response: str) -> AgentAction:
    data = _parse_json_object(raw_response)

    if not isinstance(data, dict):
        raise ValueError("Agent action must be a JSON object.")

    thought = data.get("thought")
    action = data.get("action")
    action_input = data.get("action_input")

    if action == "finish" and isinstance(action_input, str):
        action_input = {"reason": action_input}

    if not isinstance(thought, str) or not thought.strip():
        raise ValueError("Agent action requires a non-empty string thought.")

    if action not in {"search_web", "read_page", "finish"}:
        raise ValueError(f"Unknown agent action: {action}. Raw response: {raw_response}")

    if not isinstance(action_input, dict):
        raise ValueError(
            f"Agent action_input must be a JSON object. Raw response: {raw_response}"
        )

    if action == "search_web" and not (
        isinstance(action_input.get("query"), str)
        or isinstance(action_input.get("queries"), list)
    ):
        raise ValueError("search_web requires action_input.query or action_input.queries.")

    if action == "read_page" and not isinstance(action_input.get("url"), str):
        raise ValueError("read_page requires action_input.url.")

    if action == "finish" and not isinstance(action_input.get("reason"), str):
        if isinstance(action_input.get("answer"), str):
            action_input["reason"] = "The model attempted to finish with an answer."
        else:
            raise ValueError("finish requires action_input.reason.")

    return {
        "thought": thought.strip(),
        "action": action,
        "action_input": action_input,
    }


def _fallback_action(raw_response: str, steps: list[AgentStep]) -> AgentAction:
    try:
        data = _parse_json_object(raw_response)
    except ValueError:
        data = {}

    thought = data.get("thought")
    if not isinstance(thought, str) or not thought.strip():
        thought = _extract_json_string_field(raw_response, "thought")
    if not isinstance(thought, str) or not thought.strip():
        thought = "Model returned incomplete action JSON; using deterministic fallback."

    next_url = _first_unread_search_url(steps)
    if next_url:
        return {
            "thought": f"{thought.strip()} Fallback: read the first unread search result.",
            "action": "read_page",
            "action_input": {"url": next_url},
        }

    query = _extract_json_string_field(raw_response, "query") or thought.strip()
    return {
        "thought": f"{thought.strip()} Fallback: start with a web search.",
        "action": "search_web",
        "action_input": {"query": query},
    }


def _extract_json_string_field(text: str, field: str) -> str | None:
    pattern = rf'"{re.escape(field)}"\s*:\s*"((?:[^"\\]|\\.)*)"'
    match = re.search(pattern, text)
    if match is None:
        return None

    try:
        value = json.loads(f'"{match.group(1)}"')
    except json.JSONDecodeError:
        return None

    return value if isinstance(value, str) and value.strip() else None


def _first_unread_search_url(steps: list[AgentStep]) -> str | None:
    for url in _unread_urls_from_steps(steps, limit=1):
        return url

    return None


def _unread_urls_from_steps(steps: list[AgentStep], limit: int) -> list[str]:
    read_urls = {
        step["action_input"].get("url")
        for step in steps
        if step["action"] == "read_page"
    }

    urls: list[str] = []
    for step in steps:
        observation = step["observation"]
        if step["action"] not in {"plan_search", "search_web"} or not isinstance(
            observation,
            list,
        ):
            continue

        for item in observation:
            if not isinstance(item, dict):
                continue
            url = item.get("url")
            if isinstance(url, str) and url not in read_urls:
                read_urls.add(url)
                urls.append(url)
                if len(urls) >= limit:
                    return urls

    return urls


def _unread_urls_from_results(
    results: list[Any],
    steps: list[AgentStep],
    limit: int,
) -> list[str]:
    read_urls = {
        step["action_input"].get("url")
        for step in steps
        if step["action"] == "read_page"
    }

    urls: list[str] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        url = item.get("url")
        if isinstance(url, str) and url not in read_urls:
            read_urls.add(url)
            urls.append(url)
            if len(urls) >= limit:
                return urls

    return urls


def _parse_json_object(raw_response: str) -> dict[str, Any]:
    try:
        data = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        extracted = _extract_json_object(raw_response)
        if extracted is None:
            raise ValueError(f"Model did not return valid JSON: {exc}") from exc

        try:
            data = json.loads(extracted)
        except json.JSONDecodeError as extracted_exc:
            raise ValueError(
                f"Model did not return valid JSON: {extracted_exc}"
            ) from extracted_exc

    if not isinstance(data, dict):
        raise ValueError("Model response must be a JSON object.")

    return data


def _extract_json_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False

    for index in range(start, len(text)):
        char = text[index]

        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    return None
