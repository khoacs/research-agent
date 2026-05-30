from __future__ import annotations

import json
from typing import Any

from llm import Message


def build_memory_selection_messages(
    question: str,
    memory_candidates: dict[str, Any],
) -> list[Message]:
    return [
        {
            "role": "system",
            "content": (
                "You select relevant long-term memory for a research agent.\n"
                "Return only valid JSON. Do not wrap the JSON in Markdown.\n\n"
                "Memory is retrieval guidance only. It can help decide where to search, "
                "but it is not evidence for the final answer.\n\n"
                "Return exactly this shape:\n"
                "{\n"
                '  "useful_domains": ["..."],\n'
                '  "failed_urls": ["..."],\n'
                '  "rationale": "brief explanation"\n'
                "}\n\n"
                "Rules:\n"
                "- Select only memory that is semantically relevant to the new research question.\n"
                "- Prefer domains that previously produced high-quality sources for similar topics.\n"
                "- Select failed URLs only if they are relevant enough that the planner might otherwise retry them.\n"
                "- Return empty lists when the memory is unrelated.\n"
                "- Do not select a domain only because it has a high weight; it must fit the question."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Research question: {question}\n\n"
                "Memory candidates:\n"
                f"{json.dumps(memory_candidates, ensure_ascii=False)[:12000]}\n\n"
                "Select the relevant memory slice."
            ),
        },
    ]


def build_search_plan_messages(
    question: str,
    memory_context: dict[str, Any] | None = None,
) -> list[Message]:
    memory_text = _format_search_memory_context(memory_context)
    return [
        {
            "role": "system",
            "content": (
                "You plan web searches for a research agent.\n"
                "Return only valid JSON. Do not wrap the JSON in Markdown.\n\n"
                "Return exactly this shape:\n"
                "{\n"
                '  "queries": ["...", "..."],\n'
                '  "preferred_source_types": ["primary", "academic"],\n'
                '  "rationale": "brief explanation"\n'
                "}\n\n"
                "Rules:\n"
                "- Generate 2 or 3 search queries.\n"
                "- Prefer targeted queries likely to find primary, academic, government, or official sources.\n"
                "- Use site: filters when an obvious authoritative domain exists.\n"
                "- If memory lists useful domains that fit the question, you may use site: filters for them.\n"
                "- If memory lists failed URLs, do not target those exact URLs.\n"
                "- Memory is only retrieval guidance. Do not treat old remembered claims as current evidence.\n"
                "- Include one broader query if the topic may need recent news or context.\n"
                "- Keep each query concise."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Research question: {question}\n\n"
                f"Memory context for retrieval only:\n{memory_text}\n\n"
                "Create a search plan."
            ),
        },
    ]


def build_agent_step_messages(
    question: str,
    steps: list[dict[str, Any]] | None = None,
    current_step: int | None = None,
    max_steps: int | None = None,
    successful_source_count: int = 0,
    min_sources: int = 2,
) -> list[Message]:
    previous_steps = _format_steps(steps or [])
    progress = _format_progress(current_step, max_steps)
    evidence_status = _format_evidence_status(successful_source_count, min_sources)
    return [
        {
            "role": "system",
            "content": (
                "You are a research agent choosing exactly one next action.\n"
                "Return only valid JSON. Do not wrap the JSON in Markdown.\n"
                "Do not call tools yourself. Only describe the next action.\n\n"
                "Available actions:\n"
                "1. search_web\n"
                "   Use when you need to find candidate web pages.\n"
                '   action_input shape: {"query": "..."}\n\n'
                "2. read_page\n"
                "   Use when you already have a specific URL to inspect.\n"
                '   action_input shape: {"url": "..."}\n\n'
                "3. finish\n"
                "   Use only when you already have enough evidence to answer.\n"
                '   action_input shape: {"reason": "..."}\n\n'
                "Guidelines:\n"
                "- Start with search_web unless useful search results are already available.\n"
                "- If a search plan already ran, choose from its search results before making another search.\n"
                "- Use read_page with URLs from previous search results.\n"
                "- Read at least two successful, relevant pages before finishing, unless the question is very simple.\n"
                "- A failed read_page observation with an error does not count as evidence.\n"
                "- When finishing, explain briefly why the evidence is enough.\n"
                "- Do not write the final answer in the finish action. A separate synthesis step will do that.\n\n"
                "Stopping rules:\n"
                "- At every step, ask whether you have enough evidence to answer.\n"
                "- Prefer finish once you have read at least two successful, relevant pages.\n"
                "- If the evidence status says the minimum source count is met, choose finish now.\n"
                "- If this is the final available step, choose finish with the evidence you have.\n"
                "- Do not keep searching unless current evidence is weak, irrelevant, or contradictory.\n\n"
                "Stay on the original research question. Do not introduce unrelated topics.\n\n"
                "Return exactly this shape:\n"
                "{\n"
                '  "thought": "brief reason for the action",\n'
                '  "action": "search_web | read_page | finish",\n'
                '  "action_input": { ... }\n'
                "}\n\n"
                "Important JSON rules:\n"
                "- Use normal double quotes for every JSON string.\n"
                "- Escape newline characters inside JSON strings as \\n.\n"
                "- Do not include literal line breaks inside JSON string values."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Research question: {question}\n\n"
                f"{progress}\n\n"
                f"{evidence_status}\n\n"
                f"Previous steps:\n{previous_steps}\n\n"
                "Choose the next action."
            ),
        },
    ]


def build_evidence_extraction_messages(
    question: str,
    page: dict[str, Any],
    max_notes: int = 5,
) -> list[Message]:
    return [
        {
            "role": "system",
            "content": (
                "You extract compact evidence notes from a web page for a research agent.\n"
                "Return only valid JSON. Do not wrap the JSON in Markdown.\n"
                "Do not answer the full research question.\n\n"
                "Return exactly this shape:\n"
                "{\n"
                '  "relevance": "high | medium | low",\n'
                '  "summary": "one sentence about what this page contributes",\n'
                '  "notes": [\n'
                "    {\n"
                '      "claim": "a concise factual point relevant to the question",\n'
                '      "supporting_text": "a short phrase from the page, 20 words or fewer"\n'
                "    }\n"
                "  ]\n"
                "}\n\n"
                "Rules:\n"
                f"- Extract at most {max_notes} notes.\n"
                "- Only include notes that help answer the research question.\n"
                "- Use low relevance and an empty notes list if the page is not useful.\n"
                "- Keep supporting_text short. Do not quote long passages.\n"
                "- Use normal double quotes for every JSON string."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Research question: {question}\n\n"
                f"Page title: {page.get('title') or 'Untitled'}\n"
                f"Page URL: {page.get('url')}\n"
                f"Page domain: {page.get('domain')}\n\n"
                f"Page text:\n{str(page.get('text', ''))[:6000]}\n\n"
                "Extract evidence notes."
            ),
        },
    ]


def build_source_quality_messages(
    question: str,
    page: dict[str, Any],
) -> list[Message]:
    evidence = page.get("evidence")
    evidence_text = json.dumps(evidence, ensure_ascii=False) if evidence else "None."
    return [
        {
            "role": "system",
            "content": (
                "You score the quality of a web source for a research agent.\n"
                "Return only valid JSON. Do not wrap the JSON in Markdown.\n\n"
                "Return exactly this shape:\n"
                "{\n"
                '  "source_type": "primary | academic | government | news | company | blog | reference | unknown",\n'
                '  "credibility": "high | medium | low",\n'
                '  "relevance": "high | medium | low",\n'
                '  "weight": 1,\n'
                '  "reason": "brief explanation"\n'
                "}\n\n"
                "Rules:\n"
                "- Primary means the organization, author, or project directly responsible for the information.\n"
                "- Academic means scholarly or research-institution material.\n"
                "- Government means official public agency material.\n"
                "- Blog means informal commentary or secondary write-up.\n"
                "- weight must be an integer from 1 to 5, where 5 means strongest source.\n"
                "- Prefer high credibility for official, primary, academic, or government sources.\n"
                "- Lower the score if the page is unfocused, promotional, anonymous, or only loosely relevant."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Research question: {question}\n\n"
                f"Page title: {page.get('title') or 'Untitled'}\n"
                f"Page URL: {page.get('url')}\n"
                f"Page domain: {page.get('domain')}\n"
                f"Extracted chars: {page.get('char_count')}\n"
                f"Was truncated: {page.get('was_truncated')}\n\n"
                f"Evidence extracted from page:\n{evidence_text}\n\n"
                f"Page text excerpt:\n{str(page.get('text', ''))[:2000]}\n\n"
                "Score this source."
            ),
        },
    ]


def build_answer_synthesis_messages(
    question: str,
    steps: list[dict[str, Any]],
) -> list[Message]:
    evidence = _format_evidence_for_synthesis(steps)
    sources = _format_sources_for_synthesis(steps)
    return [
        {
            "role": "system",
            "content": (
                "You synthesize the final answer for a research agent.\n"
                "Use the provided evidence notes and sources.\n"
                "Return only valid JSON. Do not wrap the JSON in Markdown.\n\n"
                "Return exactly this shape:\n"
                "{\n"
                '  "answer": "clear answer in Markdown",\n'
                '  "confidence": "high | medium | low",\n'
                '  "limitations": "brief note about evidence limits, or empty string"\n'
                "}\n\n"
                "Rules:\n"
                "- Ground the answer in the evidence notes.\n"
                "- Give more weight to higher-quality, more relevant sources.\n"
                "- Do not cite sources that were not provided.\n"
                "- If evidence is thin or from only one source, say so in limitations.\n"
                "- Keep the answer concise but useful."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Research question: {question}\n\n"
                f"Sources:\n{sources}\n\n"
                f"Evidence notes:\n{evidence}\n\n"
                "Write the final answer."
            ),
        },
    ]


def build_answer_reflection_messages(
    question: str,
    answer: str,
    confidence: str,
    limitations: str,
    steps: list[dict[str, Any]],
) -> list[Message]:
    evidence = _format_evidence_for_synthesis(steps)
    sources = _format_sources_for_synthesis(steps)
    return [
        {
            "role": "system",
            "content": (
                "You verify and reflect on a research agent's final answer.\n"
                "Use the evidence notes to check grounding. Use your broader knowledge "
                "only to identify likely missing angles or follow-up searches, not to add "
                "unsupported facts to the answer.\n"
                "Return only valid JSON. Do not wrap the JSON in Markdown.\n\n"
                "Return exactly this shape:\n"
                "{\n"
                '  "grounded": true,\n'
                '  "complete": true,\n'
                '  "recommended_action": "accept | revise | search_more",\n'
                '  "issues": [\n'
                "    {\n"
                '      "claim": "...",\n'
                '      "problem": "..."\n'
                "    }\n"
                "  ],\n"
                '  "missing_angles": ["..."],\n'
                '  "follow_up_queries": ["..."],\n'
                '  "revision_advice": "brief advice, or empty string"\n'
                "}\n\n"
                "Rules:\n"
                "- grounded is true only if the answer's substantive claims are supported by evidence notes.\n"
                "- complete is true if the answer covers the main likely angles of the question.\n"
                "- Use revise when the answer can be fixed with the current evidence.\n"
                "- Use search_more when an important missing angle needs new evidence.\n"
                "- Use accept when grounded and sufficiently complete.\n"
                "- Do not demand exhaustive coverage for broad questions; focus on important gaps."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Research question: {question}\n\n"
                f"Current answer:\n{answer}\n\n"
                f"Confidence: {confidence}\n"
                f"Limitations: {limitations}\n\n"
                f"Sources:\n{sources}\n\n"
                f"Evidence notes:\n{evidence}\n\n"
                "Verify grounding and reflect on completeness."
            ),
        },
    ]


def build_answer_revision_messages(
    question: str,
    answer: str,
    reflection: dict[str, Any],
    steps: list[dict[str, Any]],
) -> list[Message]:
    evidence = _format_evidence_for_synthesis(steps)
    sources = _format_sources_for_synthesis(steps)
    return [
        {
            "role": "system",
            "content": (
                "You revise a research agent's final answer using verifier feedback.\n"
                "Return only valid JSON. Do not wrap the JSON in Markdown.\n\n"
                "Return exactly this shape:\n"
                "{\n"
                '  "answer": "revised clear answer in Markdown",\n'
                '  "confidence": "high | medium | low",\n'
                '  "limitations": "brief note about evidence limits, or empty string"\n'
                "}\n\n"
                "Rules:\n"
                "- Use only the provided evidence notes and sources.\n"
                "- Fix unsupported or overstated claims identified by the verifier.\n"
                "- If the verifier asked for new search, do not invent missing evidence; note the limitation."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Research question: {question}\n\n"
                f"Original answer:\n{answer}\n\n"
                f"Verifier feedback:\n{json.dumps(reflection, ensure_ascii=False)}\n\n"
                f"Sources:\n{sources}\n\n"
                f"Evidence notes:\n{evidence}\n\n"
                "Revise the answer."
            ),
        },
    ]


def _format_steps(steps: list[dict[str, Any]]) -> str:
    if not steps:
        return "None yet."

    formatted = []
    for index, step in enumerate(steps, start=1):
        formatted.append(
            "\n".join(
                [
                    f"{index}. Thought: {step['thought']}",
                    f"   Action: {step['action']}",
                    f"   Input: {json.dumps(step['action_input'], ensure_ascii=False)}",
                    f"   Observation: {json.dumps(step['observation'], ensure_ascii=False)[:3000]}",
                ]
            )
        )

    return "\n\n".join(formatted)


def _format_progress(current_step: int | None, max_steps: int | None) -> str:
    if current_step is None or max_steps is None:
        return "Step progress: unknown."

    remaining = max(max_steps - current_step, 0)
    return (
        f"Current step: {current_step} of {max_steps}\n"
        f"Remaining steps after this action: {remaining}"
    )


def _format_evidence_status(successful_source_count: int, min_sources: int) -> str:
    if successful_source_count >= min_sources:
        return (
            f"Successful source reads: {successful_source_count} of {min_sources} required.\n"
            "Minimum source count is met. You should choose finish now."
        )

    return (
        f"Successful source reads: {successful_source_count} of {min_sources} required.\n"
        "Minimum source count is not met yet."
    )


def _format_search_memory_context(memory_context: dict[str, Any] | None) -> str:
    if not memory_context:
        return "No memory context available."

    lines: list[str] = []
    useful_domains = memory_context.get("useful_domains")
    if isinstance(useful_domains, list) and useful_domains:
        lines.append("Useful domains from previous runs:")
        for domain in useful_domains[:5]:
            if not isinstance(domain, dict):
                continue
            name = domain.get("domain")
            if not isinstance(name, str) or not name:
                continue
            source_types = domain.get("source_types")
            source_type_text = ""
            if isinstance(source_types, list) and source_types:
                source_type_text = f", types={', '.join(str(item) for item in source_types[:3])}"
            lines.append(
                "- "
                f"{name}: "
                f"{domain.get('successful_reads', 0)} successful reads, "
                f"{domain.get('failed_reads', 0)} failures, "
                f"avg weight {domain.get('average_weight', 0)}"
                f"{source_type_text}"
            )

    failed_urls = memory_context.get("failed_urls")
    if isinstance(failed_urls, list) and failed_urls:
        if lines:
            lines.append("")
        lines.append("Previously failed URLs to avoid when possible:")
        for failure in failed_urls[:5]:
            if not isinstance(failure, dict):
                continue
            url = failure.get("url")
            if isinstance(url, str) and url:
                lines.append(f"- {url} ({failure.get('error_type', 'error')})")

    if not lines:
        return "No useful retrieval memory available."

    return "\n".join(lines)


def _format_evidence_for_synthesis(steps: list[dict[str, Any]]) -> str:
    blocks: list[str] = []

    for step in steps:
        observation = step.get("observation")
        if not isinstance(observation, dict):
            continue

        evidence = observation.get("evidence")
        if not isinstance(evidence, dict):
            continue

        notes = evidence.get("notes")
        if not isinstance(notes, list) or not notes:
            continue

        title = observation.get("title") or observation.get("url") or "Untitled"
        domain = observation.get("domain") or ""
        source_quality = observation.get("source_quality")
        header = f"Source: {title}"
        if domain:
            header += f" ({domain})"

        lines = [header]
        if isinstance(source_quality, dict):
            lines.append(
                "Quality: "
                f"type={source_quality.get('source_type', 'unknown')}, "
                f"credibility={source_quality.get('credibility', 'unknown')}, "
                f"relevance={source_quality.get('relevance', 'unknown')}, "
                f"weight={source_quality.get('weight', 'unknown')}"
            )
        summary = evidence.get("summary")
        if isinstance(summary, str) and summary.strip():
            lines.append(f"Summary: {summary.strip()}")

        for note in notes:
            if not isinstance(note, dict):
                continue
            claim = note.get("claim")
            support = note.get("supporting_text")
            if isinstance(claim, str) and claim.strip():
                line = f"- {claim.strip()}"
                if isinstance(support, str) and support.strip():
                    line += f" Evidence phrase: {support.strip()}"
                lines.append(line)

        blocks.append("\n".join(lines))

    if not blocks:
        return "No evidence notes were extracted."

    return "\n\n".join(blocks)


def _format_sources_for_synthesis(steps: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    seen_urls: set[str] = set()

    for step in steps:
        observation = step.get("observation")
        if not isinstance(observation, dict) or not observation.get("url"):
            continue

        url = str(observation["url"])
        if url in seen_urls:
            continue

        title = observation.get("title") or url
        domain = observation.get("domain") or ""
        if domain:
            lines.append(f"- {title} ({domain}): {url}")
        else:
            lines.append(f"- {title}: {url}")
        seen_urls.add(url)

    if not lines:
        return "No successful sources were read."

    return "\n".join(lines)
