from __future__ import annotations

import json
from pathlib import Path

from agent import AgentResult


def save_markdown_report(result: AgentResult, output_path: str | Path) -> Path:
    if not result["answer"]:
        raise ValueError("Cannot save a report before the agent produces an answer.")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown_report(result), encoding="utf-8")
    return path


def save_json_trace(result: AgentResult, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def trace_path_for_report(output_path: str | Path) -> Path:
    path = Path(output_path)
    return path.with_suffix(f"{path.suffix}.trace.json")


def render_markdown_report(result: AgentResult) -> str:
    sources = _collect_sources(result)
    source_lines = "\n".join(
        f"- [{title}]({url}) ({domain})" if domain else f"- [{title}]({url})"
        for title, url, domain in sources
    )
    if not source_lines:
        source_lines = "- No sources recorded."

    evidence_lines = _render_evidence_notes(result)
    quality_lines = _render_source_quality(result)
    reflection_lines = _render_reflection(result)

    return (
        "# Research Report\n\n"
        "## Question\n\n"
        f"{result['question']}\n\n"
        "## Run Metadata\n\n"
        f"{_render_metadata(result)}\n\n"
        "## Answer\n\n"
        f"{result['answer']}\n\n"
        "## Confidence\n\n"
        f"{result.get('confidence') or 'unknown'}\n\n"
        "## Limitations\n\n"
        f"{result.get('limitations') or 'No limitations recorded.'}\n\n"
        "## Reflection\n\n"
        f"{reflection_lines}\n\n"
        "## Source Quality\n\n"
        f"{quality_lines}\n\n"
        "## Evidence Notes\n\n"
        f"{evidence_lines}\n\n"
        "## Sources\n\n"
        f"{source_lines}\n"
    )


def _collect_sources(result: AgentResult) -> list[tuple[str, str, str]]:
    sources: list[tuple[str, str, str]] = []
    seen_urls: set[str] = set()

    for step in result["steps"]:
        observation = step["observation"]
        if isinstance(observation, dict) and observation.get("url"):
            url = str(observation["url"])
            title = str(observation.get("title") or url)
            domain = str(observation.get("domain") or "")
            if url not in seen_urls:
                sources.append((title, url, domain))
                seen_urls.add(url)

    return sources


def _render_metadata(result: AgentResult) -> str:
    metadata = result.get("metadata")
    if not isinstance(metadata, dict):
        return "- No metadata recorded."

    lines = []
    for key in [
        "provider",
        "model",
        "max_steps",
        "min_sources",
        "min_source_chars",
        "started_at",
        "finished_at",
        "duration_seconds",
    ]:
        value = metadata.get(key)
        if value is not None:
            lines.append(f"- {key}: {value}")

    if not lines:
        return "- No metadata recorded."

    return "\n".join(lines)


def _render_evidence_notes(result: AgentResult) -> str:
    lines: list[str] = []

    for step in result["steps"]:
        observation = step["observation"]
        if not isinstance(observation, dict):
            continue

        evidence = observation.get("evidence")
        if not isinstance(evidence, dict):
            continue

        notes = evidence.get("notes")
        if not isinstance(notes, list) or not notes:
            continue

        title = str(observation.get("title") or observation.get("url") or "Untitled")
        domain = str(observation.get("domain") or "")
        heading = f"### {title}"
        if domain:
            heading = f"{heading} ({domain})"

        lines.append(heading)
        summary = evidence.get("summary")
        if isinstance(summary, str) and summary.strip():
            lines.append(f"\n{summary.strip()}\n")

        for note in notes:
            if not isinstance(note, dict):
                continue
            claim = note.get("claim")
            if isinstance(claim, str) and claim.strip():
                lines.append(f"- {claim.strip()}")

        lines.append("")

    if not lines:
        return "No evidence notes recorded."

    return "\n".join(lines).strip()


def _render_source_quality(result: AgentResult) -> str:
    lines: list[str] = []

    for step in result["steps"]:
        observation = step["observation"]
        if not isinstance(observation, dict):
            continue

        quality = observation.get("source_quality")
        if not isinstance(quality, dict):
            continue

        title = str(observation.get("title") or observation.get("url") or "Untitled")
        domain = str(observation.get("domain") or "")
        heading = f"### {title}"
        if domain:
            heading = f"{heading} ({domain})"

        lines.append(heading)
        lines.append("")
        lines.append(f"- Type: {quality.get('source_type', 'unknown')}")
        lines.append(f"- Credibility: {quality.get('credibility', 'unknown')}")
        lines.append(f"- Relevance: {quality.get('relevance', 'unknown')}")
        lines.append(f"- Weight: {quality.get('weight', 'unknown')}/5")
        reason = quality.get("reason")
        if isinstance(reason, str) and reason.strip():
            lines.append(f"- Reason: {reason.strip()}")
        lines.append("")

    if not lines:
        return "No source quality scores recorded."

    return "\n".join(lines).strip()


def _render_reflection(result: AgentResult) -> str:
    reflection = result.get("reflection")
    if not isinstance(reflection, dict):
        return "No reflection recorded."

    lines = [
        f"- Grounded: {reflection.get('grounded')}",
        f"- Complete: {reflection.get('complete')}",
        f"- Recommended action: {reflection.get('recommended_action')}",
    ]

    revision_advice = reflection.get("revision_advice")
    if isinstance(revision_advice, str) and revision_advice.strip():
        lines.append(f"- Revision advice: {revision_advice.strip()}")

    issues = reflection.get("issues")
    if isinstance(issues, list) and issues:
        lines.append("")
        lines.append("Issues:")
        for issue in issues:
            if not isinstance(issue, dict):
                continue
            claim = issue.get("claim")
            problem = issue.get("problem")
            if isinstance(claim, str) and isinstance(problem, str):
                lines.append(f"- {claim}: {problem}")

    missing_angles = reflection.get("missing_angles")
    if isinstance(missing_angles, list) and missing_angles:
        lines.append("")
        lines.append("Missing angles:")
        for angle in missing_angles:
            if isinstance(angle, str) and angle.strip():
                lines.append(f"- {angle.strip()}")

    follow_up_queries = reflection.get("follow_up_queries")
    if isinstance(follow_up_queries, list) and follow_up_queries:
        lines.append("")
        lines.append("Follow-up queries:")
        for query in follow_up_queries:
            if isinstance(query, str) and query.strip():
                lines.append(f"- {query.strip()}")

    return "\n".join(lines)
