from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_EVAL_JSON_PATH = Path("outputs/evals/summary.json")
DEFAULT_EVAL_MARKDOWN_PATH = Path("outputs/evals/summary.md")


def find_eval_trace_paths(trace_dir: str | Path = "outputs") -> list[Path]:
    path = Path(trace_dir)
    if not path.exists():
        return []

    return sorted(path.glob("*.trace.json"))


def evaluate_trace_file(path: str | Path) -> dict[str, Any]:
    trace_path = Path(path)
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    if not isinstance(trace, dict):
        raise ValueError(f"Trace must contain a JSON object: {trace_path}")

    return evaluate_trace(trace, str(trace_path))


def evaluate_trace(trace: dict[str, Any], trace_path: str = "") -> dict[str, Any]:
    steps = trace.get("steps") if isinstance(trace.get("steps"), list) else []
    metadata = trace.get("metadata") if isinstance(trace.get("metadata"), dict) else {}
    reflection = trace.get("reflection") if isinstance(trace.get("reflection"), dict) else {}

    source_observations = _source_observations(steps)
    successful_sources = [
        observation
        for observation in source_observations
        if not observation.get("error") and int(observation.get("char_count") or 0) >= 500
    ]
    source_weights = [_source_weight(observation) for observation in successful_sources]
    domains = sorted(
        {
            str(observation.get("domain"))
            for observation in successful_sources
            if observation.get("domain")
        }
    )
    errors = [
        step
        for step in steps
        if isinstance(step, dict)
        and isinstance(step.get("observation"), dict)
        and step["observation"].get("error")
    ]

    memory = metadata.get("memory") if isinstance(metadata.get("memory"), dict) else {}
    finished = bool(trace.get("answer"))
    grounded = reflection.get("grounded") is True
    complete = reflection.get("complete") is True
    score = _quality_score(
        finished=finished,
        grounded=grounded,
        complete=complete,
        successful_source_count=len(successful_sources),
        average_source_weight=_average(source_weights),
        error_count=len(errors),
    )

    return {
        "trace_path": trace_path,
        "question": trace.get("question", ""),
        "finished": finished,
        "confidence": trace.get("confidence"),
        "duration_seconds": metadata.get("duration_seconds"),
        "step_count": len(steps),
        "successful_source_count": len(successful_sources),
        "domains": domains,
        "average_source_weight": _average(source_weights),
        "error_count": len(errors),
        "reflection_action": reflection.get("recommended_action"),
        "grounded": grounded,
        "complete": complete,
        "memory_used": memory.get("used_for_search_planning") is True,
        "memory_selection": memory.get("selection", "none"),
        "search_more_used": any(
            isinstance(step, dict)
            and step.get("thought") == "Reflection found a missing angle; running bounded follow-up search."
            for step in steps
        ),
        "score": score,
    }


def build_eval_summary(trace_paths: list[str | Path]) -> dict[str, Any]:
    runs = [evaluate_trace_file(path) for path in trace_paths]
    finished_runs = [run for run in runs if run["finished"]]
    memory_runs = [run for run in runs if run["memory_used"]]

    return {
        "schema_version": 1,
        "trace_count": len(runs),
        "finished_count": len(finished_runs),
        "memory_used_count": len(memory_runs),
        "average_score": _average([run["score"] for run in runs]),
        "average_duration_seconds": _average(
            [
                run["duration_seconds"]
                for run in runs
                if isinstance(run.get("duration_seconds"), (int, float))
            ]
        ),
        "runs": runs,
        "comparisons": _compare_by_question(runs),
    }


def save_eval_summary(summary: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def save_eval_markdown(summary: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_eval_markdown(summary), encoding="utf-8")
    return path


def render_eval_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Research Agent Eval Summary",
        "",
        "## Overview",
        "",
        f"- Traces evaluated: {summary.get('trace_count', 0)}",
        f"- Finished runs: {summary.get('finished_count', 0)}",
        f"- Memory-guided runs: {summary.get('memory_used_count', 0)}",
        f"- Average score: {summary.get('average_score', 0)}",
        f"- Average duration seconds: {summary.get('average_duration_seconds', 0)}",
        "",
        "## Runs",
        "",
    ]

    runs = summary.get("runs") if isinstance(summary.get("runs"), list) else []
    if not runs:
        lines.append("No traces found.")
    for run in runs:
        question = str(run.get("question") or "Untitled question")
        lines.extend(
            [
                f"### {question}",
                "",
                f"- Trace: `{run.get('trace_path', '')}`",
                f"- Finished: {run.get('finished')}",
                f"- Score: {run.get('score')}",
                f"- Confidence: {run.get('confidence')}",
                f"- Sources: {run.get('successful_source_count')} successful reads",
                f"- Domains: {', '.join(run.get('domains', [])) or 'none'}",
                f"- Average source weight: {run.get('average_source_weight')}",
                f"- Reflection: {run.get('reflection_action')} "
                f"(grounded={run.get('grounded')}, complete={run.get('complete')})",
                f"- Memory: {run.get('memory_selection')} "
                f"(used={run.get('memory_used')})",
                f"- Duration seconds: {run.get('duration_seconds')}",
                "",
            ]
        )

    comparisons = summary.get("comparisons")
    if isinstance(comparisons, list) and comparisons:
        lines.extend(["## Memory Comparisons", ""])
        for comparison in comparisons:
            lines.extend(
                [
                    f"### {comparison.get('question', '')}",
                    "",
                    f"- Memory runs: {comparison.get('memory_run_count', 0)}",
                    f"- Non-memory runs: {comparison.get('no_memory_run_count', 0)}",
                    f"- Best memory score: {comparison.get('best_memory_score')}",
                    f"- Best non-memory score: {comparison.get('best_no_memory_score')}",
                    "",
                ]
            )

    return "\n".join(lines).rstrip() + "\n"


def _source_observations(steps: list[Any]) -> list[dict[str, Any]]:
    observations = []
    for step in steps:
        if not isinstance(step, dict) or step.get("action") != "read_page":
            continue
        observation = step.get("observation")
        if isinstance(observation, dict):
            observations.append(observation)
    return observations


def _source_weight(observation: dict[str, Any]) -> int:
    quality = observation.get("source_quality")
    if not isinstance(quality, dict):
        return 1
    weight = quality.get("weight")
    if isinstance(weight, int) and 1 <= weight <= 5:
        return weight
    return 1


def _quality_score(
    finished: bool,
    grounded: bool,
    complete: bool,
    successful_source_count: int,
    average_source_weight: float,
    error_count: int,
) -> float:
    score = 0.0
    if finished:
        score += 2.0
    if grounded:
        score += 2.0
    if complete:
        score += 2.0
    score += min(successful_source_count, 3) * 1.0
    score += min(average_source_weight, 5.0) / 5.0 * 2.0
    score -= min(error_count, 3) * 0.5
    return round(max(score, 0.0), 2)


def _compare_by_question(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_question: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        question = str(run.get("question") or "")
        if not question:
            continue
        by_question.setdefault(question, []).append(run)

    comparisons = []
    for question, question_runs in sorted(by_question.items()):
        memory_runs = [run for run in question_runs if run["memory_used"]]
        no_memory_runs = [run for run in question_runs if not run["memory_used"]]
        if not memory_runs or not no_memory_runs:
            continue
        comparisons.append(
            {
                "question": question,
                "memory_run_count": len(memory_runs),
                "no_memory_run_count": len(no_memory_runs),
                "best_memory_score": max(run["score"] for run in memory_runs),
                "best_no_memory_score": max(run["score"] for run in no_memory_runs),
            }
        )
    return comparisons


def _average(values: list[Any]) -> float:
    numbers = [float(value) for value in values if isinstance(value, (int, float))]
    if not numbers:
        return 0.0
    return round(sum(numbers) / len(numbers), 3)
