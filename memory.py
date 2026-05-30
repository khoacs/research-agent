from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


DEFAULT_TRACE_DIR = Path("outputs")
DEFAULT_MEMORY_PATH = Path("memory/index.json")
DEFAULT_VECTOR_MEMORY_PATH = Path("memory/vectors.json")


def find_trace_paths(trace_dir: str | Path = DEFAULT_TRACE_DIR) -> list[Path]:
    path = Path(trace_dir)
    if not path.exists():
        return []

    return sorted(path.glob("*.trace.json"))


def build_memory_index(trace_paths: list[str | Path]) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    sources_by_url: dict[str, dict[str, Any]] = {}
    domains: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []

    for trace_path in trace_paths:
        path = Path(trace_path)
        trace = _load_trace(path)
        question = str(trace.get("question") or "")
        metadata = trace.get("metadata") if isinstance(trace.get("metadata"), dict) else {}

        runs.append(
            {
                "trace_path": str(path),
                "question": question,
                "answer_present": bool(trace.get("answer")),
                "confidence": trace.get("confidence"),
                "provider": metadata.get("provider"),
                "model": metadata.get("model"),
                "started_at": metadata.get("started_at"),
                "duration_seconds": metadata.get("duration_seconds"),
            }
        )

        for step in _steps(trace):
            observation = step.get("observation")
            if not isinstance(observation, dict):
                continue

            if observation.get("error"):
                failure = _failure_record(path, question, step, observation)
                failures.append(failure)
                if failure["domain"]:
                    _domain_record(domains, failure["domain"])["failed_reads"] += 1
                continue

            if step.get("action") != "read_page" or not observation.get("url"):
                continue

            source = _source_record(path, question, observation)
            existing = sources_by_url.get(source["url"])
            if existing is None:
                sources_by_url[source["url"]] = source
            else:
                _merge_source(existing, source)

            domain = source["domain"]
            if domain:
                _merge_domain(_domain_record(domains, domain), source)

    source_records = sorted(
        sources_by_url.values(),
        key=lambda item: (-int(item["best_weight"]), item["domain"], item["title"]),
    )
    domain_records = sorted(
        (_public_domain_record(domain) for domain in domains.values()),
        key=lambda item: (
            -int(item["successful_reads"]),
            -float(item["average_weight"]),
            item["domain"],
        ),
    )

    return {
        "schema_version": 1,
        "trace_count": len(trace_paths),
        "successful_run_count": sum(1 for run in runs if run["answer_present"]),
        "runs": runs,
        "sources": source_records,
        "domains": domain_records,
        "failures": failures,
    }


def save_memory_index(
    memory: dict[str, Any],
    output_path: str | Path = DEFAULT_MEMORY_PATH,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(memory, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def build_memory_items(memory: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []

    for domain in memory.get("domains", []):
        if not isinstance(domain, dict) or not domain.get("domain"):
            continue
        domain_name = str(domain["domain"])
        items.append(
            {
                "id": f"domain:{domain_name}",
                "kind": "domain",
                "text": (
                    f"Domain {domain_name}. "
                    f"Successful reads: {domain.get('successful_reads', 0)}. "
                    f"Failed reads: {domain.get('failed_reads', 0)}. "
                    f"Average source quality weight: {domain.get('average_weight', 0)}. "
                    f"Source types: {_join_list(domain.get('source_types'))}."
                ),
                "metadata": {
                    "domain": domain_name,
                    "successful_reads": domain.get("successful_reads", 0),
                    "failed_reads": domain.get("failed_reads", 0),
                    "average_weight": domain.get("average_weight", 0),
                    "source_types": domain.get("source_types", []),
                },
            }
        )

    for source in memory.get("sources", []):
        if not isinstance(source, dict) or not source.get("url"):
            continue
        url = str(source["url"])
        items.append(
            {
                "id": f"source:{url}",
                "kind": "source",
                "text": (
                    f"Source title: {source.get('title', '')}. "
                    f"Domain: {source.get('domain', '')}. "
                    f"Type: {source.get('source_type', 'unknown')}. "
                    f"Credibility: {source.get('credibility', 'unknown')}. "
                    f"Relevance: {source.get('relevance', 'unknown')}. "
                    f"Best source quality weight: {source.get('best_weight', 1)}. "
                    f"Past questions: {_join_list(source.get('questions'))}. "
                    f"Claims: {_join_list(source.get('claims'))}."
                ),
                "metadata": {
                    "url": url,
                    "domain": source.get("domain", ""),
                    "title": source.get("title", ""),
                    "best_weight": source.get("best_weight", 1),
                    "source_type": source.get("source_type", "unknown"),
                    "credibility": source.get("credibility", "unknown"),
                    "relevance": source.get("relevance", "unknown"),
                    "questions": source.get("questions", []),
                    "claims": source.get("claims", []),
                },
            }
        )

    for failure in memory.get("failures", []):
        if not isinstance(failure, dict) or not failure.get("url"):
            continue
        url = str(failure["url"])
        items.append(
            {
                "id": f"failure:{url}",
                "kind": "failure",
                "text": (
                    f"Failed URL: {url}. "
                    f"Domain: {failure.get('domain', '')}. "
                    f"Error type: {failure.get('error_type', '')}. "
                    f"Question: {failure.get('question', '')}."
                ),
                "metadata": {
                    "url": url,
                    "domain": failure.get("domain", ""),
                    "error_type": failure.get("error_type", ""),
                    "question": failure.get("question", ""),
                },
            }
        )

    return items


def build_vector_memory(
    memory: dict[str, Any],
    embed_fn,
    embedding_model: str,
) -> dict[str, Any]:
    vector_items = []
    for item in build_memory_items(memory):
        vector_items.append(
            {
                **item,
                "embedding": embed_fn(item["text"]),
            }
        )

    return {
        "schema_version": 1,
        "embedding_model": embedding_model,
        "items": vector_items,
    }


def save_vector_memory(
    vector_memory: dict[str, Any],
    output_path: str | Path = DEFAULT_VECTOR_MEMORY_PATH,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(vector_memory, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_vector_memory(
    path: str | Path = DEFAULT_VECTOR_MEMORY_PATH,
) -> dict[str, Any] | None:
    vector_path = Path(path)
    if not vector_path.exists():
        return None

    data = json.loads(vector_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Vector memory must contain a JSON object: {vector_path}")
    return data


def retrieve_vector_memory(
    vector_memory: dict[str, Any] | None,
    query_embedding: list[float],
    top_k: int = 5,
    min_score: float = 0.25,
) -> list[dict[str, Any]]:
    if not vector_memory:
        return []

    scored_items = []
    for item in vector_memory.get("items", []):
        if not isinstance(item, dict):
            continue
        embedding = item.get("embedding")
        if not isinstance(embedding, list):
            continue
        score = cosine_similarity(query_embedding, [float(value) for value in embedding])
        if score < min_score:
            continue
        scored_items.append(
            {
                **{key: value for key, value in item.items() if key != "embedding"},
                "score": round(score, 4),
            }
        )

    return sorted(scored_items, key=lambda item: item["score"], reverse=True)[:top_k]


def build_search_memory_context_from_vector_matches(
    matches: list[dict[str, Any]],
    max_domains: int = 5,
    max_failed_urls: int = 5,
) -> dict[str, Any]:
    useful_domains = []
    failed_urls = []
    seen_domains: set[str] = set()
    seen_failed_urls: set[str] = set()

    for match in matches:
        kind = match.get("kind")
        metadata = match.get("metadata") if isinstance(match.get("metadata"), dict) else {}
        if kind == "domain":
            domain = metadata.get("domain")
            if isinstance(domain, str) and domain and domain not in seen_domains:
                useful_domains.append(
                    {
                        "domain": domain,
                        "successful_reads": metadata.get("successful_reads", 0),
                        "failed_reads": metadata.get("failed_reads", 0),
                        "average_weight": metadata.get("average_weight", 0),
                        "source_types": metadata.get("source_types", []),
                        "score": match.get("score"),
                    }
                )
                seen_domains.add(domain)
        elif kind == "source":
            domain = metadata.get("domain")
            if isinstance(domain, str) and domain and domain not in seen_domains:
                useful_domains.append(
                    {
                        "domain": domain,
                        "successful_reads": 1,
                        "failed_reads": 0,
                        "average_weight": metadata.get("best_weight", 0),
                        "source_types": [metadata.get("source_type", "unknown")],
                        "score": match.get("score"),
                    }
                )
                seen_domains.add(domain)
        elif kind == "failure":
            url = metadata.get("url")
            if isinstance(url, str) and url and url not in seen_failed_urls:
                failed_urls.append(
                    {
                        "url": url,
                        "domain": metadata.get("domain", ""),
                        "error_type": metadata.get("error_type", ""),
                        "score": match.get("score"),
                    }
                )
                seen_failed_urls.add(url)

        if len(useful_domains) >= max_domains and len(failed_urls) >= max_failed_urls:
            break

    return {
        "useful_domains": useful_domains[:max_domains],
        "failed_urls": failed_urls[:max_failed_urls],
    }


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0

    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0

    return dot / (left_norm * right_norm)


def load_memory_index(path: str | Path = DEFAULT_MEMORY_PATH) -> dict[str, Any] | None:
    memory_path = Path(path)
    if not memory_path.exists():
        return None

    data = json.loads(memory_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Memory index must contain a JSON object: {memory_path}")
    return data


def build_search_memory_context(
    memory: dict[str, Any] | None,
    max_domains: int = 5,
    max_failed_urls: int = 5,
) -> dict[str, Any]:
    if not memory:
        return {
            "useful_domains": [],
            "failed_urls": [],
        }

    domains = memory.get("domains")
    failures = memory.get("failures")

    useful_domains = []
    if isinstance(domains, list):
        for domain in domains:
            if not isinstance(domain, dict):
                continue
            if int(domain.get("successful_reads") or 0) <= 0:
                continue
            useful_domains.append(
                {
                    "domain": domain.get("domain", ""),
                    "successful_reads": domain.get("successful_reads", 0),
                    "failed_reads": domain.get("failed_reads", 0),
                    "average_weight": domain.get("average_weight", 0),
                    "source_types": domain.get("source_types", []),
                }
            )
            if len(useful_domains) >= max_domains:
                break

    failed_urls = []
    if isinstance(failures, list):
        for failure in failures:
            if not isinstance(failure, dict):
                continue
            url = failure.get("url")
            if not isinstance(url, str) or not url:
                continue
            failed_urls.append(
                {
                    "url": url,
                    "domain": failure.get("domain", ""),
                    "error_type": failure.get("error_type", ""),
                }
            )
            if len(failed_urls) >= max_failed_urls:
                break

    return {
        "useful_domains": useful_domains,
        "failed_urls": failed_urls,
    }


def build_search_memory_candidates(
    memory: dict[str, Any] | None,
    max_domains: int = 20,
    max_sources: int = 20,
    max_failed_urls: int = 20,
) -> dict[str, Any]:
    if not memory:
        return {
            "domains": [],
            "sources": [],
            "failures": [],
        }

    domains = memory.get("domains") if isinstance(memory.get("domains"), list) else []
    sources = memory.get("sources") if isinstance(memory.get("sources"), list) else []
    failures = memory.get("failures") if isinstance(memory.get("failures"), list) else []

    return {
        "domains": [
            _domain_candidate(domain)
            for domain in domains[:max_domains]
            if isinstance(domain, dict)
        ],
        "sources": [
            _source_candidate(source)
            for source in sources[:max_sources]
            if isinstance(source, dict)
        ],
        "failures": [
            _failure_candidate(failure)
            for failure in failures[:max_failed_urls]
            if isinstance(failure, dict)
        ],
    }


def render_memory_summary(memory: dict[str, Any]) -> str:
    trace_count = memory.get("trace_count", 0)
    successful_runs = memory.get("successful_run_count", 0)
    sources = memory.get("sources") if isinstance(memory.get("sources"), list) else []
    domains = memory.get("domains") if isinstance(memory.get("domains"), list) else []
    failures = memory.get("failures") if isinstance(memory.get("failures"), list) else []

    lines = [
        f"Traces indexed: {trace_count}",
        f"Successful runs: {successful_runs}",
        f"Unique sources read: {len(sources)}",
        f"Tool failures remembered: {len(failures)}",
    ]

    if domains:
        lines.append("")
        lines.append("Top domains:")
        for domain in domains[:5]:
            lines.append(
                "- "
                f"{domain['domain']}: "
                f"{domain['successful_reads']} reads, "
                f"{domain['failed_reads']} failures, "
                f"avg weight {domain['average_weight']}"
            )

    if sources:
        lines.append("")
        lines.append("Top sources:")
        for source in sources[:5]:
            lines.append(
                "- "
                f"{source['title']} ({source['domain']}), "
                f"weight {source['best_weight']}: {source['url']}"
            )

    return "\n".join(lines)


def _load_trace(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Trace must contain a JSON object: {path}")
    return data


def _steps(trace: dict[str, Any]) -> list[dict[str, Any]]:
    steps = trace.get("steps")
    if not isinstance(steps, list):
        return []

    return [step for step in steps if isinstance(step, dict)]


def _source_record(
    trace_path: Path,
    question: str,
    observation: dict[str, Any],
) -> dict[str, Any]:
    quality = (
        observation.get("source_quality")
        if isinstance(observation.get("source_quality"), dict)
        else {}
    )
    evidence = (
        observation.get("evidence")
        if isinstance(observation.get("evidence"), dict)
        else {}
    )

    return {
        "url": str(observation["url"]),
        "domain": str(observation.get("domain") or ""),
        "title": str(observation.get("title") or observation["url"]),
        "read_count": 1,
        "trace_paths": [str(trace_path)],
        "questions": [question] if question else [],
        "best_weight": _weight(quality),
        "source_type": quality.get("source_type", "unknown"),
        "credibility": quality.get("credibility", "unknown"),
        "relevance": quality.get("relevance", "unknown"),
        "claims": _claims(evidence),
    }


def _merge_source(existing: dict[str, Any], new: dict[str, Any]) -> None:
    existing["read_count"] += 1
    _append_unique(existing["trace_paths"], new["trace_paths"])
    _append_unique(existing["questions"], new["questions"])
    _append_unique(existing["claims"], new["claims"], limit=8)

    if int(new["best_weight"]) > int(existing["best_weight"]):
        existing["best_weight"] = new["best_weight"]
        existing["source_type"] = new["source_type"]
        existing["credibility"] = new["credibility"]
        existing["relevance"] = new["relevance"]


def _failure_record(
    trace_path: Path,
    question: str,
    step: dict[str, Any],
    observation: dict[str, Any],
) -> dict[str, Any]:
    action_input = step.get("action_input")
    url = ""
    if isinstance(action_input, dict):
        url = str(action_input.get("url") or "")

    return {
        "trace_path": str(trace_path),
        "question": question,
        "action": step.get("action"),
        "url": url,
        "domain": _domain_from_url(url),
        "error_type": observation.get("error_type"),
        "error": observation.get("error"),
    }


def _domain_record(domains: dict[str, dict[str, Any]], domain: str) -> dict[str, Any]:
    if domain not in domains:
        domains[domain] = {
            "domain": domain,
            "successful_reads": 0,
            "failed_reads": 0,
            "_weight_total": 0,
            "average_weight": 0.0,
            "source_types": [],
        }

    return domains[domain]


def _merge_domain(domain: dict[str, Any], source: dict[str, Any]) -> None:
    domain["successful_reads"] += 1
    domain["_weight_total"] += int(source["best_weight"])
    domain["average_weight"] = round(
        domain["_weight_total"] / domain["successful_reads"],
        2,
    )
    source_type = source.get("source_type")
    if isinstance(source_type, str) and source_type not in domain["source_types"]:
        domain["source_types"].append(source_type)


def _public_domain_record(domain: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in domain.items()
        if not key.startswith("_")
    }


def _domain_candidate(domain: dict[str, Any]) -> dict[str, Any]:
    return {
        "domain": domain.get("domain", ""),
        "successful_reads": domain.get("successful_reads", 0),
        "failed_reads": domain.get("failed_reads", 0),
        "average_weight": domain.get("average_weight", 0),
        "source_types": domain.get("source_types", []),
    }


def _source_candidate(source: dict[str, Any]) -> dict[str, Any]:
    return {
        "url": source.get("url", ""),
        "domain": source.get("domain", ""),
        "title": source.get("title", ""),
        "best_weight": source.get("best_weight", 1),
        "source_type": source.get("source_type", "unknown"),
        "questions": source.get("questions", [])[:3]
        if isinstance(source.get("questions"), list)
        else [],
        "claims": source.get("claims", [])[:3]
        if isinstance(source.get("claims"), list)
        else [],
    }


def _failure_candidate(failure: dict[str, Any]) -> dict[str, Any]:
    return {
        "url": failure.get("url", ""),
        "domain": failure.get("domain", ""),
        "error_type": failure.get("error_type", ""),
        "question": failure.get("question", ""),
    }


def _join_list(value: Any) -> str:
    if not isinstance(value, list):
        return ""
    return "; ".join(str(item) for item in value if str(item).strip())


def _claims(evidence: dict[str, Any]) -> list[str]:
    notes = evidence.get("notes")
    if not isinstance(notes, list):
        return []

    claims: list[str] = []
    for note in notes:
        if not isinstance(note, dict):
            continue
        claim = note.get("claim")
        if isinstance(claim, str) and claim.strip():
            claims.append(claim.strip())

    return claims[:5]


def _weight(quality: dict[str, Any]) -> int:
    weight = quality.get("weight")
    if isinstance(weight, int) and 1 <= weight <= 5:
        return weight
    return 1


def _domain_from_url(url: str) -> str:
    if "://" not in url:
        return ""
    return url.split("://", 1)[1].split("/", 1)[0]


def _append_unique(target: list[Any], values: list[Any], limit: int | None = None) -> None:
    for value in values:
        if value in target:
            continue
        if limit is not None and len(target) >= limit:
            return
        target.append(value)
