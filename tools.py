from __future__ import annotations

from typing import NotRequired, TypedDict
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup


class SearchResult(TypedDict):
    title: str
    url: str
    snippet: str
    query: NotRequired[str]


class PageContent(TypedDict):
    url: str
    domain: str
    title: str
    text: str
    char_count: int
    source_char_count: int
    was_truncated: bool


def search_web(query: str, max_results: int = 5) -> list[SearchResult]:
    from ddgs import DDGS

    if not query.strip():
        raise ValueError("Search query cannot be empty.")

    results: list[SearchResult] = []
    with DDGS() as ddgs:
        for item in ddgs.text(query, max_results=max_results):
            url = item.get("href") or item.get("url")
            if not url:
                continue

            results.append(
                {
                    "title": item.get("title", "").strip(),
                    "url": url.strip(),
                    "snippet": item.get("body", "").strip(),
                    "query": query.strip(),
                }
            )

    return results


def search_web_many(
    queries: list[str],
    max_results_per_query: int = 5,
    max_total: int = 10,
) -> list[SearchResult]:
    if not queries:
        raise ValueError("At least one search query is required.")

    results: list[SearchResult] = []
    seen_urls: set[str] = set()

    for query in queries:
        if len(results) >= max_total:
            break

        for result in search_web(query, max_results=max_results_per_query):
            if result["url"] in seen_urls:
                continue

            results.append(result)
            seen_urls.add(result["url"])

            if len(results) >= max_total:
                break

    return results


def read_page(url: str, max_chars: int = 6000) -> PageContent:
    if not url.strip():
        raise ValueError("URL cannot be empty.")

    response = requests.get(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            )
        },
        timeout=20,
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()

    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    content = soup.find("article") or soup.find("main") or soup.body or soup
    text = _clean_text(content.get_text("\n"))
    returned_text = text[:max_chars]

    return {
        "url": response.url,
        "domain": urlparse(response.url).netloc,
        "title": title,
        "text": returned_text,
        "char_count": len(returned_text),
        "source_char_count": len(text),
        "was_truncated": len(text) > max_chars,
    }


def _clean_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    cleaned_lines = [line for line in lines if line]
    return "\n".join(cleaned_lines)
