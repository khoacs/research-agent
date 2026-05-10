from __future__ import annotations

from typing import TypedDict

import requests
from bs4 import BeautifulSoup
from ddgs import DDGS


class SearchResult(TypedDict):
    title: str
    url: str
    snippet: str


class PageContent(TypedDict):
    url: str
    title: str
    text: str


def search_web(query: str, max_results: int = 5) -> list[SearchResult]:
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
                }
            )

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

    return {
        "url": response.url,
        "title": title,
        "text": text[:max_chars],
    }


def _clean_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    cleaned_lines = [line for line in lines if line]
    return "\n".join(cleaned_lines)
