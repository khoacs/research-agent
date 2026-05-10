from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Literal, TypedDict

from config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_MODEL,
)


Provider = Literal["openrouter", "ollama"]


class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


def call_llm(messages: list[Message], provider: Provider, model: str | None = None) -> str:
    if provider == "openrouter":
        return call_openrouter(messages, model=model)
    if provider == "ollama":
        return call_ollama(messages, model=model)

    raise ValueError(f"Unknown provider: {provider}")


def call_openrouter(messages: list[Message], model: str | None = None) -> str:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set. Add it to .env first.")

    payload = {
        "model": model or OPENROUTER_MODEL,
        "messages": messages,
        "temperature": 0,
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://localhost/research-agent",
        "X-Title": "research-agent",
    }

    data = _post_json("https://openrouter.ai/api/v1/chat/completions", payload, headers)
    return data["choices"][0]["message"]["content"]


def call_ollama(messages: list[Message], model: str | None = None) -> str:
    payload = {
        "model": model or OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0,
        },
    }

    data = _post_json(f"{OLLAMA_BASE_URL}/api/chat", payload)
    return data["message"]["content"]


def _post_json(url: str, payload: dict, headers: dict[str, str] | None = None) -> dict:
    for attempt in range(3):
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers or {"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if exc.code == 429 and attempt < 2:
                time.sleep(_retry_after_seconds(exc.headers.get("Retry-After")))
                continue

            raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Could not reach {url}: {exc.reason}") from exc

    raise RuntimeError(f"Could not get a response from {url}")


def _retry_after_seconds(value: str | None) -> float:
    if value is None:
        return 2.0

    try:
        return max(float(value), 1.0)
    except ValueError:
        return 2.0
