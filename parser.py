#!/usr/bin/env python3
# parser.py — content parsing layer
# ---------------------------------
# Goal: turn raw bytes from the fetcher into structured Python objects.
# Start simple: detect JSON vs. everything-else; return a list of records.

from __future__ import annotations
from typing import Any, Dict, List
import json

from fetcher import FetchResult  # typed holder for fetch results


def _looks_like_json(text: str) -> bool:
    """
    Heuristic to avoid throwing on HTML/text.
    We don't parse here—just a quick check for leading JSON tokens.
    """
    s = text.lstrip()
    return (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]"))


def parse(raw: FetchResult, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convert a FetchResult into a list[dict].
    - If fetch failed: return a single error record.
    - If body is JSON: return parsed list or single-object wrapped as a list.
    - Else: return a single record with a short text snippet.
    """
    # 1) Normalize failures early so callers can handle uniformly
    if not raw.ok:
        return [{
            "url": raw.url,
            "final_url": raw.final_url,
            "status": raw.status,
            "error": raw.error or "unknown fetch error",
        }]

    # 2) Decode bytes safely
    text = raw.content.decode("utf-8", errors="replace")

    # 3) Try JSON first (common for APIs)
    if _looks_like_json(text):
        try:
            obj = json.loads(text)
            if isinstance(obj, list):
                return obj  # already a list of records
            return [obj]     # wrap single object
        except Exception as e:
            # If JSON parsing fails, fall back to a text record with error note
            return [{
                "url": raw.final_url,
                "status": raw.status,
                "error": f"JSON parse error: {e}",
                "snippet": text[:500],
            }]

    # 4) Default: treat as plain text/HTML and return a small preview
    return [{
        "url": raw.final_url,
        "status": raw.status,
        "snippet": text[:500],  # keep small to avoid flooding logs
    }]
