#!/usr/bin/env python3
# fetcher.py — stdlib HTTP GET with retries/backoff
# -------------------------------------------------
# No third-party deps; uses urllib so this runs anywhere.
# Returns a normalized FetchResult so the rest of the app has a stable interface.

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import time
import urllib.request
import urllib.error


@dataclass
class FetchConfig:
    """Network settings pulled from config.toml."""
    timeout_secs: float = 20.0          # request timeout
    max_retries: int = 2                # number of retries after the first attempt
    backoff_base: float = 0.75          # seconds; grows exponentially per retry
    headers: Optional[Dict[str, str]] = None  # HTTP headers (e.g., User-Agent)

    @staticmethod
    def from_dict(cfg: dict) -> "FetchConfig":
        # Safely read values from the loaded TOML dict
        f = cfg.get("fetch", {})
        return FetchConfig(
            timeout_secs=float(f.get("timeout_secs", 20.0)),
            max_retries=int(f.get("max_retries", 2)),
            backoff_base=float(f.get("backoff_base", 0.75)),
            headers=f.get("headers") or {},
        )


@dataclass
class FetchResult:
    """Uniform response object for callers."""
    ok: bool
    status: int
    url: str
    final_url: str
    content: bytes
    error: Optional[str] = None


def _build_request(url: str, headers: Dict[str, str]) -> urllib.request.Request:
    # Ensure a reasonable default User-Agent if none supplied
    if "User-Agent" not in headers:
        headers["User-Agent"] = "Mozilla/5.0 (compatible; UniversalScraper/0.1)"
    return urllib.request.Request(url=url, headers=headers, method="GET")


def fetch(url: str, config: dict) -> FetchResult:
    """
    Perform a GET with timeout + retries + exponential backoff.
    Returns FetchResult with bytes content or error info.
    """
    fc = FetchConfig.from_dict(config)
    attempt = 0
    last_err: Optional[str] = None

    while True:
        try:
            req = _build_request(url, dict(fc.headers or {}))  # copy per attempt
            with urllib.request.urlopen(req, timeout=fc.timeout_secs) as resp:
                data = resp.read()
                status = getattr(resp, "status", 200)
                final_url = resp.geturl()
                return FetchResult(True, status, url, final_url, data)
        except urllib.error.HTTPError as e:
            last_err = f"HTTPError {e.code}: {e.reason}"
            status = getattr(e, "code", 0) or 0
        except urllib.error.URLError as e:
            last_err = f"URLError: {getattr(e, 'reason', e)}"
            status = 0
        except Exception as e:
            last_err = f"Exception: {e}"
            status = 0

        if attempt >= fc.max_retries:
            return FetchResult(False, status, url, url, b"", last_err)

        time.sleep(fc.backoff_base * (2 ** attempt))  # exponential backoff
        attempt += 1
