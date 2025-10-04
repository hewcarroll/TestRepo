#!/usr/bin/env python3
# main.py — entry point + config load + dry-run
# -----------------------------------------------------------
# This file will orchestrate the scraper. For now, it only:
# 1) Parses CLI args
# 2) Loads and validates config.toml
# 3) Prints a dry-run plan (no fetching yet)

from __future__ import annotations  # future-proof typing (no runtime impact)
import sys                         # read Python version, exit with codes
import argparse                    # command-line interface
from pathlib import Path           # file-safe paths
import json                        # pretty printing of config summary

# Python 3.11+ has tomllib in stdlib; if you're on 3.10, we'll give a clear error.
try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError as e:
    sys.stderr.write(
        "ERROR: Python 3.11+ is required for 'tomllib'. "
        "If you must use Python 3.10, install 'tomli' and change 'import tomllib' to 'import tomli as tomllib'.\n"
    )
    sys.exit(1)


def load_config(path: Path) -> dict:
    """
    Load a TOML config safely.
    - Ensures file exists
    - Returns a Python dict
    """
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("rb") as f:
        return tomllib.load(f)


def validate_config(cfg: dict) -> None:
    """
    Validate minimal required structure for a universal scraper skeleton.
    We keep this intentionally small; we will evolve it later.

    Required top-level keys (with reasonable defaults if missing):
      - project.name (str)
      - fetch.targets (list[str]) — can be empty for now
      - fetch.headers (table/dict) — optional
      - fetch.timeout_secs (int/float) — optional
      - storage.backend (str) — one of: 'none', 'json', 'csv', 'sqlite' (we will implement later)
      - storage.options (table/dict) — optional
    """
    # project
    project = cfg.get("project", {})
    if "name" not in project or not isinstance(project["name"], str) or not project["name"].strip():
        raise ValueError("config.project.name must be a non-empty string")

    # fetch
    fetch = cfg.get("fetch", {})
    targets = fetch.get("targets", [])
    if not isinstance(targets, list):
        raise ValueError("config.fetch.targets must be a list (can be empty for now)")

    # storage
    storage = cfg.get("storage", {})
    backend = storage.get("backend", "none")
    if backend not in {"none", "json", "csv", "sqlite"}:
        raise ValueError("config.storage.backend must be one of: none, json, csv, sqlite")


def main(argv: list[str] | None = None) -> int:
    """
    CLI entry. Returns process exit code.
    """
    parser = argparse.ArgumentParser(description="Universal web scraper skeleton (dry-run only).")
    parser.add_argument(
        "-c", "--config",
        type=Path,
        default=Path("config.toml"),
        help="Path to config TOML (default: config.toml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan without fetching/parsing/storing.",
    )
    args = parser.parse_args(argv)

    # Load config
    try:
        cfg = load_config(args.config)
    except Exception as e:
        sys.stderr.write(f"Failed to load config: {e}\n")
        return 2

    # Validate config
    try:
        validate_config(cfg)
    except Exception as e:
        sys.stderr.write(f"Invalid config: {e}\n")
        return 3

    # Dry-run summary (the only thing we do right now)
    if args.dry_run:
        plan = {
            "project": cfg.get("project", {}),
            "fetch": {
                "targets_count": len(cfg.get("fetch", {}).get("targets", [])),
                "has_headers": bool(cfg.get("fetch", {}).get("headers")),
                "timeout_secs": cfg.get("fetch", {}).get("timeout_secs", None),
            },
            "storage": cfg.get("storage", {}),
            "next_steps": [
                "Implement fetcher.fetch(url, config) with retries/backoff.",
                "Implement parser.parse(raw, config) -> list[record].",
                "Implement storage.save(records, config).",
            ],
        }
        print(json.dumps(plan, indent=2))
        return 0

    # If not dry-run, we still exit early (no behavior implemented yet).
    print("No operation: implement fetch/parse/store next. Try --dry-run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
