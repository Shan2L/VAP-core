#!/usr/bin/env python3
"""Run TorchProfilerTraceSkill Perfetto SQL presets.

This runner is intentionally small and dependency-light. It executes named,
project-owned SQL presets against a Perfetto trace through trace_processor.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

QUERY_FILE = Path(__file__).with_name("queries.yaml")


def load_queries() -> dict[str, str]:
    queries: dict[str, list[str]] = {}
    current_name: str | None = None
    current_lines: list[str] = []
    for raw_line in QUERY_FILE.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        if not raw_line.startswith(" ") and raw_line.endswith(": |"):
            if current_name is not None:
                queries[current_name] = current_lines
            current_name = raw_line.split(":", 1)[0].strip()
            current_lines = []
            continue
        if current_name is not None:
            current_lines.append(
                raw_line[2:] if raw_line.startswith("  ") else raw_line
            )
    if current_name is not None:
        queries[current_name] = current_lines
    return {name: "\n".join(lines).strip() for name, lines in queries.items()}


def find_trace_processor() -> str:
    env_path = os.getenv("TRACE_PROCESSOR")
    if env_path:
        return env_path
    for candidate in (
        Path.cwd() / "bin" / "trace_processor",
        Path.cwd() / "trace_processor",
        Path("trace_processor"),
    ):
        if candidate.exists():
            return str(candidate)
    return "trace_processor"


def run_query(
    trace_processor: str, trace: Path, sql: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [trace_processor, "query", str(trace), sql],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def main() -> int:
    queries = load_queries()
    parser = argparse.ArgumentParser(
        description="Run TorchProfilerTraceSkill SQL presets."
    )
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--query", choices=sorted(queries))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--trace-processor", default=find_trace_processor())
    args = parser.parse_args()

    if not args.trace.is_file():
        parser.error(f"Trace does not exist: {args.trace}")
    if not args.all and not args.query:
        parser.error("Use --query NAME or --all")
    if args.limit < 1 or args.limit > 100:
        parser.error("--limit must be from 1 to 100")

    selected = sorted(queries) if args.all else [args.query]
    exit_code = 0
    for name in selected:
        sql = queries[name].format(limit=args.limit)
        print(f"\n## {name}\n")
        completed = run_query(args.trace_processor, args.trace, sql)
        if completed.stdout:
            print(completed.stdout.rstrip())
        if completed.stderr:
            print(completed.stderr.rstrip(), file=sys.stderr)
        if completed.returncode != 0:
            exit_code = completed.returncode
            print(
                f"Query failed with exit code {completed.returncode}", file=sys.stderr
            )
            if not args.all:
                break
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
