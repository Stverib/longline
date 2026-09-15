"""Command-line entry point for the evaluation suite.

Usage examples:
    # run both layers, writing report to evals/results/
    uv run python -m longline.eval --case-file evals/tool_calls.jsonl

    # only e2e, cheap model, quick smoke (first 3 cases)
    uv run python -m longline.eval --type e2e --model claude-haiku-4-5-20251001 --max-cases 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

from longline.eval.report import aggregate, render_markdown
from longline.eval.runner import run_suite
from longline.eval.types import E2ECase, EvalCase, ToolCallCase, load_cases

if TYPE_CHECKING:
    from collections.abc import Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _load_env_file() -> dict[str, str]:
    """Read KEY=VALUE pairs from the project-root .env file."""
    env_file = PROJECT_ROOT / ".env"
    out: dict[str, str] = {}
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, _, v = line.partition("=")
                out[k.strip()] = v.strip()
    return out


def _load_api_key() -> str:
    """Resolve the API key for eval runs.

    Priority: env ANTHROPIC_API_KEY > env OPENCODE_API_KEY > .env keys
    (ANTHROPIC_API_KEY then OPENCODE_API_KEY). OpenCode is an Anthropic-
    compatible gateway used for eval when no native key is configured.
    """
    key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENCODE_API_KEY")
    if key:
        return key
    env = _load_env_file()
    for name in ("ANTHROPIC_API_KEY", "OPENCODE_API_KEY"):
        val = env.get(name)
        if val:
            return val
    raise SystemExit("Error: no ANTHROPIC_API_KEY/OPENCODE_API_KEY in env or .env")


def _apply_base_url() -> None:
    """Point the anthropic SDK at the OpenCode gateway when configured.

    A .env OPENCODE_BASE_URL_GO takes priority over any ambient
    ANTHROPIC_BASE_URL (e.g. a session-local proxy), because the user
    explicitly configured the OpenCode endpoint for eval runs.

    The anthropic SDK appends `/v1/messages` to the base URL it is given,
    so we strip any trailing `/v1` (or `/v1/`) from the configured value to
    avoid a duplicated path segment (e.g. `/zen/go/v1/v1/messages`).
    """
    base = _load_env_file().get("OPENCODE_BASE_URL_GO")
    if not base:
        return
    for suffix in ("/v1/", "/v1"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    os.environ["ANTHROPIC_BASE_URL"] = base


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m longline.eval", description="Run the agent evaluation suite.")
    p.add_argument("--type", choices=["tool_call", "e2e", "all"], default="all")
    p.add_argument("--model", default="claude-sonnet-4-20250514")
    p.add_argument("--case-file", default=str(PROJECT_ROOT / "evals" / "tool_calls.jsonl"),
                   help="Path to a JSONL file of cases.")
    p.add_argument("--fixtures-dir", default=str(PROJECT_ROOT / "evals" / "fixtures"))
    p.add_argument("--max-cases", type=int, default=None, help="Cap the number of cases (smoke mode).")
    p.add_argument("--out-dir", default=str(PROJECT_ROOT / "evals" / "results"))
    p.add_argument("--md", action="store_true", help="Also write a .md report alongside the .json.")
    return p.parse_args(argv)


def split_cases(cases: list[EvalCase]) -> tuple[list[ToolCallCase], list[E2ECase]]:
    """Partition cases by layer, preserving order."""
    tc = [c for c in cases if isinstance(c, ToolCallCase)]
    e2e = [c for c in cases if isinstance(c, E2ECase)]
    return tc, e2e


def _select_cases(cases: list[EvalCase], type_filter: str) -> list[EvalCase]:
    if type_filter == "all":
        return cases
    wanted = ToolCallCase if type_filter == "tool_call" else E2ECase
    return [c for c in cases if isinstance(c, wanted)]


async def _run(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    _apply_base_url()  # adopt OpenCode gateway if configured (no native key)
    api_key = _load_api_key()
    case_file = Path(args.case_file)
    if not case_file.is_file():
        raise SystemExit(f"case file not found: {case_file}")
    fixtures = Path(args.fixtures_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = load_cases(case_file)
    if args.type == "tool_call":
        cases = _select_cases(cases, "tool_call")
    elif args.type == "e2e":
        cases = _select_cases(cases, "e2e")
    if args.max_cases is not None:
        cases = cases[: args.max_cases]
    if not cases:
        raise SystemExit("no cases selected — check --type / --case-file")

    results = await run_suite(cases, model=args.model, api_key=api_key, fixtures_dir=fixtures)
    report = aggregate(results)

    verb = "Tool accuracy" if report.l1_tool_accuracy is not None else "E2E pass@1"
    value = report.l1_tool_accuracy if report.l1_tool_accuracy is not None else report.l2_pass1
    if value is not None:
        print(f"[eval] {len(results)} cases | {verb}: {value * 100:.1f}%")
    else:
        print(f"[eval] {len(results)} cases")

    stem = f"{args.model.replace('/', '__')}-{len(results)}"
    json_path = out_dir / f"{stem}.json"
    json_path.write_text(json.dumps({
        "model": args.model,
        "l1_tool_accuracy": report.l1_tool_accuracy,
        "l2_pass1": report.l2_pass1,
        "avg_turns": report.avg_turns,
        "avg_input_tokens": report.avg_input_tokens,
        "avg_output_tokens": report.avg_output_tokens,
        "per_case": report.per_case,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[eval] report -> {json_path}")
    if args.md:
        md_path = out_dir / f"{stem}.md"
        md_path.write_text(render_markdown(report), encoding="utf-8")
        print(f"[eval] markdown -> {md_path}")


def main(argv: Sequence[str] | None = None) -> None:
    asyncio.run(_run(argv))


if __name__ == "__main__":
    main()
