"""Run the loop-resume suite and write its raw rows plus a summary.

Small and separate from `longline.eval.cli` on purpose. That module drives the
token/tool-selection suites through a `SUITES` registry with its own run-directory
layout, and this suite's output shape is different enough -- per-arm rates, a
side-effect journal, a drift verdict -- that bolting it on would give one CLI two
unrelated notions of what a "run" is.

    uv run python -m longline.eval.loop_resume_cli --out evals/results/loop_resume
    uv run python -m longline.eval.loop_resume_cli --no-durability --out ... --label ablation

`--no-durability` is the ablation described in `loop_resume_runner`'s docstring:
the same harness with the runtime's two new mechanisms switched off, which is how
the before/after is taken with ONE instrument.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from longline.eval.loop_resume import cases_by_failpoint, load_loop_resume_cases
from longline.eval.loop_resume_runner import aggregate_loop_resume, run_loop_resume_suite

if TYPE_CHECKING:
    from collections.abc import Sequence

DEFAULT_CASES = Path("evals/loop_resume.jsonl")
DEFAULT_FIXTURES = Path("evals/fixtures")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m longline.eval.loop_resume_cli")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--out", type=Path, required=True, help="Run directory.")
    parser.add_argument(
        "--label",
        default="durability",
        help="What this cell is, recorded in the summary so two cells cannot be confused.",
    )
    parser.add_argument(
        "--no-durability",
        dest="durability",
        action="store_false",
        help="The ablation: run with the runtime's step checkpoints and journal off.",
    )
    parser.add_argument("--api-key", default="offline")
    return parser.parse_args(argv)


def _check_the_runs_are_real(runs: Sequence[object]) -> list[str]:
    """Refuse to report a rate over runs whose child never signalled.

    A rate computed over injections that did not fire is not a measurement, and
    the failure is silent: the arms it touches simply look worse for a reason
    that has nothing to do with the thing under test.
    """
    problems: list[str] = []
    unfired = [r.case_id for r in runs if not r.failpoint_reached]  # type: ignore[attr-defined]
    if unfired:
        problems.append(f"{len(unfired)} run(s) never reached their failpoint: {unfired[:5]}")
    return problems


def run(args: argparse.Namespace) -> int:
    cases = load_loop_resume_cases(args.cases, fixtures_root=args.fixtures)
    runs = asyncio.run(
        run_loop_resume_suite(
            cases,
            api_key=args.api_key,
            fixtures_dir=args.fixtures,
            durability=args.durability,
        )
    )
    summary = aggregate_loop_resume(runs)

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "raw.jsonl").open("w", encoding="utf-8") as handle:
        for run_ in runs:
            handle.write(json.dumps(run_.to_row(), ensure_ascii=False, sort_keys=True) + "\n")

    payload = {
        "label": args.label,
        "durability": args.durability,
        "cases": str(args.cases),
        "num_runs": len(runs),
        "by_failpoint_counts": {
            name: len(group) for name, group in cases_by_failpoint(cases).items()
        },
        "problems": _check_the_runs_are_real(runs),
        **summary.to_dict(),
    }
    (args.out / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )

    print(f"[loop-resume] label       = {args.label} (durability={args.durability})")
    print(f"[loop-resume] runs        = {len(runs)}")
    print(f"[loop-resume] resume rate = {summary.loop_resume_rate}")
    print(f"[loop-resume] drift recall= {summary.drift_recall}")
    print(f"[loop-resume] false rej.  = {summary.false_reject_rate}")
    for failpoint, ratio in summary.by_failpoint.items():
        print(f"[loop-resume]   {failpoint:<26} {ratio}")
    for problem in payload["problems"]:
        print(f"[loop-resume] PROBLEM: {problem}", file=sys.stderr)
    print(f"[loop-resume] raw     -> {args.out / 'raw.jsonl'}")
    print(f"[loop-resume] summary -> {args.out / 'summary.json'}")
    return 1 if payload["problems"] else 0


def main(argv: Sequence[str] | None = None) -> None:
    raise SystemExit(run(parse_args(argv)))


if __name__ == "__main__":  # pragma: no cover
    main()
