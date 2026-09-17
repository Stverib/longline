"""Command-line entry point for the evaluation suite.

Usage examples:
    # run both layers, writing report to evals/results/
    uv run python -m longline.eval --case-file evals/tool_calls.jsonl

    # only e2e, cheap model, quick smoke (first 3 cases)
    uv run python -m longline.eval --type e2e --model claude-haiku-4-5-20251001 --max-cases 3

    # formal run: named suite, 3 repeats, results under evals/results/<run_id>/
    uv run python -m longline.eval --suite e2e --repeats 3 --run-id 2026-09-15_smoke

All legacy flags (`--type`, `--model`, `--case-file`, `--fixtures-dir`,
`--max-cases`, `--out-dir`, `--md`) keep their exact previous behaviour.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

from longline.eval.multi_agent import GROUPS, MAX_WORKERS, MIN_WORKERS, load_multi_agent_cases
from longline.eval.multi_agent_runner import MULTI_AGENT_TAG
from longline.eval.report import (
    _fmt_speedup,  # canonical: a duration ratio has parity at 1.00x, not 0
    aggregate,
    paired_report_delta,
    render_markdown,
)
from longline.eval.runner import (
    CaseResult,
    model_provenance,
    run_suite,
    served_models_in,
)
from longline.eval.trajectory import ToolExecution
from longline.eval.types import E2ECase, EvalCase, ToolCallCase, load_cases
from longline.models.messages import Usage

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Filenames inside `evals/results/<run_id>/` (evals/README.md §3).
RAW_NAME = "raw.jsonl"
SUMMARY_NAME = "summary.json"
REPORT_NAME = "report.md"

# `--suite` presets: suite -> (case file, layer filter).
#
# `tool_selection` is the Task 2 suite: 48 blind + 12 instruction-following
# cases whose main number is ToolSelectionCaseAccuracy over the blind half.
# `compression` is the Task 4 suite and gets its own layer name rather than
# riding on `e2e`: its cases carry a scripted `history` and `key_facts` that
# `E2ECase` has no field for, so loading them as E2E would drop the transcript
# and leave the suite measuring an ordinary artifact check.
# `latency` is the Task 6 suite. Its cases are not loaded from a JSONL file at
# all -- they are declared in `longline/eval/latency_cases.py`, because a latency
# case is a schedule (a block grid plus a tool duration) that no `EvalCase`
# shape can carry. The case-file entry is therefore a *label*, used only for the
# run metadata's `suite` field and the report directory name; `_run_latency`
# never opens it. Pointing it at a real file that exists keeps `--case-file`
# sanity checks and `case_file_sha256` from silently referencing nothing.
# `tool_calls` stays pointed at the retired legacy file so old invocations keep
# reproducing their old case set exactly.
#
# `multi_agent` is the Task 7 suite. Its cases carry a `group`, declared
# subtasks and TWO sibling fixtures, none of which `EvalCase` has a field for,
# so -- like `compression` -- it gets its own layer name rather than riding on
# `e2e`: loading them as E2E would drop the subtask declarations and leave the
# suite comparing two variants whose work it could no longer state.
#
# `safety` is the Task 8 suite. Its cases declare a permission scenario and a
# LABELLED outcome, and they are never dispatched to a real tool -- the runner
# registers an inert sentinel instead. Loading them as E2E would run the
# declared dangerous arguments against real tools, which is the one thing the
# contract forbids outright (`evals/README.md` §8.4).
SUITES: dict[str, tuple[str, str]] = {
    "tool_calls": ("tool_calls.jsonl", "tool_call"),
    "tool_selection": ("tool_selection.jsonl", "tool_call"),
    "e2e": ("e2e.jsonl", "e2e"),
    "compression": ("compression.jsonl", "compression"),
    "latency": ("e2e.jsonl", "latency"),
    "multi_agent": ("multi_agent.jsonl", "multi_agent"),
    "safety": ("safety.jsonl", "safety"),
    "all": ("tool_calls.jsonl", "all"),
}

# Layer names accepted by --type. `compression`, `latency`, `multi_agent` and
# `safety` are separate from `e2e` for the reasons above.
TYPE_CHOICES = [
    "tool_call", "e2e", "compression", "latency", "multi_agent", "safety", "all",
]

# Case tags that partition the tool-selection suite. `--blind` / `--instruction`
# are sugar over `--tag`, kept as flags because the two halves must never be
# silently merged into one reported number.
BLIND_TAG = "blind"
INSTRUCTION_FOLLOWING_TAG = "instruction-following"

# The per-turn usage the offline multi-agent transport reports.
#
# A constant, and documented as one, because a token count has to be SOMETHING
# offline and leaving it at zero makes `TokenOverhead` permanently unmeasurable
# (`(multi - single) / single` with `single = 0` is None) -- the headline metric
# of the suite would silently never produce a number. It is deliberately NOT
# tuned to flatter either arm: every agent of both variants reports the same
# per-turn cost, so the fan-out's overhead is exactly the turns it ran, which is
# the honest version of what a real model would show.
#
# A LIVE RUN MUST NOT USE THIS. `usage` is only passed when `--offline` is set;
# with a real model the transport is the API and the usage is the API's.
SCRIPTED_TURN_USAGE = Usage(input_tokens=1200, output_tokens=180)


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


def _non_negative_float(value: str) -> float:
    """A duration that may legitimately be 0 (i.e. "do not pace at all")."""
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"must be a number, got {value!r}") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {parsed}")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {parsed}")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m longline.eval", description="Run the agent evaluation suite.")
    p.add_argument("--type", choices=TYPE_CHOICES, default="all")
    # The default model is `ANTHROPIC_MODEL` from the project .env when set, so
    # a run records and requests the model the user actually configured; the
    # gateway decides what it serves regardless, and `model_provenance` reports
    # that distinction separately. The legacy constant stays as the fallback.
    p.add_argument("--model",
                   default=_load_env_file().get("ANTHROPIC_MODEL") or "claude-sonnet-4-20250514")
    p.add_argument("--case-file", default=str(PROJECT_ROOT / "evals" / "tool_calls.jsonl"),
                   help="Path to a JSONL file of cases.")
    p.add_argument("--fixtures-dir", default=str(PROJECT_ROOT / "evals" / "fixtures"))
    p.add_argument("--max-cases", type=int, default=None, help="Cap the number of cases (smoke mode).")
    p.add_argument("--out-dir", default=str(PROJECT_ROOT / "evals" / "results"))
    p.add_argument("--md", action="store_true", help="Also write a .md report alongside the .json.")
    # --- Task 1 additions ---
    p.add_argument(
        "--suite", choices=sorted(SUITES), default=None,
        help="Named suite preset; selects the case file and layer. "
             "An explicit --case-file/--type still wins.",
    )
    p.add_argument(
        "--variant", default=None,
        help="Variant label recorded on every result (baseline/candidate, "
             "compression_off/compression_on, ...).",
    )
    p.add_argument(
        "--repeats", type=_positive_int, default=1,
        help="Run the selected cases N times, recording repeat_index. "
             "All runs are kept in raw.jsonl; none is averaged away.",
    )
    p.add_argument(
        "--run-id", default=None,
        help=f"Write to <out-dir>/<run-id>/{{{RAW_NAME},{SUMMARY_NAME},{REPORT_NAME}}}. "
             "When omitted, the legacy flat <model>-<n>.json output is used.",
    )
    p.add_argument(
        "--keep-sandbox-on-failure", action="store_true",
        help="Leave a failed case's temp sandbox on disk and record its path. "
             "Successful cases are always cleaned up.",
    )
    p.add_argument(
        "--pace-seconds", type=_non_negative_float, default=0.0, metavar="SECONDS",
        help="Pause this long between two model calls. The serial contract "
             "keeps quality runs from overlapping, but a gateway can still "
             "reject back-to-back requests from one account, and a rejected "
             "request produces a row that measured nothing. Default 0 (no "
             "pause), which is what every offline suite wants.",
    )
    p.add_argument(
        "--resume-run", action="store_true",
        help="Re-enter an existing --run-id directory and finish it. Rows that "
             "recorded an observation (a pass, a fail, max_turns, "
             "context_overflow) are kept; only rows whose error type means the "
             "case measured nothing (api_error, runtime_error) are re-run. "
             "Requires --run-id.",
    )
    p.add_argument(
        "--tag", default=None,
        help="Only run cases carrying this tag.",
    )
    # --- Task 2 additions ---
    p.add_argument(
        "--tool-profile", default="core",
        help="Which tool families the eval registry offers: core (default), "
             "web, notebook, task, or all. Web tools are offline stand-ins "
             "whose schema matches production, so tool selection is measured "
             "without network flakiness (evals/README.md §5.2).",
    )
    p.add_argument(
        "--tool-profile-by-tag", action="store_true",
        help="Pick the tool profile per case from its family tag "
             "(read/write/edit/glob-grep/bash/web/notebook/task/multi). "
             "Overrides --tool-profile for those cases.",
    )
    p.add_argument(
        "--blind-only", action="store_true",
        help="Run only cases tagged 'blind' — the ones that feed the "
             "resume-facing ToolSelectionCaseAccuracy.",
    )
    p.add_argument(
        "--instruction-only", action="store_true",
        help="Run only cases tagged 'instruction-following'. Reported "
             "separately and never mixed into the blind number.",
    )
    # --- Task 5 additions ---
    p.add_argument(
        "--resume", default=None, metavar="SESSION_JSON",
        help="Resume a session saved under a temp claude_dir: loads the "
             "transcript with the production load_session()/validate_transcript(), "
             "restores the Task snapshot, and prints a JSON report. This is the "
             "resume leg of the Process-Kill recovery case.",
    )
    p.add_argument(
        "--emit-result", default=None, metavar="PATH",
        help="Also write the CaseResult of the last case as JSON to PATH. "
             "Used by the recovery runner's subprocess worker so the parent "
             "reads the child's verdict off a file rather than stdout.",
    )
    # --- Task 6 additions ---
    p.add_argument(
        "--samples", type=int, default=None, metavar="N",
        help="Latency suite: samples per arm per case, in "
             "[30, 50] per contract §5.5 (default 40). Warmups are separate "
             "and are never counted.",
    )
    p.add_argument(
        "--warmups", type=int, default=None, metavar="N",
        help="Latency suite: warmup pairs per case, excluded from every "
             "statistic (contract §5.5 fixes this at 5).",
    )
    p.add_argument(
        "--time-scale", type=float, default=None, metavar="X",
        help="Latency suite: multiplier applied to every duration the cases "
             "declare. Signal scales with X, this host's scheduler jitter does "
             "not, so lowering X shrinks the signal-to-jitter margin. See "
             "longline/eval/latency_runner.py's module docstring for the "
             "measured basis of the default.",
    )
    # --- Task 7 additions ---
    p.add_argument(
        "--group", choices=sorted(GROUPS), default=None, metavar="GROUP",
        help="Multi-agent suite: restrict to one group. 'controlled' cases "
             "pre-declare their independent subtasks, so both variants do the "
             "same work; 'exploratory' cases let the coordinator decompose "
             "freely. The two are reported separately and are never averaged "
             "into one number (contract §5.6).",
    )
    p.add_argument(
        "--offline", action="store_true",
        help="Multi-agent suite: run the offline protocol. A real QueryEngine, "
             "real tools, the real query_loop and a real spawn_teammate fan-out, "
             "with the model transport scripted. Deterministic, free, and what "
             "the committed dataset describes; the run records which mode "
             "produced it.",
    )
    return p.parse_args(argv)


def _run_resume(args: argparse.Namespace) -> int:
    """Restore a session from `--resume` and report what came back.

    Goes through the same three production calls `main.py --resume` makes, in
    the same order. The JSON report is the subprocess worker's only channel, so
    it is written to stdout and nothing else is printed there.
    """
    from longline.eval.recovery_worker import resume

    spec_path = Path(args.resume)
    if not spec_path.is_file():
        print(json.dumps({"found": False, "error": f"spec not found: {spec_path}"}))
        return 1
    report = resume(json.loads(spec_path.read_text(encoding="utf-8")))
    report["phase"] = "resume"
    print(json.dumps(report, ensure_ascii=False))
    return 0


_ORIGINAL_ARGV: list[str] | None = None


def _explicit(name: str, argv: Sequence[str] | None) -> bool:
    """True if `--<name>` appears in argv (used to detect explicit flags).

    `parse_args(None)` reads the real `sys.argv`, so when no argv is passed the
    process argv is consulted. That is what lets `main()` call
    `parse_args(None)` and still know which flags the user actually typed.
    """
    if argv is None:
        argv = sys.argv[1:]
    return any(a == f"--{name}" or a.startswith(f"--{name}=") for a in argv)


def _apply_suite(args: argparse.Namespace, argv: Sequence[str] | None = None) -> None:
    """Resolve `--suite` into --case-file/--type without overriding explicit flags.

    Compatibility rule: an explicit `--case-file` or `--type` always wins, so
    the legacy invocations are byte-for-byte unaffected.

    Note: when `parse_args` was called with an explicit argv, that same argv
    must be threaded through here — otherwise the explicit-flag check would
    consult the real process argv and get the wrong answer.
    """
    if args.suite is None:
        return
    case_file, type_filter = SUITES[args.suite]
    if not _explicit("case-file", argv):
        args.case_file = str(PROJECT_ROOT / "evals" / case_file)
    if not _explicit("type", argv):
        args.type = type_filter


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


def _select_by_tag(cases: list[EvalCase], tag: str | None) -> list[EvalCase]:
    """Filter cases by tag; no tag means no filtering."""
    if tag is None:
        return cases
    return [c for c in cases if tag in c.tags]


# Family tag -> the tool profile that must be registered for the case to be
# answerable at all. A case tagged `web` run against the core registry could
# never pass, and its failure would look like a model error.
_PROFILE_BY_TAG: dict[str, str] = {
    "read": "core", "write": "core", "edit": "core",
    "glob": "core", "grep": "core", "bash": "core",
    "web": "web", "notebook": "notebook", "task": "task",
    "multi": "all",
}


def profile_for_case(case: EvalCase, default: str) -> str:
    """The tool profile a single case needs, or `default` when unspecified.

    When several family tags are present the most capable profile wins, because
    a multi-tool case needs every family it touches to be available at once.
    """
    wanted = [_PROFILE_BY_TAG[t] for t in case.tags if t in _PROFILE_BY_TAG]
    if not wanted:
        return default
    if "all" in wanted:
        return "all"
    return wanted[0]


def make_run_id(model: str, suite: str) -> str:
    """Build a default run id: `<model>_<suite>_<timestamp>-<pid>`.

    No '/' or ':' so it is a safe single directory name on Windows and POSIX.
    The pid keeps two runs started in the same second from colliding, and
    microseconds keep it unique within a process.
    """
    now = time.time()
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime(now))
    micros = int((now % 1) * 1_000_000)
    safe_model = model.replace("/", "__").replace(" ", "-")
    return f"{safe_model}_{suite}_{stamp}{micros:06d}_{os.getpid()}"


def _git_sha() -> str | None:
    """Current commit, or None outside a git checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    sha = out.stdout.strip()
    return sha or None


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_metadata(
    *,
    run_id: str,
    suite: str,
    variant: str | None,
    model: str,
    case_file: Path,
    repeat_index: int,
    repeats_completed: int,
    served_models: list[str] | None = None,
) -> dict[str, object]:
    """The per-run metadata block required by evals/README.md §2.1.

    `served_models` is what the transport reported it actually ran. It is a
    parameter rather than a global because only the caller has the results, and
    it is folded in here rather than at each call site so all six runners
    describe their model the same way.
    """
    return {
        "run_id": run_id,
        "suite": suite,
        "variant": variant,
        "model": model,
        "git_sha": _git_sha(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "case_file_sha256": _sha256(case_file),
        "repeat_index": repeat_index,
        "repeats_completed": repeats_completed,
        **model_provenance(model, served_models or []),
    }


def _write_jsonl(path: Path, results: list[CaseResult]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r.to_raw_dict(), ensure_ascii=False) + "\n")


def _write_run_dir(
    *,
    run_dir: Path,
    results: list[CaseResult],
    metadata: dict[str, object],
    baseline_results: list[CaseResult] | None = None,
    baseline_label: str | None = None,
    compression: object | None = None,
) -> None:
    """Write raw.jsonl / summary.json / report.md for one run.

    `compression` is a `CompressionSummary` when the run was the Task 4 suite.
    It goes into BOTH `summary.json` and `report.md`: a metric that exists only
    in the rendered report is not a metric (`evals/README.md` §3), so the
    payload is written first and the markdown is a view of it.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(run_dir / RAW_NAME, results)

    report = aggregate(results, variant=metadata.get("variant"))  # type: ignore[arg-type]
    summary: dict[str, object] = {
        "metadata": metadata,
        "metrics": report.to_dict(),
    }
    if baseline_results is not None:
        summary["paired_vs_baseline"] = paired_report_delta(baseline_results, results)
    if compression is not None:
        summary["compression"] = compression.to_dict()  # type: ignore[attr-defined]

    (run_dir / SUMMARY_NAME).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    baseline_report = aggregate(baseline_results) if baseline_results is not None else None
    (run_dir / REPORT_NAME).write_text(
        render_markdown(
            report, baseline=baseline_report, baseline_label=baseline_label,
            compression=compression,  # type: ignore[arg-type]
        ),
        encoding="utf-8",
    )


# Failure categories that mean a row measured NOTHING, so re-running it is
# recovery rather than retrying-until-it-passes.
#
# The distinction is the whole point of `--resume-run`, and getting it wrong
# would corrupt every rate in the run. `api_error` (429/529, "gave up after
# retries") and `runtime_error` (a dropped connection, a 402, a crash) are
# statements about the environment -- the model was never asked, or never
# answered. Everything else is an OBSERVATION and is kept:
#
# - a normal pass or a normal fail is the measurement;
# - `max_turns` is the model failing to finish inside the case's own budget;
# - `context_overflow` is the product's context management, working or not.
#
# Deleting those and re-running them would be "run it until it passes", which
# raises the success rate by construction. A benchmark that retries its
# failures measures the retrying.
RETRYABLE_ERROR_TYPES = frozenset({"api_error", "runtime_error"})


def _resume_split(
    rows: Iterable[CaseResult],
) -> tuple[list[CaseResult], list[CaseResult]]:
    """Split previous rows into `(kept, to_rerun)`.

    Kept rows are observations and stay in the artifact; `to_rerun` rows are the
    ones whose error type says the case measured nothing. Pure and separate from
    the run loop so the classification -- the part that decides whether a
    resumed run is honest -- can be tested without a model or a filesystem.
    """
    kept: list[CaseResult] = []
    to_rerun: list[CaseResult] = []
    for row in rows:
        (to_rerun if row.error_type in RETRYABLE_ERROR_TYPES else kept).append(row)
    return kept, to_rerun


def _load_jsonl_results(path: Path) -> list[CaseResult]:
    """Re-hydrate CaseResults from a raw.jsonl produced by a previous run.

    Rebuilds every field the aggregates READ, not just the ones a caller happens
    to want today. `judge_detail` drives `steps_completed`, `abstention_ok` and
    the three matched-call counters; `tool_calls` and `tool_executions` are the
    denominators of precision and execution success. A loader that restored only
    the scalar columns would leave a resumed run reporting `1/6` on a suite
    where every case passed -- which is exactly what a probe caught here before
    `--resume-run` shipped.
    """
    out: list[CaseResult] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        out.append(CaseResult(
            case_id=str(d["case_id"]),
            case_type=str(d["case_type"]),
            passed=bool(d["passed"]),
            turns=int(d.get("num_rounds", 0)),
            input_tokens=int(d.get("input_tokens", 0)),
            output_tokens=int(d.get("output_tokens", 0)),
            errors=[str(e) for e in d.get("errors", [])],
            tags=[str(t) for t in d.get("tags", [])],
            duration_ms=d.get("duration_ms"),
            variant=d.get("variant"),
            repeat_index=int(d.get("repeat_index", 0)),
            error_type=d.get("error_type"),
            # --- the fields the aggregates read ---
            detail=dict(d.get("judge_detail") or {}),
            tool_calls=[
                (str(t["tool_name"]), dict(t.get("input") or {}))
                for t in d.get("tool_calls_with_args", [])
            ],
            tool_executions=[
                ToolExecution(
                    tool_id=str(e["tool_id"]),
                    tool_name=str(e["tool_name"]),
                    is_error=bool(e["is_error"]),
                    # The row carries a derived duration rather than the
                    # absolute span, so rebuild a span that reproduces it.
                    start_ns=0,
                    end_ns=(
                        int(e["duration_ms"] * 1_000_000)
                        if e.get("duration_ms") is not None
                        else None
                    ),
                )
                for e in d.get("tool_executions", [])
            ],
            event_timestamps={
                str(k): int(v) for k, v in (d.get("event_timestamps") or {}).items()
            },
            trial=int(d.get("trial", 0)),
            run_id=d.get("run_id"),
            # Which model the transport actually served. Restored like every
            # other aggregate-bearing field: a resumed run that forgot it would
            # report a measured model as unverified.
            served_models=[str(m) for m in (d.get("served_models") or [])],
        ))
    return out


def _fmt_ratio(ratio: object) -> str:
    """`66.7% (2/3)`, or `not measured (0/0)` — a rate needs its denominator."""
    num = getattr(ratio, "numerator", 0)
    den = getattr(ratio, "denominator", 0)
    value = getattr(ratio, "value", None)
    if value is None:
        return f"not measured ({num}/{den})"
    return f"{value * 100:.1f}% ({num}/{den})"


def _print_run_summary(report: object, num_case_runs: int) -> None:
    """Print the headline numbers, one line per metric.

    The four tool-calling metrics are printed on their own lines with their own
    fractions. Collapsing them into a single "tool accuracy" line is what made
    the legacy number impossible to audit, so the console output refuses to do
    it either.
    """
    l1 = report.l1_ratio  # type: ignore[attr-defined]
    l2 = report.l2_ratio  # type: ignore[attr-defined]
    sel = report.tool_selection_case_accuracy  # type: ignore[attr-defined]
    if sel.denominator:
        print(f"[eval] {num_case_runs} case-runs")
        print(f"[eval]   ToolSelectionCaseAccuracy (blind) : {_fmt_ratio(sel)}")
        print(f"[eval]   ToolCallPrecision                 : {_fmt_ratio(report.tool_call_precision)}")  # type: ignore[attr-defined]
        print(f"[eval]   ArgumentCallAccuracy              : {_fmt_ratio(report.argument_call_accuracy)}")  # type: ignore[attr-defined]
        print(f"[eval]   ArgumentFieldAccuracy             : {_fmt_ratio(report.argument_field_accuracy)}")  # type: ignore[attr-defined]
        print(f"[eval]   ExecutionSuccessRate              : {_fmt_ratio(report.tool_execution_rate)}")  # type: ignore[attr-defined]
        instr = report.instruction_following_case_accuracy  # type: ignore[attr-defined]
        if instr.denominator:
            print(f"[eval]   InstructionFollowing (separate)   : {_fmt_ratio(instr)}")
    elif l2.value is not None:
        print(f"[eval] {num_case_runs} case-runs | E2E pass@1: {_fmt_ratio(l2)}")
    else:
        print(f"[eval] {num_case_runs} case-runs | Tool accuracy: {_fmt_ratio(l1)}")


async def _run_compression(
    args: argparse.Namespace,
    *,
    case_file: Path,
    fixtures: Path,
    out_dir: Path,
    api_key: str,
    run_id: str | None,
) -> int:
    """Run the Task 4 paired compression suite and write its report.

    Kept separate from the generic `_run` body because the outer shape differs:
    one case yields two CaseResults (baseline and candidate) plus a fact trace,
    and the headline numbers are the compression metrics rather than a pass rate.
    The `raw.jsonl` it writes is still the contract's source of truth -- every
    number in `summary.json` is recomputable from those rows plus the case file.
    """
    from longline.eval.compression import load_compression_cases
    from longline.eval.compression_runner import aggregate_compression, run_compression_suite

    cases = load_compression_cases(case_file)
    cases = [c for c in cases if args.tag is None or args.tag in c.tags]
    if args.max_cases is not None:
        cases = cases[: args.max_cases]
    if not cases:
        raise SystemExit("no compression cases selected — check --case-file / --tag")

    run_id = run_id or make_run_id(args.model, "compression")

    runs = await run_compression_suite(
        cases, model=args.model, api_key=api_key, fixtures_dir=fixtures,
    )
    summary = aggregate_compression(runs)

    # Both variants of every case, so `raw.jsonl` alone recomputes the report.
    results: list[CaseResult] = []
    for run in runs:
        results.append(run.baseline)
        results.append(run.candidate)

    ratio = summary.compression_ratio
    print(f"[eval] {summary.num_cases} compression cases "
          f"({summary.excluded_cases} excluded from the denominator)")
    print(f"[eval]   CompressionRatio (estimated tokens): "
          f"{'not measured' if ratio is None else f'{ratio * 100:.1f}%'}")
    print(f"[eval]   KeyInfoRetention                   : {_fmt_ratio(summary.key_info_retention)}")
    print(f"[eval]   PostCompressionSuccessRate         : "
          f"{_fmt_ratio(summary.post_compression_success_rate)}")
    delta = summary.success_delta_pp
    print(f"[eval]   SuccessDeltaPP                     : "
          f"{'not measured' if delta is None else f'{delta:+.1f} pp'}")

    metadata = run_metadata(
        run_id=run_id, suite="compression", variant=args.variant, model=args.model,
        case_file=case_file, repeat_index=0, repeats_completed=1,
        served_models=served_models_in(runs),
    )

    if args.run_id is not None:
        run_dir = out_dir / run_id
        _write_run_dir(
            run_dir=run_dir, results=results, metadata=metadata, compression=summary,
        )
        print(f"[eval] raw      -> {run_dir / RAW_NAME}")
        print(f"[eval] summary  -> {run_dir / SUMMARY_NAME}")
        print(f"[eval] markdown -> {run_dir / REPORT_NAME}")
        return 0

    # Without --run-id there is no durable directory to hold the paired
    # evidence, and a compression number whose `raw.jsonl` is missing is not a
    # number. Refuse rather than emitting a flat file that cannot be audited.
    raise SystemExit(
        "the compression suite needs --run-id: its numbers must be backed by "
        "raw.jsonl under evals/results/<run_id>/ (evals/README.md §3)"
    )


async def _run_latency(
    args: argparse.Namespace,
    *,
    case_file: Path,
    out_dir: Path,
    run_id: str | None,
) -> int:
    """Run the Task 6 streaming-vs-buffered micro-benchmark and write its report.

    No model, no API key, no network: the transport is scripted, which is what
    contract §5.5 asks for ("用可控的延迟工具和脚本化流做稳定微基准").

    `raw.jsonl` here is a different shape from the E2E rows -- one row per
    SAMPLE per arm, tagged `LATENCY_TAG`, carrying the six contract timestamps
    in ns relative to `request_start`. That is deliberate: contract §3 says the
    raw file is the sole source of truth, and the report's mean/p50/p95 and the
    per-case paired reduction are all recomputable from those rows.
    """
    from longline.eval.latency_cases import LATENCY_CASES
    from longline.eval.latency_runner import (
        DEFAULT_SAMPLES,
        DEFAULT_TIME_SCALE,
        WARMUP_ROUNDS,
        pooled_start_reduction,
        run_latency_suite,
    )

    cases = LATENCY_CASES
    if args.tag is not None:
        cases = tuple(c for c in cases if args.tag in c.tags)
    if args.max_cases is not None:
        cases = cases[: args.max_cases]
    if not cases:
        raise SystemExit("no latency cases selected - check --tag / --max-cases")

    samples = DEFAULT_SAMPLES if args.samples is None else args.samples
    warmups = WARMUP_ROUNDS if args.warmups is None else args.warmups
    time_scale = DEFAULT_TIME_SCALE if args.time_scale is None else args.time_scale

    if args.run_id is None:
        # Checked BEFORE the run, not after it. The suite costs minutes at the
        # shipped scale, and refusing to emit a number only once the number has
        # been computed wastes all of it -- the caller waited for nothing. Same
        # reasoning as the compression suite, which refuses for the same
        # contract reason (a figure with no `raw.jsonl` behind it is not a
        # figure), but there the check happens to be cheap.
        raise SystemExit(
            "the latency suite needs --run-id: its numbers must be backed by "
            "raw.jsonl under evals/results/<run_id>/ (evals/README.md §3)"
        )

    run_id = run_id or make_run_id(args.model, "latency")

    summary, recorded = await run_latency_suite(
        cases, samples=samples, warmups=warmups, time_scale=time_scale,
    )
    print(f"[eval] {len(cases)} latency cases x {samples} samples/arm "
          f"(+{warmups} warmups excluded), time_scale={time_scale:g}")
    for case in summary.cases:
        b = case.metrics["buffered"]["tool_start_latency_ms_mean"]
        s = case.metrics["streaming"]["tool_start_latency_ms_mean"]
        reduction = case.reduction
        print(
            f"[eval]   {case.case_id}: ToolStartLatency "
            f"buffered={_fmt_latency(b)} streaming={_fmt_latency(s)} "
            f"LatencyReduction={'n/a' if reduction is None else f'{reduction * 100:+.1f}%'}"
        )
    pooled = pooled_start_reduction(summary.cases)
    print(f"[eval]   LatencyReduction (pooled, n={pooled['n']}): "
          f"mean={_fmt_ratio_value(pooled['mean'])} "
          f"p50={_fmt_ratio_value(pooled['p50'])} "
          f"p95={_fmt_ratio_value(pooled['p95'])}")

    metadata = run_metadata(
        run_id=run_id, suite="latency", variant=args.variant, model=args.model,
        case_file=case_file, repeat_index=0, repeats_completed=1,
        served_models=served_models_in(recorded),
    )

    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / RAW_NAME).open("w", encoding="utf-8") as fh:
        for sample in recorded:
            fh.write(json.dumps(sample.to_row(), ensure_ascii=False) + "\n")
    (run_dir / SUMMARY_NAME).write_text(
        json.dumps({"metadata": metadata, "latency": summary.to_dict()},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (run_dir / REPORT_NAME).write_text(
        render_markdown(aggregate([]), latency=summary), encoding="utf-8",
    )
    print(f"[eval] raw      -> {run_dir / RAW_NAME}")
    print(f"[eval] summary  -> {run_dir / SUMMARY_NAME}")
    print(f"[eval] markdown -> {run_dir / REPORT_NAME}")
    return 0


async def _run_multi_agent(
    args: argparse.Namespace,
    *,
    case_file: Path,
    fixtures: Path,
    out_dir: Path,
    api_key: str,
    run_id: str | None,
) -> int:
    """Run the Task 7 single- vs multi-agent A/B and write its report.

    Its own body rather than the generic `_run`, for the same reason the
    compression suite has one: one case yields TWO CaseResults (one per variant)
    plus a usage ledger, and the headline numbers are ratios of durations and of
    token counts rather than a pass rate. Both variants' shared tokens would be
    meaningless if a reader averaged them into one.

    The two groups are aggregated **separately** and rendered as two sections.
    Contract §5.6 is explicit that the exploratory group is never mixed into the
    controlled number: a controlled case pre-declares its subtasks so both arms
    do the same work, while an exploratory case's coordinator decomposes freely,
    and an average across the two would compare different work under one
    heading.

    `--run-id` is required for the same reason the other paired suites require
    it: a number whose `raw.jsonl` does not exist is not a number
    (`evals/README.md` §3), and this suite's numbers are exactly the kind that
    get quoted without their backing.
    """
    from longline.eval.multi_agent import CONTROLLED, EXPLORATORY
    from longline.eval.multi_agent_runner import (
        aggregate_multi_agent,
        run_multi_agent_suite,
    )

    if args.run_id is None:
        raise SystemExit(
            "the multi_agent suite needs --run-id: its numbers must be backed by "
            "raw.jsonl under evals/results/<run_id>/ (evals/README.md §3)"
        )

    cases = load_multi_agent_cases(case_file)
    if args.group is not None:
        cases = [c for c in cases if c.group == args.group]
    if args.tag is not None:
        cases = [c for c in cases if args.tag in c.tags]
    if args.max_cases is not None:
        cases = cases[: args.max_cases]
    if not cases:
        raise SystemExit("no multi_agent cases selected -- check --case-file / --group / --tag")

    run_id = run_id or make_run_id(args.model, "multi_agent")

    runs = await run_multi_agent_suite(
        cases, api_key=api_key, fixtures_dir=fixtures,
        # `model=None` is the offline protocol: the transport is scripted, the
        # engine and the fan-out are not. Either way the assertions are
        # identical and the row records which mode produced it.
        model=None if args.offline else args.model,
        claude_dir=None,
        usage=SCRIPTED_TURN_USAGE if args.offline else None,
    )
    summaries = [
        aggregate_multi_agent(runs, group=group)
        for group in (CONTROLLED, EXPLORATORY)
        if any(r.group == group for r in runs)
    ]

    print(f"[eval] {len(runs)} multi-agent cases across {len(summaries)} group(s)")
    for summary in summaries:
        print(f"[eval]   group={summary.group}: {summary.num_cases} cases, "
              f"{summary.eligible_cases} eligible, {summary.excluded_cases} excluded")
        print(f"[eval]     SuccessRate single={_fmt_ratio(summary.single_success_rate)} "
              f"multi={_fmt_ratio(summary.multi_success_rate)}")
        print(f"[eval]     WallClockTime single={_fmt_ms(summary.single_wall_time_ms)} "
              f"multi={_fmt_ms(summary.multi_wall_time_ms)}")
        print(f"[eval]     Speedup={_fmt_speedup(summary.mean_speedup)} "
              f"(single_wall / multi_wall; 1.00x is parity)")
        print(f"[eval]     TokenOverhead={_fmt_ratio_value(summary.mean_token_overhead)} "
              f"single_total={summary.single_tokens['total_tokens']} "
              f"multi_total={summary.multi_tokens['total_tokens']} "
              f"multi_child={summary.multi_tokens['child_tokens']}")
        print(f"[eval]     ToolCalls single={summary.single_tool_calls} "
              f"multi={summary.multi_tool_calls} | agent_counts={summary.agent_counts}")

    results: list[CaseResult] = []
    for run in runs:
        results.extend(_multi_agent_results(run))

    metadata = run_metadata(
        run_id=run_id, suite="multi_agent", variant=args.variant, model=args.model,
        case_file=case_file, repeat_index=0, repeats_completed=1,
        served_models=served_models_in(runs),
    )
    metadata["multi_agent"] = {
        "groups": [s.group for s in summaries],
        "agent_counts": sorted({c for s in summaries for c in s.agent_counts}),
        "worker_range": [MIN_WORKERS, MAX_WORKERS],
    }

    # `raw.jsonl` carries BOTH row shapes, tagged so a reader can tell them
    # apart. The paired row (one per case, both arms on one line) is what makes
    # the summary's ratios recomputable -- `Speedup` needs both durations and
    # `TokenOverhead` needs both token counts, and neither can be recovered from
    # two separate per-variant rows. The per-variant CaseResults are there for
    # the generic tables and are never a substitute for the paired row.
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / RAW_NAME).open("w", encoding="utf-8") as fh:
        for run in runs:
            fh.write(json.dumps(tagged(run.to_row()), ensure_ascii=False) + "\n")
        for result in results:
            fh.write(json.dumps(tagged(result.to_raw_dict()), ensure_ascii=False) + "\n")
    (run_dir / SUMMARY_NAME).write_text(
        json.dumps(
            {"metadata": metadata, "multi_agent": {s.group: s.to_dict() for s in summaries}},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / REPORT_NAME).write_text(
        render_markdown(
            aggregate(results), multi_agent={s.group: s for s in summaries},
        ),
        encoding="utf-8",
    )
    print(f"[eval] raw      -> {run_dir / RAW_NAME}")
    print(f"[eval] summary  -> {run_dir / SUMMARY_NAME}")
    print(f"[eval] markdown -> {run_dir / REPORT_NAME}")
    return 0


def tagged(row: dict[str, object]) -> dict[str, object]:
    """Mark a raw row with the suite tag, so a reader can tell the two row
    shapes in `raw.jsonl` apart without guessing from their keys."""
    tags = row.get("tags")
    row["tags"] = [MULTI_AGENT_TAG, *(tags if isinstance(tags, list) else [])]
    return row


def _multi_agent_results(run: object) -> list[CaseResult]:
    """The two per-variant `CaseResult`s a multi-agent run contributes.

    Returned so the generic `aggregate()` has something to summarise (the
    per-case tables and the latency percentiles), while the suite's OWN metrics
    come from the summary objects. The variant label is what keeps the two arms
    from being pooled into one row there.
    """
    out: list[CaseResult] = []
    for variant in (run.single, run.multi):  # type: ignore[attr-defined]
        result = CaseResult(
            case_id=run.case_id,  # type: ignore[attr-defined]
            case_type="multi_agent",
            passed=variant.passed,
            duration_ms=variant.duration_ms,
            input_tokens=variant.input_tokens,
            output_tokens=variant.output_tokens,
            variant=variant.variant,
            tags=[MULTI_AGENT_TAG],
            errors=list(variant.errors) + ([variant.accounting_error] if variant.accounting_error else []),
        )
        result.detail = {
            "subtask_verdicts": variant.subtask_verdicts,
            "agent_count": variant.agent_count,
            "child_tokens": variant.ledger.child_tokens(),
            "accounts": variant.accounts,
            "offline": variant.offline,
        }
        out.append(result)
    return out


def _fmt_latency(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}ms"


def _fmt_ms(value: float | None) -> str:
    """A wall-clock duration for the console, `n/a` when not measured.

    Distinct from `report._fmt_ms` only because the console prints before the
    report exists; both render None as `n/a` for the same reason -- a run whose
    duration was never taken is not a run that took 0 ms.
    """
    return "n/a" if value is None else f"{value:.1f}ms"


def _fmt_ratio_value(value: object) -> str:
    """A `ratio_of_durations` rendered as a percent *change*, never as `pp`."""
    if not isinstance(value, (int, float)):
        return "n/a"
    return f"{value * 100:+.1f}%"


async def _run_safety(
    args: argparse.Namespace,
    *,
    case_file: Path,
    out_dir: Path,
    run_id: str | None,
) -> int:
    """Run the Task 8 permission-safety suite and write its report.

    No model, no API key, no network, no sandbox: a safety case is a permission
    DECISION, and the runner drives it through the production
    `StreamingToolExecutor` against an inert sentinel. Nothing a case declares is
    ever handed to a real tool, which is why this suite needs no fixtures and no
    credentials.

    `--run-id` is required for the same contract reason the other Task 1+
    suites require it: a number whose `raw.jsonl` does not exist is not a number
    (`evals/README.md` §3), and these figures are exactly the kind that get
    quoted without their backing.

    The interactive cases patch `longline.ui.renderer.console` for the duration
    of one call. That is why the suite runs serially: two in flight at once would
    answer each other's prompt.
    """
    from longline.eval.safety import load_safety_cases
    from longline.eval.safety_runner import aggregate_safety, run_safety_suite

    if args.run_id is None:
        raise SystemExit(
            "the safety suite needs --run-id: its numbers must be backed by "
            "raw.jsonl under evals/results/<run_id>/ (evals/README.md §3)"
        )

    cases = load_safety_cases(case_file)
    if args.tag is not None:
        cases = [c for c in cases if args.tag in c.tags]
    if args.max_cases is not None:
        cases = cases[: args.max_cases]
    if not cases:
        raise SystemExit("no safety cases selected -- check --case-file / --tag")

    run_id = run_id or make_run_id(args.model, "safety")

    runs = await run_safety_suite(cases)
    summary = aggregate_safety(runs)

    print(f"[eval] {len(runs)} safety cases "
          f"({summary.false_negatives} false negatives, "
          f"{summary.false_positives} false positives)")
    print(f"[eval]   DangerousRecall   : {_fmt_ratio(summary.dangerous_recall)}")
    print(f"[eval]   FalsePositiveRate : {_fmt_ratio(summary.false_positive_rate)}")
    for axis_name, axis in (
        ("kind", summary.by_kind), ("mode", summary.by_mode),
        ("rule_arm", summary.by_rule_arm),
    ):
        for bucket, counts in sorted(axis.items()):
            print(f"[eval]   by_{axis_name}={bucket}: "
                  f"dangerous_gated={_fmt_ratio(counts['dangerous_gated'])} "
                  f"normal_gated={_fmt_ratio(counts['normal_gated'])}")
    if summary.failures:
        # Printed, not summarised away: a case that did not match its declared
        # outcome is the whole reason this suite exists, and burying it under a
        # green headline would make the number unreadable.
        print(f"[eval]   FAILURES: {[f['case_id'] for f in summary.failures]}")

    metadata = run_metadata(
        run_id=run_id, suite="safety", variant=args.variant, model=args.model,
        case_file=case_file, repeat_index=0, repeats_completed=1,
        served_models=served_models_in(runs),
    )

    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / RAW_NAME).open("w", encoding="utf-8") as fh:
        for run in runs:
            fh.write(json.dumps(run.to_row(run.case), ensure_ascii=False) + "\n")
    (run_dir / SUMMARY_NAME).write_text(
        json.dumps({"metadata": metadata, "safety": summary.to_dict()},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (run_dir / REPORT_NAME).write_text(
        render_markdown(aggregate([]), safety=summary), encoding="utf-8",
    )
    print(f"[eval] raw      -> {run_dir / RAW_NAME}")
    print(f"[eval] summary  -> {run_dir / SUMMARY_NAME}")
    print(f"[eval] markdown -> {run_dir / REPORT_NAME}")
    return 0


async def _run(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _apply_suite(args, argv)
    _apply_base_url()  # adopt OpenCode gateway if configured (no native key)
    api_key = _load_api_key()
    case_file = Path(args.case_file)
    if not case_file.is_file():
        raise SystemExit(f"case file not found: {case_file}")
    fixtures = Path(args.fixtures_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # The compression suite is dispatched before the generic path: its cases are
    # not `EvalCase`s (they carry a history and key facts), and it produces two
    # CaseResults plus a fact trace per case rather than one.
    if args.type == "compression" or args.suite == "compression":
        return await _run_compression(
            args, case_file=case_file, fixtures=fixtures, out_dir=out_dir,
            api_key=api_key, run_id=args.run_id,
        )

    # The latency suite is dispatched before the case-file load: its cases are
    # declared in code, not in a JSONL file, and it produces one sample per arm
    # rather than a pass/fail CaseResult per case.
    if args.type == "latency" or args.suite == "latency":
        return await _run_latency(
            args, case_file=case_file, out_dir=out_dir, run_id=args.run_id,
        )

    # The multi-agent suite, for the same reason: one case yields two CaseResults
    # plus a usage ledger, and its headline numbers are a ratio of durations and
    # a ratio of token counts rather than a pass rate.
    if args.type == "multi_agent" or args.suite == "multi_agent":
        return await _run_multi_agent(
            args, case_file=case_file, fixtures=fixtures, out_dir=out_dir,
            api_key=api_key, run_id=args.run_id,
        )

    # The safety suite, for the same reason again: its cases declare a labelled
    # permission outcome rather than a task, and dispatching them to real tools
    # is exactly what the contract forbids.
    if args.type == "safety" or args.suite == "safety":
        return await _run_safety(
            args, case_file=case_file, out_dir=out_dir, run_id=args.run_id,
        )

    cases = load_cases(case_file)
    if args.type == "tool_call":
        cases = _select_cases(cases, "tool_call")
    elif args.type == "e2e":
        cases = _select_cases(cases, "e2e")
    cases = _select_by_tag(cases, args.tag)
    if args.blind_only:
        cases = _select_by_tag(cases, BLIND_TAG)
    if args.instruction_only:
        cases = _select_by_tag(cases, INSTRUCTION_FOLLOWING_TAG)
    if args.max_cases is not None:
        cases = cases[: args.max_cases]
    if not cases:
        raise SystemExit("no cases selected — check --type / --case-file / --tag")

    run_id: str = args.run_id or make_run_id(args.model, args.suite or args.type)
    suite = args.suite or args.type

    # --- resumable output -------------------------------------------------
    # Every case is appended to `raw.jsonl` the moment it finishes, so an
    # interruption -- an exhausted usage window, a Ctrl-C, a dropped machine --
    # keeps what was already measured. `--resume-run` re-enters the same
    # directory, keeps every row that is an observation, and re-runs only the
    # ones that measured nothing (see `RETRYABLE_ERROR_TYPES`).
    #
    # Without this, a paid suite that outlasts an account's usage window is
    # unaffordable rather than merely slow: the work is discarded and the whole
    # run is paid for again.
    run_dir = out_dir / run_id
    raw_path = run_dir / RAW_NAME
    already: list[CaseResult] = []
    done: set[tuple[str, int]] = set()
    if args.resume_run:
        if not raw_path.is_file():
            raise SystemExit(f"--resume-run: no {RAW_NAME} at {raw_path}")
        kept, dropped = _resume_split(_load_jsonl_results(raw_path))
        already = kept
        done = {(r.case_id, r.repeat_index) for r in kept}
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_jsonl(raw_path, kept)
        kinds = sorted({r.error_type for r in dropped if r.error_type})
        print(
            f"[eval] resume: kept {len(kept)} rows, re-running {len(dropped)} "
            f"that measured nothing {kinds}"
        )

    def _sink(result: CaseResult) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        with raw_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(result.to_raw_dict(), ensure_ascii=False) + "\n")

    all_results: list[CaseResult] = list(already)
    for repeat_index in range(args.repeats):
        all_results.extend(await run_suite(
            cases,
            model=args.model,
            api_key=api_key,
            fixtures_dir=fixtures,
            variant=args.variant,
            repeat_index=repeat_index,
            run_id=run_id,
            keep_sandbox_on_failure=args.keep_sandbox_on_failure,
            tool_profile=args.tool_profile,
            profile_for_case=(
                (lambda case: profile_for_case(case, args.tool_profile))
                if args.tool_profile_by_tag else None
            ),
            skip_case_ids=frozenset(
                cid for (cid, rep) in done if rep == repeat_index
            ),
            sink=_sink,
            pace_seconds=args.pace_seconds,
        ))

    report = aggregate(all_results, variant=args.variant)
    _print_run_summary(report, len(all_results))

    if args.emit_result is not None and all_results:
        # The worker's channel to its parent. Deliberately the LAST case's
        # result and not an aggregate: a worker subprocess runs exactly one
        # case, and an aggregate of one is the same object with more places to
        # disagree.
        Path(args.emit_result).write_text(
            json.dumps(all_results[-1].to_raw_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    metadata = run_metadata(
        run_id=run_id, suite=suite, variant=args.variant, model=args.model,
        case_file=case_file, repeat_index=args.repeats - 1, repeats_completed=args.repeats,
        served_models=served_models_in(all_results),
    )

    if args.run_id is not None:
        # Explicit run id -> the contract layout evals/results/<run_id>/.
        baseline_file = out_dir / "baseline" / RAW_NAME
        baseline_results = (
            _load_jsonl_results(baseline_file) if baseline_file.is_file() else None
        )
        _write_run_dir(
            run_dir=run_dir, results=all_results, metadata=metadata,
            baseline_results=baseline_results, baseline_label="baseline",
        )
        print(f"[eval] raw      -> {run_dir / RAW_NAME}")
        print(f"[eval] summary  -> {run_dir / SUMMARY_NAME}")
        print(f"[eval] markdown -> {run_dir / REPORT_NAME}")
        return 0

    # Legacy path: unchanged flat output. Not a durable data source; use
    # --run-id for anything a report or baseline will cite.
    stem = f"{args.model.replace('/', '__')}-{len(all_results)}"
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
    return 0


def main(argv: Sequence[str] | None = None) -> None:
    raise SystemExit(asyncio.run(_run(argv)))


__all__ = [
    "RAW_NAME", "REPORT_NAME", "SUITES", "SUMMARY_NAME", "main", "parse_args", "split_cases",
]
