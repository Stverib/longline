"""Deterministic judges for the evaluation suite.

Layer 1 (tool calls): `judge_steps()` maps the agent's calls onto the case's
    accepted decision steps, and `judge_case_args()` decides argument
    correctness both per field and per call. Together they expose the
    numerator/denominator of all four tool-calling metrics
    (`evals/README.md` §5.2) — nothing is collapsed into a single boolean,
    because a fused boolean cannot carry an independent denominator.
    `check_tools()` / `check_args()` remain as the legacy boolean wrappers.
Layer 2 (E2E): judge_case() dispatches named, side-effect-limited checks run
    against a sandbox directory. All pass/fail is computed with re, the
    filesystem, and subprocess exit codes — never an LLM.

Paths in judge args are relative to the sandbox directory and resolved here.
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from longline.eval.trajectory import ToolCall


# --- Layer-1 decision-step matching ---


@dataclass(frozen=True)
class StepMatch:
    """How one call sequence lines up with a case's accepted decision steps.

    Two index lists, deliberately kept separate:

    - `step_indices[i]` is the call index that satisfied step `i`, or None if
      that step was never satisfied. `step_indices` is therefore the per-step
      detail the contract asks to be emitted, not a pass/fail summary.
    - `extra_call_indices` are the calls that matched no step. They are the
      ToolCallPrecision denominator's "invalid" half, so they must be reported
      rather than discarded.
    """

    step_indices: list[int | None]
    matched_call_indices: list[int]
    extra_call_indices: list[int]

    @property
    def all_steps_matched(self) -> bool:
        """ToolSelectionCaseAccuracy's per-case outcome (contract §5.2)."""
        return all(i is not None for i in self.step_indices)

    @property
    def num_extra_calls(self) -> int:
        return len(self.extra_call_indices)

    @property
    def num_matched_calls(self) -> int:
        return len(self.matched_call_indices)

    def exceeded_extra_budget(self, max_extra_calls: int) -> bool:
        """True when unmatched calls outnumber the case's `max_extra_calls`.

        This never flips `all_steps_matched`: extra calls are visible as a
        *rate* (ToolCallPrecision), not as a hidden rewrite of the case result.
        """
        return self.num_extra_calls > max_extra_calls

    def to_detail(self) -> dict[str, object]:
        return {
            "step_indices": self.step_indices,
            "matched_call_indices": self.matched_call_indices,
            "extra_call_indices": self.extra_call_indices,
            "all_steps_matched": self.all_steps_matched,
            "num_extra_calls": self.num_extra_calls,
        }


def judge_steps(
    calls: list[ToolCall],
    accepted_tool_steps: list[list[str]],
) -> StepMatch:
    """Match calls against steps, IN ORDER, greedily from the left.

    Each step consumes the earliest not-yet-consumed call whose tool name is in
    that step's candidate set. This generalises the legacy ordered-subsequence
    check to steps that accept several plausible tools: `[["Glob", "Grep"]]`
    passes for either, whereas `check_tools` could only name one.

    Matching is monotonic — a call already spent on step `i` cannot also satisfy
    step `i+1`. That is what keeps a single `Read` from silently satisfying a
    two-step plan that wanted "find it, then read it".

    Calls that match no step are collected in `extra_call_indices` instead of
    being ignored; the contract requires them to be counted against precision.
    """
    step_indices: list[int | None] = [None] * len(accepted_tool_steps)
    consumed: set[int] = set()
    matched: list[int] = []

    # Ordered subsequence matching with alternatives, one step at a time.
    #
    # `cursor` is the earliest call index a step may still consume. It advances
    # only when a step actually matches, so a step that finds nothing does not
    # sabotage the steps after it — (Bash, Read) against (Grep, Read) is one
    # wrong call, not two, and the Read still satisfies step 1.
    #
    # It is NOT reset per step, which is what keeps order meaningful: in
    # (Read, Grep) against (Grep, Read), Grep lands on step 0 and the Read
    # that preceded it can no longer be reached.
    cursor = 0
    for si, accepted in enumerate(accepted_tool_steps):
        accepted_set = set(accepted)
        for ci in range(cursor, len(calls)):
            if calls[ci][0] not in accepted_set:
                continue
            step_indices[si] = ci
            consumed.add(ci)
            matched.append(ci)
            cursor = ci + 1
            break

    extras = [ci for ci in range(len(calls)) if ci not in consumed]
    return StepMatch(
        step_indices=step_indices,
        matched_call_indices=matched,
        extra_call_indices=extras,
    )


# --- Layer-1 argument checking ---


@dataclass(frozen=True)
class ArgCheckResult:
    """Argument correctness at two granularities, from one pass.

    `correct_calls` / `checked_calls` is ArgumentCallAccuracy's ratio;
    `correct_fields` / `checked_fields` is ArgumentFieldAccuracy's. They are
    reported together but computed independently — a call with one wrong field
    of two moves the field ratio and not the call ratio, which is precisely the
    distinction the fused legacy boolean destroyed.
    """

    correct_calls: int
    checked_calls: int
    correct_fields: int
    checked_fields: int

    @property
    def all_calls_correct(self) -> bool:
        """Legacy-compatible view: every checked call had every field correct."""
        return self.correct_calls == self.checked_calls

    def per_tool_detail(self) -> dict[str, dict[str, int]]:
        return self._detail

    _detail: dict[str, dict[str, int]]

    def to_detail(self) -> dict[str, object]:
        return {
            "correct_calls": self.correct_calls,
            "checked_calls": self.checked_calls,
            "correct_fields": self.correct_fields,
            "checked_fields": self.checked_fields,
            "all_calls_correct": self.all_calls_correct,
            "per_tool": self._detail,
        }


def judge_case_args(
    calls: list[ToolCall],
    expect_args: dict[str, dict[str, str]],
) -> ArgCheckResult:
    """Check declared argument regexes and return per-field *and* per-call counts.

    Denominator rule (contract §5.2): only **matched calls whose arguments the
    case declares** are checked. A tool that the case says nothing about, and a
    declared tool the agent never called, both contribute to neither numerator
    nor denominator. A declared tool that was never called is a missing *step*,
    which `judge_steps` already scores — charging it here as well would let one
    mistake depress two independent metrics.

    Per tool, the **best** call is the one that satisfies the most declared
    fields. A retry that fixes the arguments therefore reads as "the arguments
    were ultimately right", instead of double-charging the first attempt.
    """
    correct_calls = 0
    checked_calls = 0
    correct_fields = 0
    checked_fields = 0
    detail: dict[str, dict[str, int]] = {}

    for tool_name, arg_pats in expect_args.items():
        candidates = [ti for name, ti in calls if name == tool_name]
        if not candidates:
            # Declared but never called: a step-level miss, not an arg miss.
            detail[tool_name] = {
                "correct_fields": 0, "checked_fields": 0, "best_call_correct": 0, "calls": 0,
            }
            continue

        best_correct = 0
        for name, tool_input in calls:
            if name != tool_name:
                continue
            hits = sum(
                1 for arg, pat in arg_pats.items() if _match_arg(tool_input.get(arg), pat)
            )
            best_correct = max(best_correct, hits)

        n_fields = len(arg_pats)
        checked_calls += 1
        checked_fields += n_fields
        correct_fields += best_correct
        if best_correct == n_fields:
            correct_calls += 1
        detail[tool_name] = {
            "correct_fields": best_correct,
            "checked_fields": n_fields,
            "best_call_correct": int(best_correct == n_fields),
            "calls": len(candidates),
        }

    return ArgCheckResult(
        correct_calls=correct_calls,
        checked_calls=checked_calls,
        correct_fields=correct_fields,
        checked_fields=checked_fields,
        _detail=detail,
    )


def check_tools(calls: list[ToolCall], expect_tools: list[str]) -> bool:
    """True if expect_tools appears, in order, as a subsequence of calls.

    Legacy boolean wrapper (each tool is a one-candidate step); new code should
    read `judge_steps()` so extra calls and per-step detail survive.
    """
    return judge_steps(calls, [[t] for t in expect_tools]).all_steps_matched


def check_args(calls: list[ToolCall], expect_args: dict[str, dict[str, str]]) -> bool:
    """True if for every tool in expect_args, at least one call with that tool
    name matches ALL the given regexes on its named arguments.

    Argument values are stringified before matching; missing args fail the check.
    """
    return judge_case_args(calls, expect_args).all_calls_correct


def _match_arg(value: object, pattern: str) -> bool:
    if value is None:
        return False
    return re.search(pattern, str(value), re.IGNORECASE) is not None


# --- Layer-2 deterministic judges ---
# Each takes (sandbox: Path, args: dict[str, Any]) and returns bool.
#
# Every judge here must be able to *fail*. A check whose predicate the fixture
# already satisfies before the agent runs measures nothing; the dataset
# contract tests in tests/unit/eval/test_e2e_cases.py carry the per-case
# mutation variants that prove each judge rejects a broken artifact.


# Judges that run a program. `shell=True` is gone: the command comes from a
# data file under evals/, so a shell string there is an arbitrary-code channel
# for anyone (or any future agent) who can edit that file.
_RUN_TIMEOUT_S = 60


@dataclass(frozen=True)
class CommandSpec:
    """A declared, argv-shaped command.

    `argv` is passed to `subprocess.run` as a **list with shell=False**, so no
    shell parses it: `;`, `&&`, backticks, `$VAR` and globs in an argument stay
    literal data. That closes the *shell injection* channel, which was the
    concrete hole in the previous implementation (a case string like
    `python -c "..."` was executed by cmd.exe / /bin/sh).

    It does NOT make the eval a sandbox. `python -c "<code>"` still runs
    arbitrary Python as the current user, with that user's filesystem, network
    and environment. See `judge_python_test` for the limits we can enforce and
    a plain statement of the residual risk.
    """

    argv: list[str]

    def to_dict(self) -> dict[str, object]:
        return {"argv": list(self.argv)}


def parse_command(args: dict[str, Any], *, judge: str) -> CommandSpec:
    """Read the `command` field of a judge's args as an argv list.

    Accepts a bare string too, because a shell-free command genuinely needs no
    quoting and `"command": "pytest -q"` is the ergonomic way to write it. The
    string is split with `shlex.split` (POSIX rules, no shell expansion), then
    the *result* is validated against the same allowlist as a pre-split list —
    the string form is sugar for the list form, never a second execution path.

    A list with a non-string element, or an empty command, is a case-data bug
    and raises rather than being coerced into something runnable.
    """
    command = args.get("command")
    if isinstance(command, str):
        argv = shlex.split(command, posix=True)
    elif isinstance(command, list) and all(isinstance(a, str) for a in command):
        argv = list(command)
    else:
        raise ValueError(
            f"{judge}: 'command' must be a str or list[str], got {command!r}"
        )
    if not argv:
        raise ValueError(f"{judge}: empty command")
    return CommandSpec(argv=argv)


def _looks_like_a_path(program: str) -> bool:
    """True when the program is addressed by path rather than by name.

    `./pytest`, `/usr/bin/python` and `C:\\Python\\python.exe` all bypass `PATH`
    lookup; the latter two also need no `PATH` entry to resolve. Detecting these
    is what lets `ensure_controlled` reject them by *shape* rather than by
    trying to guess what they resolve to.
    """
    return "/" in program or "\\" in program or ":" in program


def ensure_controlled(
    spec: CommandSpec,
    *,
    judge: str,
    allowed: frozenset[str] | None,
    declared: list[str] | None,
) -> None:
    """Enforce the case's declared allowlist on an argv command.

    Three rules, each closing a different bypass:

    1. The case must declare `allowed_commands`. An undeclared command is
       rejected outright — "no allowlist" must not mean "anything goes", or
       the declaration is decorative.
    2. A program addressed by path (`./tool`, `/usr/bin/x`) is rejected: only
       bare `PATH`-resolved names are allowed, so the allowlist cannot be
       satisfied by shipping a binary next to the fixture.
    3. The **basename** of argv[0] must be admitted by the allowlist, with a
       version suffix tolerated — declaring `python` admits `python3.12` and
       `python.exe`, but not `pythonx`. See `_permitted` for why.

    Note the allowlist constrains the *program*, not the arguments: a case
    declaring `python` can still pass `-c <arbitrary code>`. This is a
    data-provenance control, not a sandbox.
    """
    if not allowed:
        raise ValueError(
            f"{judge}: case declares no 'allowed_commands'; a command judge "
            "without a declared allowlist would run whatever the data file says"
        )
    program = spec.argv[0]
    if _looks_like_a_path(program):
        raise ValueError(
            f"{judge}: command must be a bare executable name resolved via PATH, "
            f"got a path {program!r}"
        )
    names = {Path(p).name for p in (*allowed, *(declared or ()))}
    if not _permitted(Path(program).name, names):
        raise ValueError(
            f"{judge}: {program!r} is not in the case's declared allowed_commands "
            f"{sorted(names)}"
        )


def _permitted(program: str, names: set[str]) -> bool:
    """Whether a program basename is admitted by the declared allowlist.

    A declared `python` must admit `python3`, `python3.12` and (on Windows)
    `python.exe`. Two reasons this matters rather than being a nicety:

    - On Debian/Ubuntu there is often no bare `python` on PATH at all, so an
      exact-match rule would make a declared command unrunnable there while
      passing on the author's machine — a portability bug that shows up as a
      mysterious eval failure, not as a security hole.
    - With an exact rule, a case author hits the wall and "fixes" it by adding
      a wildcard; the version-suffix rule gives them nothing to work around.

    Matching is on a version suffix only (`<name>` or `<name><digits>[.<digits>...]`,
    case-insensitive, `.exe` stripped), so `pythonx` is still rejected — unlike a
    `startswith` rule, which would admit it.
    """
    lowered = program.lower()
    if lowered.endswith(".exe"):
        lowered = lowered[:-4]
    for name in names:
        if lowered == name.lower():
            return True
        if not lowered.startswith(name.lower()):
            continue
        suffix = lowered[len(name):]
        if suffix and all(c.isdigit() or c == "." for c in suffix):
            return True
    return False


def run_declared_command(
    sandbox: Path,
    args: dict[str, Any],
    *,
    judge: str,
    allowed_commands: list[str] | None = None,
    interpreter: str = "python",
    timeout_s: int = _RUN_TIMEOUT_S,
) -> subprocess.CompletedProcess[str]:
    """Run a declared command under the allowlist, with no shell in between.

    Kept as one function so there is exactly one place a subprocess is spawned
    in this module; anything that adds a second is visibly a new execution
    path rather than a quiet variation.
    """
    spec = parse_command(args, judge=judge)
    allow = frozenset(allowed_commands or ())
    ensure_controlled(spec, judge=judge, allowed=allow, declared=allowed_commands)

    argv = list(spec.argv)
    # `python3.12` is a perfectly good way to say `python`, and without this
    # normalisation the docstring above would be false on Debian/Ubuntu where
    # there is no `python` on PATH at all. An explicit `interpreter` overrides.
    if paths_equivalent(argv[0], interpreter):
        executable = sys.executable
        if executable:
            argv[0] = executable

    return subprocess.run(
        argv,
        shell=False,
        cwd=sandbox,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )


def paths_equivalent(program: str, interpreter: str) -> bool:
    """Whether `program` names the Python interpreter `interpreter` names.

    Compares basenames case-insensitively (Windows ships `Python.exe`) after
    matching the interpreter's stem, so `python`, `python3`, `python3.12` and
    `python.exe` all count while `pytest` does not.
    """
    prog = Path(program).name.lower()
    want = Path(interpreter).name.lower()
    return prog == want or prog.startswith(want)


def judge_file_content(sandbox: Path, args: dict[str, Any]) -> bool:
    path = sandbox / str(args["path"])
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8", errors="replace")
    if "contains" in args and re.search(str(args["contains"]), text, re.IGNORECASE) is None:
        return False
    return not ("not_contains" in args
                and re.search(str(args["not_contains"]), text, re.IGNORECASE) is not None)


def judge_file_exists(sandbox: Path, args: dict[str, Any]) -> bool:
    return (sandbox / str(args["path"])).is_file()


def judge_json_value(sandbox: Path, args: dict[str, Any]) -> bool:
    """Compare a value inside a JSON file against an expected one.

    `path` is a **decoded-key/list-index** path, not a JSON Pointer string:
    ``{"path": ["servers", 0, "port"]}`` is more readable in a JSONL data file
    than ``"/servers/0/port"``, and it cannot be ambiguous about a key that
    literally contains a slash.

    Comparison rules (all deliberate, all tested):
    - exact equality on the decoded value, so ``8080`` (int) does not equal
      ``"8080"`` (str). A judge that accepted either would pass an artifact
      that any strict consumer rejects.
    - ``.0 == 0`` compares equal, because JSON has one number type and a
      writer that emits ``80.0`` has not changed the value.
    - a missing file, unparseable JSON, or a path that does not exist FAILS.
      It never raises out of the judge: a malformed artifact is a failed case,
      not a crashed run.
    """
    path = sandbox / str(args["path"])
    if not path.is_file():
        return False
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return False
    try:
        actual = _at_path(document, args.get("key_path", []))
    except (KeyError, IndexError, TypeError):
        return False
    return _json_equal(actual, args.get("equals"))


def _at_path(document: Any, key_path: Any) -> Any:
    if not isinstance(key_path, list):
        raise TypeError(f"key_path must be a list, got {key_path!r}")
    node = document
    for key in key_path:
        node = node[int(key)] if isinstance(node, list) else node[str(key)]
    return node


def _json_equal(actual: Any, expected: Any) -> bool:
    if isinstance(actual, bool) or isinstance(expected, bool):
        return actual is expected
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return float(actual) == float(expected)
    return bool(actual == expected)


def judge_line_set_equals(sandbox: Path, args: dict[str, Any]) -> bool:
    """Compare the SET of non-empty, stripped lines of a file to an expectation.

    Set semantics, not sequence: the contract for these cases is "these files
    each got their own line", and a differently-ordered listing is the same
    answer. Blank lines and trailing whitespace are noise from whichever tool
    wrote the file, so they are dropped before comparing.

    Duplicates are collapsed rather than counted, which is the one place this
    is a weaker assertion than `line_set_equals`'s name suggests — a case that
    needs to catch a line written twice should assert on file content, and
    ``duplicates_allowed=False`` exists for when the count matters.

    An optional ``not_contains`` fails any line matching it, so a case can
    assert "exactly these, and none of the junk the agent might leave behind".
    """
    path = sandbox / str(args["path"])
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = {ln.strip() for ln in text.splitlines() if ln.strip()}
    expected = {str(ln).strip() for ln in args.get("equals", []) if str(ln).strip()}
    if lines != expected:
        return False
    if not args.get("duplicates_allowed", True):
        kept = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if len(kept) != len(lines):
            return False
    not_contains = args.get("not_contains")
    if not_contains is not None:
        return not any(re.search(str(not_contains), ln, re.IGNORECASE) for ln in lines)
    return True


def judge_python_test(sandbox: Path, args: dict[str, Any]) -> bool:
    """Run a declared Python test inside the sandbox and require exit code 0.

    This is `command_ok` with a narration. `path` and `test` are recorded in
    the detail so a failure report says *which* test failed without a human
    re-deriving it from the raw command, and `scope` documents the claim being
    made.

    **What this is not.** It is not a sandbox, and the case data is not trusted
    input. `python -c` and `python <file>` both execute code the eval file
    names, as the current user: the process can read the host filesystem, open
    sockets, and mutate anything the user can. The controls here are:

    - no shell (`shell=False`, argv list), so arguments cannot break out into
      further commands;
    - argv[0] must be a bare `PATH`-resolved name in the case's declared
      `allowed_commands`;
    - a wall-clock timeout, so a hung test cannot stall a run.

    Anything stronger — a container, a dedicated OS user, a network namespace —
    is a deployment concern and is **not** implemented here. Freezing the
    dataset does not fix this either: whoever can edit `evals/*.jsonl` can
    point a command judge at arbitrary code, and the only real defence is that
    the eval file is a reviewed, version-controlled artifact. Say that plainly
    rather than calling this "sandboxed".
    """
    proc = run_declared_command(
        sandbox,
        args,
        judge="python_test",
        allowed_commands=args.get("allowed_commands"),
        interpreter=str(args.get("interpreter", "python")),
        timeout_s=int(args.get("timeout_s", _RUN_TIMEOUT_S)),
    )
    return proc.returncode == 0


def judge_command_ok(sandbox: Path, args: dict[str, Any]) -> bool:
    """Run a declared command and require exit code 0 (shell-free)."""
    proc = run_declared_command(
        sandbox, args, judge="command_ok",
        allowed_commands=args.get("allowed_commands"),
        timeout_s=int(args.get("timeout_s", _RUN_TIMEOUT_S)),
    )
    return proc.returncode == 0


def judge_command_output_contains(sandbox: Path, args: dict[str, Any]) -> bool:
    """Run a declared command and search its stdout (shell-free)."""
    proc = run_declared_command(
        sandbox, args, judge="command_output_contains",
        allowed_commands=args.get("allowed_commands"),
        timeout_s=int(args.get("timeout_s", _RUN_TIMEOUT_S)),
    )
    return re.search(str(args["contains"]), proc.stdout, re.IGNORECASE) is not None


def judge_directory_snapshot(sandbox: Path, args: dict[str, Any]) -> bool:
    """Compare the sandbox's file set (and optionally file sizes) to a snapshot.

    `path` picks the subtree (default: the sandbox root); `equals` is the
    expected set of POSIX-style relative paths. ``files_exact`` (default True)
    requires the sets to match exactly, so an agent that leaves a stray scratch
    file behind fails — that is usually the point of a "clean up after
    yourself" case. ``files_contains`` is the weaker subset form for cases
    where extra files are legitimate.

    ``min_sizes`` catches the "empty file passes a file_exists check" family of
    vacuous passes: a snapshot that only names paths cannot tell a written
    artifact from a touched one. Sizes are a floor, not an equality, because
    the exact byte count of generated prose is not something a case should
    pin.

    Directories are ignored entirely — agent harnesses create and remove
    scratch directories freely, and asserting on them would make the check
    fail for reasons unrelated to the task.
    """
    root = sandbox / str(args.get("path", "."))
    if not root.is_dir():
        return False
    present = sorted(
        p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()
    )
    expected = sorted(str(p) for p in args.get("equals", []))
    if args.get("files_exact", True):
        if present != expected:
            return False
    else:
        wanted = set(args.get("files_contains", []))
        if not wanted <= set(present):
            return False
    for rel, min_size in args.get("min_sizes", {}).items():
        candidate = root / str(rel)
        if not candidate.is_file() or candidate.stat().st_size < int(min_size):
            return False
    return True


_JUDGES: dict[str, Any] = {
    "file_content": judge_file_content,
    "file_exists": judge_file_exists,
    "command_ok": judge_command_ok,
    "command_output_contains": judge_command_output_contains,
    "json_value": judge_json_value,
    "line_set_equals": judge_line_set_equals,
    "python_test": judge_python_test,
    "directory_snapshot": judge_directory_snapshot,
}


def judge_case(fn_name: str, sandbox: Path, args: dict[str, Any]) -> bool:
    """Dispatch a named judge against a sandbox directory."""
    fn = _JUDGES.get(fn_name)
    if fn is None:
        raise ValueError(f"unknown judge fn: {fn_name!r} (known: {sorted(_JUDGES)})")
    return bool(fn(sandbox, args))


def case_passed(
    checks: list[dict[str, Any]],
    sandbox: Path,
    *,
    mode: str = "all",
) -> tuple[bool, list[dict[str, Any]]]:
    """Evaluate every check of an E2E case and return (passed, per-check detail).

    Default mode is **all**: a case passes only when every check passes
    (contract §5.1). "any" exists for genuinely disjunctive cases and is not a
    way to soften a flaky expectation — under "any" the other checks cannot
    change the outcome, which the detail makes visible.

    A check that raises (a malformed judge name, an undeclared command) is
    recorded as failed with the message, not propagated. One broken check must
    not abort a 40-case run whose other 39 cases are still measurable; the
    per-check detail is what makes the breakage diagnosable afterwards.

    Returns the per-check list in case order so a failure report can point at
    the exact assertion that broke rather than a single opaque boolean.
    """
    if mode not in ("all", "any"):
        raise ValueError(f"checks_mode must be 'all' or 'any', got {mode!r}")

    results: list[dict[str, Any]] = []
    for check in checks:
        fn_name = str(check.get("fn"))
        args = check.get("args") or {}
        try:
            ok = judge_case(fn_name, sandbox, args)
            error = None
        except Exception as exc:  # a broken check is a failed check, not a crash
            ok = False
            error = f"{type(exc).__name__}: {exc}"
        results.append({"fn": fn_name, "args": args, "passed": ok, "error": error})

    outcomes = [bool(r["passed"]) for r in results]
    passed = any(outcomes) if mode == "any" else all(outcomes)
    return passed, results
