"""Integration tests: the recovery suite driven end to end, offline.

These exercise the real wiring rather than one function at a time: the loader
reads the committed `evals/recovery.jsonl`, each case runs against a real
`QueryEngine` with real tools and the production `query_loop`, the Process-Kill
class kills a real child process, and the aggregate produces the two contract
rates with the denominators the contract fixes.

A small offline run over every fault class is the point: a unit test can prove
one injector fires, but only this can prove the six classes coexist in one
report with the right denominators and that nothing in the suite writes outside
its temp directories.

Cost note: this does NOT call the API. `run_recovery_case(model=None)` replaces
the model transport with a scripted one, so the whole file runs in seconds.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from longline.eval.recovery import cases_by_fault, load_recovery_cases
from longline.eval.recovery_runner import (
    CLAUDE_DIR_PREFIX,
    PER_CASE_FIELDS,
    aggregate_recovery,
    run_recovery_case,
    run_recovery_suite,
)

REPO = Path(__file__).resolve().parents[2]
DATASET = REPO / "evals" / "recovery.jsonl"
FIXTURES = REPO / "evals" / "fixtures"


def _hash_tree(root: Path) -> str | None:
    """Digest of a directory's contents, or None when it does not exist."""
    if not root.exists():
        return None
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        if path.is_file():
            digest.update(path.read_bytes())
    return digest.hexdigest()


@pytest.mark.asyncio
async def test_one_run_per_fault_class_produces_the_contract_split() -> None:
    """Every class runs, and the aggregate splits 50/10 the way the contract says."""
    grouped = cases_by_fault(load_recovery_cases(DATASET))
    one_each = [group[0] for group in grouped.values() if group]

    runs = await run_recovery_suite(one_each, api_key="offline", fixtures_dir=FIXTURES)
    assert len(runs) == 6

    summary = aggregate_recovery(runs)
    # One case per class here, so the denominators are the *shape* of the split
    # (five runtime classes vs one resume class), not the contract's 50 and 10.
    assert summary.runtime_recovery_rate.denominator == 5
    assert summary.session_resume_rate.denominator == 1

    for run in runs:
        assert run.fault_injected is True, f"{run.case_id}: the fault never fired"
        assert run.success is True, f"{run.case_id}: {run.judge_detail} {run.notes}"


@pytest.mark.asyncio
async def test_every_row_carries_the_six_contract_fields() -> None:
    grouped = cases_by_fault(load_recovery_cases(DATASET))
    runs = await run_recovery_suite(
        [group[0] for group in grouped.values() if group],
        api_key="offline", fixtures_dir=FIXTURES,
    )
    for run in runs:
        row = run.to_row()
        for field in PER_CASE_FIELDS:
            assert field in row, f"{run.case_id} is missing {field}"

    # And the whole set is JSON-serialisable, which is what makes raw.jsonl a
    # usable source of truth for every number in the summary.
    json.dumps([r.to_row() for r in runs])


@pytest.mark.asyncio
async def test_process_kill_resumes_through_production_functions() -> None:
    """The resume must go through load_session/validate_transcript/snapshot restore."""
    case = cases_by_fault(load_recovery_cases(DATASET))["process_kill"][0]
    run = await run_recovery_case(case, api_key="offline", fixtures_dir=FIXTURES)

    assert run.fault_injected is True
    assert run.checkpoint_loaded is True
    assert run.structural_errors == []
    assert run.task_states == {"b-1a2b3c4d": "killed"}
    assert run.duplicate_persisted_tool_calls == 0
    assert run.side_effect_classification == "clean_resume"
    assert run.success is True


@pytest.mark.asyncio
async def test_the_suite_does_not_touch_the_real_state_directory() -> None:
    """`get_sessions_dir(None)` falls back to `~/.longline`.

    One dropped `claude_dir` argument anywhere in the worker chain would write
    benchmark sessions into the operator's real state, so the digest is compared
    across a full suite run.
    """
    real = Path.home() / ".longline"
    before = _hash_tree(real)

    grouped = cases_by_fault(load_recovery_cases(DATASET))
    await run_recovery_suite(
        [group[0] for group in grouped.values() if group],
        api_key="offline", fixtures_dir=FIXTURES,
    )

    assert _hash_tree(real) == before, f"the real state directory {real} was modified"


@pytest.mark.asyncio
async def test_no_temp_claude_dirs_are_left_behind() -> None:
    import tempfile

    tmp = Path(tempfile.gettempdir())
    before = {p.name for p in tmp.glob(f"{CLAUDE_DIR_PREFIX}*")}

    case = cases_by_fault(load_recovery_cases(DATASET))["process_kill"][0]
    await run_recovery_case(case, api_key="offline", fixtures_dir=FIXTURES)

    after = {p.name for p in tmp.glob(f"{CLAUDE_DIR_PREFIX}*")}
    assert after == before, "a temp claude_dir leaked"


@pytest.mark.asyncio
async def test_no_fixture_was_modified_in_place() -> None:
    """Cases copy fixtures into a sandbox; `evals/fixtures/` must be untouched."""
    digests_before = {
        p: _hash_tree(p) for p in sorted(FIXTURES.iterdir()) if p.is_dir()
    }

    grouped = cases_by_fault(load_recovery_cases(DATASET))
    await run_recovery_suite(
        [group[0] for group in grouped.values() if group],
        api_key="offline", fixtures_dir=FIXTURES,
    )

    digests_after = {
        p: _hash_tree(p) for p in sorted(FIXTURES.iterdir()) if p.is_dir()
    }
    assert digests_after == digests_before


def test_the_worker_runs_as_a_subprocess_module() -> None:
    """The kill path shells out to `-m longline.eval.recovery_worker`."""
    proc = subprocess.run(
        [sys.executable, "-m", "longline.eval.recovery_worker", "--help"],
        capture_output=True, text=True, cwd=str(REPO), timeout=60, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "prepare" in proc.stdout
    assert "resume" in proc.stdout


def test_the_eval_cli_exposes_the_resume_leg() -> None:
    """`--resume` is how the Process-Kill worker restores a session."""
    from longline.eval.cli import parse_args

    args = parse_args(["--resume", "spec.json"])
    assert args.resume == "spec.json"


def test_the_dataset_is_committed_and_readable() -> None:
    """A number whose dataset is missing is not a number (`evals/README.md` §3)."""
    assert DATASET.is_file()
    lines = [ln for ln in DATASET.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 6, "one case definition per fault class"
    for line in lines:
        case = json.loads(line)
        assert case["type"] == "recovery"
        assert case["repeat"] == 10
