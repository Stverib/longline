"""The two side-effect metrics, and the three tool shapes they must tell apart."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from longline.eval.side_effect_journal import (
    KILLED,
    RESUMED,
    SideEffectEntry,
    SideEffectJournal,
    compute_side_effect_metrics,
    input_fingerprint,
    read_journal,
)

if TYPE_CHECKING:
    from pathlib import Path


def _entry(
    seq: int,
    leg: str,
    tool: str,
    input_fp: str,
    *,
    outcome: str = "ok",
    pre: dict[str, str] | None = None,
    post: dict[str, str] | None = None,
) -> SideEffectEntry:
    return SideEffectEntry(
        seq=seq,
        leg=leg,
        tool=tool,
        input_fp=input_fp,
        outcome=outcome,
        pre_state=pre if pre is not None else {"f": "A"},
        post_state=post if post is not None else {"f": "B"},
    )


def test_journal_appends_and_reads_back(tmp_path: Path) -> None:
    journal = SideEffectJournal(tmp_path / "journal.jsonl", KILLED)
    journal.record(
        tool="Edit",
        tool_input={"file_path": "a.py", "old_string": "x"},
        outcome="ok",
        pre_state={"a.py": "A"},
        post_state={"a.py": "B"},
    )
    journal.record(
        tool="Bash",
        tool_input={"command": "echo hi >> NOTES.md"},
        outcome="ok",
        pre_state={"a.py": "B"},
        post_state={"a.py": "B"},
    )
    entries = read_journal(tmp_path / "journal.jsonl")
    assert [e.seq for e in entries] == [1, 2]
    assert [e.leg for e in entries] == [KILLED, KILLED]
    assert entries[0].tool == "Edit"
    assert entries[0].changed_state is True
    assert entries[1].changed_state is False


def test_journal_file_is_valid_jsonl_on_disk(tmp_path: Path) -> None:
    """The journal has to be readable by a process that did not write it."""
    path = tmp_path / "journal.jsonl"
    SideEffectJournal(path, RESUMED).record(
        tool="Bash",
        tool_input={"command": "echo x >> f"},
        outcome="ok",
        pre_state={"f": "A"},
        post_state={"f": "B"},
    )
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 1
    assert json.loads(lines[0])["tool"] == "Bash"


def test_two_legs_append_to_the_same_file(tmp_path: Path) -> None:
    """Both legs write to one journal, distinguished by `leg`. Two files would
    let a leg's entries go missing without anyone noticing."""
    path = tmp_path / "journal.jsonl"
    SideEffectJournal(path, KILLED).record(
        tool="Bash", tool_input={"c": 1}, outcome="ok",
        pre_state={"f": "A"}, post_state={"f": "B"},
    )
    SideEffectJournal(path, RESUMED).record(
        tool="Bash", tool_input={"c": 1}, outcome="ok",
        pre_state={"f": "B"}, post_state={"f": "C"},
    )
    entries = read_journal(path)
    assert [e.leg for e in entries] == [KILLED, RESUMED]


def test_read_journal_skips_a_torn_tail(tmp_path: Path) -> None:
    """A kill mid-append can leave half a line, and refusing to read the whole
    journal because of it would turn the fault under study into a crash."""
    path = tmp_path / "journal.jsonl"
    SideEffectJournal(path, KILLED).record(
        tool="Bash", tool_input={"c": 1}, outcome="ok",
        pre_state={"f": "A"}, post_state={"f": "B"},
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"seq": 2, "leg": "kill')
    entries = read_journal(path)
    assert len(entries) == 1


def test_read_journal_returns_empty_for_a_missing_file(tmp_path: Path) -> None:
    assert read_journal(tmp_path / "nope.jsonl") == []


def test_journal_rejects_an_unknown_leg(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="leg must be"):
        SideEffectJournal(tmp_path / "j.jsonl", "sideways")


def test_input_fingerprint_is_stable_under_key_order() -> None:
    a = input_fingerprint("Edit", {"file_path": "a.py", "old_string": "x"})
    b = input_fingerprint("Edit", {"old_string": "x", "file_path": "a.py"})
    assert a == b


def test_input_fingerprint_distinguishes_tools_and_inputs() -> None:
    assert input_fingerprint("Edit", {"p": 1}) != input_fingerprint("Write", {"p": 1})
    assert input_fingerprint("Edit", {"p": 1}) != input_fingerprint("Edit", {"p": 2})


def test_append_tool_is_both_redundant_and_duplicated() -> None:
    """Bash append: the second run succeeds AND changes the world again."""
    fp = input_fingerprint("Bash", {"command": "echo x >> f"})
    entries = [
        _entry(1, KILLED, "Bash", fp, pre={"f": "A"}, post={"f": "B"}),
        _entry(2, RESUMED, "Bash", fp, pre={"f": "B"}, post={"f": "C"}),
    ]
    metrics = compute_side_effect_metrics(entries)
    assert metrics.denominator == 1
    assert metrics.redundant == 1
    assert metrics.duplicated == 1
    assert metrics.duplicate_side_effect_rate.numerator == 1
    assert metrics.by_tool["Bash"] == {"redundant": 1, "duplicated": 1}


def test_idempotent_write_is_redundant_but_not_duplicated() -> None:
    """Write with identical content: the second run succeeds, the world is
    unchanged. Counting this as a duplicated side effect would inflate the
    rate with work that was merely wasted."""
    fp = input_fingerprint("Write", {"file_path": "a", "content": "x"})
    entries = [
        _entry(1, KILLED, "Write", fp, pre={"a": "A"}, post={"a": "B"}),
        _entry(2, RESUMED, "Write", fp, pre={"a": "B"}, post={"a": "B"}),
    ]
    metrics = compute_side_effect_metrics(entries)
    assert metrics.denominator == 1
    assert metrics.redundant == 1
    assert metrics.duplicated == 0


def test_failing_edit_replay_is_redundant_but_not_duplicated() -> None:
    """Edit with the same old_string: the second run FAILS because the first
    already consumed the match. No second side effect happened."""
    fp = input_fingerprint("Edit", {"old_string": "x"})
    entries = [
        _entry(1, KILLED, "Edit", fp, pre={"a": "A"}, post={"a": "B"}),
        _entry(2, RESUMED, "Edit", fp, outcome="error", pre={"a": "B"}, post={"a": "B"}),
    ]
    metrics = compute_side_effect_metrics(entries)
    assert metrics.redundant == 1
    assert metrics.duplicated == 0


def test_successful_replay_that_changed_nothing_is_not_duplicated() -> None:
    """The state comparison is load-bearing on its own, independent of the
    outcome: `ok` alone must not be enough to call something a duplicate."""
    fp = "fp"
    entries = [
        _entry(1, KILLED, "Bash", fp, pre={"f": "A"}, post={"f": "B"}),
        _entry(2, RESUMED, "Bash", fp, outcome="ok", pre={"f": "B"}, post={"f": "B"}),
    ]
    metrics = compute_side_effect_metrics(entries)
    assert metrics.redundant == 1
    assert metrics.duplicated == 0


def test_clean_resume_has_zero_redundant() -> None:
    entries = [
        _entry(1, KILLED, "Bash", "fp-1", pre={"f": "A"}, post={"f": "B"}),
        _entry(2, RESUMED, "Bash", "fp-2", pre={"f": "B"}, post={"f": "C"}),
    ]
    metrics = compute_side_effect_metrics(entries)
    assert metrics.denominator == 1
    assert metrics.redundant == 0
    assert metrics.duplicated == 0


def test_read_only_tool_is_excluded_from_the_denominator() -> None:
    """A tool that changed nothing was not a side effect, so it cannot be
    counted as one that was duplicated."""
    fp = "fp-read"
    entries = [
        _entry(1, KILLED, "Read", fp, pre={"f": "A"}, post={"f": "A"}),
        _entry(2, RESUMED, "Read", fp, pre={"f": "A"}, post={"f": "A"}),
    ]
    metrics = compute_side_effect_metrics(entries)
    assert metrics.denominator == 0
    assert metrics.redundant == 0
    assert metrics.duplicated == 0
    # "not measured" must never render as 0%.
    assert metrics.duplicate_side_effect_rate.denominator == 0
    assert metrics.duplicate_side_effect_rate.value is None


def test_resumed_leg_entries_alone_are_not_a_duplicate() -> None:
    """Without a killed-leg entry there is nothing to be a duplicate OF."""
    entries = [_entry(1, RESUMED, "Bash", "fp-1", pre={"f": "A"}, post={"f": "B"})]
    metrics = compute_side_effect_metrics(entries)
    assert metrics.denominator == 0
    assert metrics.redundant == 0


def test_double_duplication_counts_twice() -> None:
    entries = [
        _entry(1, KILLED, "Bash", "fp-1", pre={"f": "A"}, post={"f": "B"}),
        _entry(2, RESUMED, "Bash", "fp-1", pre={"f": "B"}, post={"f": "C"}),
        _entry(3, RESUMED, "Bash", "fp-1", pre={"f": "C"}, post={"f": "D"}),
    ]
    metrics = compute_side_effect_metrics(entries)
    assert metrics.redundant == 2
    assert metrics.duplicated == 2
    assert metrics.denominator == 1


def test_a_replay_is_compared_against_the_first_run_not_the_last() -> None:
    """Two killed-leg executions of the same request: the comparison base is
    the FIRST one, because that is where the side effect first happened."""
    fp = "fp-1"
    entries = [
        _entry(1, KILLED, "Bash", fp, pre={"f": "A"}, post={"f": "B"}),
        _entry(2, KILLED, "Bash", fp, pre={"f": "B"}, post={"f": "C"}),
        _entry(3, RESUMED, "Bash", fp, pre={"f": "C"}, post={"f": "D"}),
    ]
    metrics = compute_side_effect_metrics(entries)
    assert metrics.denominator == 2
    assert metrics.redundant == 1
    assert metrics.duplicated == 1
