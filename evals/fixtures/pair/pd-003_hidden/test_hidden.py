"""Hidden judge for a dependent chain: the final artifacts must agree.

Generated from the same fixture template that wrote `notes/rollout.md`, so the
expected owners cannot drift from the source the chain reads. Asserting the
COUNTS rather than only the names is what distinguishes a chain that carried
values forward from one that invented plausible ones: `alice` owns two sections
and the other two own one each, and a summary that says so had to have counted.
"""

import json
from pathlib import Path

EXPECTED_COUNTS = {"alice": 2, "bob": 1, "carol": 1}


def test_checklist_carries_every_section_and_its_owner():
    entries = json.loads(Path("out/notes/checklist.json").read_text(encoding="utf-8"))
    owners = {
        entry["owner"] for entry in entries.values()
        if isinstance(entry, dict) and "owner" in entry
    }
    assert owners == set(EXPECTED_COUNTS), (
        "checklist.json must carry an owner for every section; step two added "
        "these, so a missing one means the chain did not carry forward"
    )


def test_owners_are_counted_not_guessed():
    counts = json.loads(Path("out/notes/owners.json").read_text(encoding="utf-8"))
    assert counts == EXPECTED_COUNTS


def test_summary_names_every_owner():
    summary = Path("out/notes/summary.md").read_text(encoding="utf-8")
    for owner in EXPECTED_COUNTS:
        assert owner in summary, f"summary.md never mentions {{owner}}"
