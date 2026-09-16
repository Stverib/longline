"""Coverage of the direction no existing test checks.

TestFixturesAreNotPreSatisfied proves a judge is not ALREADY satisfied (no
vacuous-true). The mutation pairs prove a judge REJECTS a broken artifact.

Nothing proves the third direction: that a judge ACCEPTS a correct one. That
gap is what let 25 anchored assertions live underground -- judge_file_content
ran without re.MULTILINE, so `^PORT = 3000$` bound to the whole file and every
correct artifact failed. Every existing guard pointed the other way, so the
suite stayed green while 15 of 40 cases could not be passed by doing the task
correctly.

This module does not synthesise correct artifacts per case (that would be a
second, independent model of every task, and a wrong one would be worse than no
test). What it does is plant a probe: for a set of representative *shapes* --
the exact shapes the dead assertions used -- it builds the artifact a correct
agent would produce and asserts the judge accepts it. Shape coverage is the
point; the shapes below are drawn from the cases that were actually broken.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from longline.eval.judges import case_passed, judge_case

CASE_FILE = Path(__file__).resolve().parent.parent.parent.parent / "evals" / "e2e.jsonl"


def _cases_by_id() -> dict[str, dict]:
    rows = {}
    for line in CASE_FILE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["id"]] = row
    return rows


class TestCorrectArtifactsAreAccepted:
    """Each probe writes what a correct agent would write and must pass."""

    def test_anchored_line_assertion_accepts_the_line(self, tmp_path: Path) -> None:
        """The e2e-103 shape: `^PORT = 3000$` inside a multi-line file."""
        (tmp_path / "config.py").write_text(
            'HOST = "127.0.0.1"\nPORT = 3000\nDEBUG = True\n', encoding="utf-8"
        )
        ok, detail = case_passed(
            [
                {"fn": "file_content", "args": {"path": "config.py", "contains": r"^PORT = 3000$"}},
                {"fn": "file_content", "args": {"path": "config.py", "contains": r'^HOST = "127\.0\.0\.1"$'}},
            ],
            tmp_path,
        )
        assert ok, detail

    def test_markdown_table_shape_accepts_every_row(self, tmp_path: Path) -> None:
        """The e2e-501..508 shape: several anchored table rows in one file."""
        (tmp_path / "report.md").write_text(
            "| stage | total |\n| alpha | 10 |\n| beta | 15 |\n| gamma | 3 |\n",
            encoding="utf-8",
        )
        checks = [
            {"fn": "file_content", "args": {"path": "report.md", "contains": rf"^\{row}\$"}}
            for row in (
                r"| stage | total |",
                r"| alpha | 10 |",
                r"| beta | 15 |",
                r"| gamma | 3 |",
            )
        ]
        ok, detail = case_passed(checks, tmp_path)
        assert ok, detail

    def test_appended_line_keeps_original_lines(self, tmp_path: Path) -> None:
        """The e2e-102 shape: append one line, preserve the others."""
        (tmp_path / "notes.txt").write_text(
            "original content\nTODO: fill in more here.\nACCOUNT=4021\n", encoding="utf-8"
        )
        ok, detail = case_passed(
            [
                {"fn": "file_content", "args": {"path": "notes.txt", "contains": r"^ACCOUNT=4021$"}},
                {"fn": "file_content", "args": {"path": "notes.txt", "contains": r"^TODO: fill in more here\.$"}},
            ],
            tmp_path,
        )
        assert ok, detail

    def test_negated_anchored_assertion_accepts_clean_file(self, tmp_path: Path) -> None:
        """The e2e-404 shape: `not_contains ^Reviewed$` on a file without it."""
        (tmp_path / "tickets.md").write_text("# Tickets\n\n- one\n- two\n", encoding="utf-8")
        ok, detail = case_passed(
            [
                {"fn": "file_content", "args": {"path": "tickets.md", "contains": r"^# Tickets$"}},
                {"fn": "file_content", "args": {"path": "tickets.md", "not_contains": r"^Reviewed$"}},
            ],
            tmp_path,
        )
        assert ok, detail

    def test_json_value_accepts_the_expected_value(self, tmp_path: Path) -> None:
        """The e2e-407 shape: a numeric field written as a number, not a string."""
        (tmp_path / "version.json").write_text('{"version": "0.1.0", "count": 1}', encoding="utf-8")
        ok, detail = case_passed(
            [{"fn": "json_value", "args": {"path": "version.json", "key_path": ["count"], "equals": 1}}],
            tmp_path,
        )
        assert ok, detail

    def test_line_set_equals_accepts_the_expected_set(self, tmp_path: Path) -> None:
        (tmp_path / "out.txt").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
        ok, detail = case_passed(
            [{"fn": "line_set_equals", "args": {"path": "out.txt", "equals": ["gamma", "alpha", "beta"]}}],
            tmp_path,
        )
        assert ok, detail

    def test_directory_snapshot_accepts_the_expected_tree(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("x\n", encoding="utf-8")
        (tmp_path / "b.txt").write_text("y\n", encoding="utf-8")
        ok, detail = case_passed(
            [{"fn": "directory_snapshot", "args": {"path": ".", "equals": ["a.txt", "b.txt"]}}],
            tmp_path,
        )
        assert ok, detail


class TestEveryAnchorInTheDatasetIsLineAnchored:
    """A dataset-level guard, independent of any single judge implementation.

    For every `contains`/`not_contains` pattern in evals/e2e.jsonl that uses
    `^` or `$`, assert the judge treats it as a LINE anchor. The probes build a
    file where the pattern must match (for `contains`) or must not appear (for
    `not_contains`) on a line that is neither the first nor the last.
    """

    def _patterns(self) -> list[tuple[str, str, str]]:
        out = []
        for cid, row in _cases_by_id().items():
            for check in row.get("checks") or []:
                if check.get("fn") != "file_content":
                    continue
                args = check.get("args") or {}
                for key in ("contains", "not_contains"):
                    pat = args.get(key)
                    if pat and ("^" in pat or pat.endswith("$")):
                        out.append((cid, key, pat))
        return out

    def test_the_dataset_actually_has_anchored_patterns(self) -> None:
        """If this drops to zero the guard below silently stops testing anything."""
        assert len(self._patterns()) >= 20

    def test_anchored_patterns_are_not_whole_file_anchored(self) -> None:
        r"""Each anchored pattern must be satisfiable on an inner line.

        Probes the judge, not the pattern text. For a pattern `^BODY$` it builds
        a file whose middle line is produced by *matching the body against the
        body* -- `re.search(body, body)` -- which yields the literal text the
        pattern would match, with escapes resolved. Placing that on line 2 and
        asking the judge to match the original pattern must succeed.

        Constructing that literal by stripping metacharacters instead is the
        obvious approach and it is wrong: `^ROLLUP_VERSION = "1\\.0"$` contains
        an escaped dot whose literal rendering is `1.0`, not `1\.0`, so a
        strip-based probe reports false failures for every escaped pattern.
        Deriving the text from the regex engine avoids modelling the escaping
        rules by hand.
        """
        misses = []
        for cid, _key, pat in self._patterns():
            stripped = pat[4:] if pat.startswith("(?m)") else pat
            if not stripped.startswith("^") or not stripped.endswith("$"):
                continue
            body = stripped[1:-1]
            try:
                hit = re.search(body, body)
            except re.error:
                continue
            if hit is None:
                continue
            literal = hit.group(0)
            text = f"head\n{literal}\ntail\n"
            if not judge_case("file_content", self._dir(text), {"path": "f.txt", "contains": pat}):
                misses.append((cid, pat, literal))
        assert not misses, f"anchored patterns that did not match an inner line: {misses}"

    @staticmethod
    def _dir(body: str) -> Path:
        import tempfile

        d = Path(tempfile.mkdtemp(prefix="anchor-probe-"))
        (d / "f.txt").write_text(body, encoding="utf-8")
        return d
