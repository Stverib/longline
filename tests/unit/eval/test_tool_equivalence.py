"""Boundary tests for the Bash <-> dedicated-tool equivalence table.

The table's value is entirely in what it REFUSES to classify. A test suite that
only checked the positive cases would pass on a table that classifies every
command, and such a table makes the rate meaningless: every `pytest` run would
count as a missed Read, and the number would move when the table was edited
rather than when the agent changed what it reaches for.
"""

from __future__ import annotations

import pytest

from longline.eval.tool_equivalence import DEDICATED_TOOLS, classify_bash


class TestCommandsWithADedicatedEquivalent:
    @pytest.mark.parametrize(("command", "expected"), [
        ("cat src/config.py", "Read"),
        ("head -30 README.md", "Read"),
        ("tail -n 5 log.txt", "Read"),
        ("sed -n '1,40p' src/app.py", "Read"),
        ("grep -R 'timeout' src/", "Grep"),
        ("rg port src/", "Grep"),
        ("find . -name '*.py'", "Glob"),
        ("ls src/", "Glob"),
        ("sed -i 's/a/b/' src/app.py", "Edit"),
    ])
    def test_classified(self, command: str, expected: str) -> None:
        assert classify_bash(command) == expected

    def test_a_path_qualified_binary_still_classifies(self) -> None:
        """/usr/bin/grep is still grep."""
        assert classify_bash("/usr/bin/grep -R x src/") == "Grep"


class TestNeutralCommands:
    @pytest.mark.parametrize("command", [
        "pytest tests/ -q",
        "python -m pytest",
        "git diff HEAD",
        "npm test",
        "make build",
        "curl https://example.com",
        "echo hello",
        "mkdir -p out",
        "rm -rf .cache",
        "mv a.txt b.txt",
        "cp a.txt b.txt",
        "chmod +x run.sh",
    ])
    def test_not_classified(self, command: str) -> None:
        """A `git diff` is not a missed Read, and counting it as one would let
        the rate be moved by editing the table rather than by changing
        behaviour."""
        assert classify_bash(command) is None


class TestCompoundCommands:
    @pytest.mark.parametrize("command", [
        "cat a.py && grep x a.py",
        "grep x a.py | head -5",
        "cd src && cat app.py",
        "for f in *.py; do cat $f; done",
        "cat a.py > out.txt",
        "grep x a.py || echo none",
        "cat a.py; ls",
        "cat a.py & ls",
        "tail -f log.txt < input",
    ])
    def test_not_classified(self, command: str) -> None:
        """The equivalence of the whole is not the equivalence of its parts.

        `cd src && cat app.py` is a shell script whose second half happens to be
        a read; judging it as a missed Read would be a claim about the string
        that the string does not support.
        """
        assert classify_bash(command) is None


class TestDegenerateInput:
    def test_empty(self) -> None:
        assert classify_bash("") is None

    def test_whitespace_only(self) -> None:
        assert classify_bash("   ") is None

    def test_unbalanced_quotes(self) -> None:
        """A malformed command supports no claim at all."""
        assert classify_bash('cat "unterminated') is None

    def test_a_command_that_is_only_a_path(self) -> None:
        assert classify_bash("/tmp") is None


def test_the_dedicated_set_names_the_tools_the_table_maps_to() -> None:
    """Numerator and denominator must name the same primitive operations.

    `Write` and `NotebookEdit` are deliberately absent: no shell command in the
    table is their equivalent, so counting them as "dedicated" would inflate the
    rate with calls that never had an alternative.
    """
    assert set(DEDICATED_TOOLS) == {"Read", "Grep", "Glob", "Edit"}

    for command in ("cat a.py", "grep x .", "ls", "sed -i s/a/b/ a.py"):
        assert classify_bash(command) in DEDICATED_TOOLS
