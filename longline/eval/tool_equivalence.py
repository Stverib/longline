"""Which Bash invocations a dedicated tool could have served.

`DedicatedToolPreferenceRate` answers "when a primitive operation had a
dedicated tool, did the agent reach for it?" -- the routing question this round
is about, made observable without waiting for a pass rate to move.

The table is deliberately narrow. Its value is in what it REFUSES to classify:
a command the table cannot judge from the command string alone must fall out of
the denominator entirely, or the rate becomes a number that can be moved by
editing the table instead of by changing agent behaviour.

Boundary, in order:

1. Shell operators split the command (`;`, `|`, `&`, `>`, `<`, a newline). A
   command with any of them is NEUTRAL: the equivalence of the whole is not the
   equivalence of its parts, and `cd src && cat app.py` is a shell script whose
   second half happens to be a read. Judging it as a missed Read would be a
   claim about the string that the string does not support.
2. Only then is the command classified, by its first word -- path-qualified
   names included, so `/usr/bin/grep` is still grep.
3. `sed`/`awk`/`perl` are flag-dependent and classified only when the flag that
   decides the answer is present: `sed -i` writes, `sed -n` reads.

Everything else is NEUTRAL, which includes every build, test, package manager,
VCS and process command (`pytest`, `npm`, `git`, `make`, `python`, `curl`,
`echo`, `mkdir`, `rm`, `mv`, `cp`). Those are exactly the operations the
dedicated tools do NOT cover, and a table that claimed otherwise would be
measuring its own opinions.
"""

from __future__ import annotations

import shlex

# The dedicated tools this module reasons about. A metric counts a call as
# "dedicated" only for these four, so numerator and denominator name the same
# set of primitive operations. `Write` and `NotebookEdit` are absent on purpose:
# no shell command in the table is their equivalent, so including them would
# inflate the rate with calls that never had an alternative.
DEDICATED_TOOLS: frozenset[str] = frozenset({"Read", "Grep", "Glob", "Edit"})

# Any of these makes a command compound. The single characters already cover
# their doubled forms (`&&`, `||`), so listing those too would just be noise.
_SHELL_OPERATORS: tuple[str, ...] = (";", "|", "&", ">", "<", "\n")

# first word -> the dedicated tool that should have been used instead.
_EQUIVALENT: dict[str, str] = {
    "cat": "Read",
    "head": "Read",
    "tail": "Read",
    "less": "Read",
    "more": "Read",
    "nl": "Read",
    "grep": "Grep",
    "rg": "Grep",
    "ag": "Grep",
    "egrep": "Grep",
    "fgrep": "Grep",
    "find": "Glob",
    "ls": "Glob",
    "tree": "Glob",
}

# Flags that decide an ambiguous command's answer, checked in order.
# `sed -i` writes while `sed -n` only reads, so neither can be classified from
# the program name alone -- the flag is the thing that carries the meaning.
_WRITE_FLAGS: frozenset[str] = frozenset({"-i", "--in-place"})
_READ_FLAGS: frozenset[str] = frozenset({"-n", "--quiet", "--silent"})
_FLAG_DEPENDENT: frozenset[str] = frozenset({"sed", "awk", "perl"})


def classify_bash(command: str) -> str | None:
    """The dedicated tool this command should have used, or None if neutral.

    None means "excluded from the measurement", never "counted against the
    agent": the caller must not treat the two as the same.
    """
    text = command.strip()
    if not text:
        return None
    if any(operator in text for operator in _SHELL_OPERATORS):
        return None

    try:
        words = shlex.split(text)
    except ValueError:
        # Unbalanced quotes: the command is malformed, so no claim is made.
        return None
    if not words:
        return None

    head = words[0].rsplit("/", 1)[-1]
    if head in _EQUIVALENT:
        return _EQUIVALENT[head]
    if head in _FLAG_DEPENDENT:
        flags = set(words[1:])
        if flags & _WRITE_FLAGS:
            return "Edit"
        if head == "sed" and flags & _READ_FLAGS:
            return "Read"
    return None
