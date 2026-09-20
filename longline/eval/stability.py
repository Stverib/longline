"""Per-case run stability, and the redundant actions behind an unstable run.

`pass@1` averages over runs; `pass^k` reports whether a case passed every time.
Neither says WHY a case is unreliable, and "the agent is unreliable" is not an
intervention. This module splits the cases that pass sometimes into three
mechanically-derived causes so the next change can target the one that moved:

- `content_driven` -- two runs with byte-identical tool sequences and opposite
  outcomes. No routing policy can move these: the same actions produced
  different files, so the difference is in what was written.
- `overrun` -- the failing run's sequence strictly extends a passing run's. The
  information was already sufficient and the agent kept going.
- `routing` -- the sequences diverge. The recorded first-divergence index is
  the actionable part: index 0 is a first-tool choice, a late index is a
  mid-chain decision, and those are different problems.

The rules OVERLAP and are therefore ORDERED. `content_driven` is checked first
because a failing run can satisfy the overrun prefix test while its real
evidence is an identical sibling -- see `_classify_mixed` and its tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from longline.eval.runner import CaseResult

STABLE = "stable"
MIXED = "mixed"
ALWAYS_FAIL = "always_fail"
SINGLE = "single"

ROUTING = "routing"
OVERRUN = "overrun"
CONTENT_DRIVEN = "content_driven"

# Tool name -> the argument naming the resource it touches. A repeat only
# counts when it names the SAME resource: reading two different files is not
# redundancy, and counting it as such would make a thorough run look wasteful.
_PATH_ARG: dict[str, str] = {
    "Read": "file_path",
    "Write": "file_path",
    "Edit": "file_path",
    "NotebookEdit": "notebook_path",
}

_WRITE_TOOLS: frozenset[str] = frozenset({"Write", "Edit", "NotebookEdit"})


@dataclass
class CaseStability:
    """How one case behaved across its repeats, within ONE variant."""

    case_id: str
    runs: int
    passes: int
    kind: str
    # None for a single-arm suite. Set when a paired suite put two arms in one
    # result list, so the two rows this case now produces are tellable apart.
    variant: str | None = None
    mixed_cause: str | None = None
    first_divergence: int | None = None
    sequences: list[list[str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "variant": self.variant,
            "runs": self.runs,
            "passes": self.passes,
            "kind": self.kind,
            "mixed_cause": self.mixed_cause,
            "first_divergence": self.first_divergence,
        }


def _outcomes(results: Sequence[CaseResult]) -> list[tuple[tuple[str, ...], bool]]:
    """Each repeat as (tool-name sequence, passed), in repeat order."""
    ordered = sorted(results, key=lambda r: r.repeat_index)
    return [(tuple(name for name, _ in r.tool_calls), r.passed) for r in ordered]


def _classify_mixed(seqs: list[tuple[tuple[str, ...], bool]]) -> tuple[str, int | None]:
    """The cause of a case that passes sometimes. The rules are ORDERED.

    Rule 1 before rule 2 is not a preference: a failing run can extend a
    passing run AND have an identical twin, and the twin is the stronger
    evidence. Reversing them misclassifies e2e-406.
    """
    for index, (sequence, passed) in enumerate(seqs):
        for other, other_passed in seqs[index + 1:]:
            if sequence == other and passed != other_passed:
                return CONTENT_DRIVEN, None

    for failing, failing_passed in seqs:
        if failing_passed:
            continue
        for passing, passing_passed in seqs:
            if not passing_passed:
                continue
            if len(failing) > len(passing) and failing[: len(passing)] == passing:
                return OVERRUN, None

    shortest = min(len(sequence) for sequence, _ in seqs)
    for index in range(shortest):
        if len({sequence[index] for sequence, _ in seqs}) > 1:
            return ROUTING, index
    # Same prefix, split outcome, but no identical pair and no extension: the
    # runs differ only past the shortest one, so there is no index to report.
    return ROUTING, None


def case_stability(results: Sequence[CaseResult]) -> list[CaseStability]:
    """One `CaseStability` per (case id, variant), sorted by id then variant.

    The variant is part of the key, not decoration. A paired suite deliberately
    puts two arms of the SAME case id into one result list, and grouping by case
    id alone reads their disagreement as instability: one run each of
    "single passed, multi failed" becomes `mixed(content_driven)`, a claim about
    run-to-run flakiness made from a single sample of each. `repeat_index` is
    the repeat axis; `variant` is a different axis, and only the first is
    evidence about stability.

    Suites that run one variant leave it None, so their grouping is unchanged.
    """
    grouped: dict[tuple[str, str | None], list[CaseResult]] = {}
    for result in results:
        grouped.setdefault((result.case_id, result.variant), []).append(result)

    out: list[CaseStability] = []
    for case_id, variant in sorted(grouped, key=lambda k: (k[0], k[1] or "")):
        seqs = _outcomes(grouped[(case_id, variant)])
        passes = sum(1 for _, passed in seqs if passed)
        cause: str | None = None
        divergence: int | None = None
        if len(seqs) < 2:
            # One run is not evidence of stability, and saying `stable` here
            # would flatter every single-repeat run in the suite.
            kind = SINGLE
        elif passes == len(seqs):
            kind = STABLE
        elif passes == 0:
            kind = ALWAYS_FAIL
        else:
            kind = MIXED
            cause, divergence = _classify_mixed(seqs)
        out.append(CaseStability(
            case_id=case_id,
            variant=variant,
            runs=len(seqs),
            passes=passes,
            kind=kind,
            mixed_cause=cause,
            first_divergence=divergence,
            sequences=[list(sequence) for sequence, _ in seqs],
        ))
    return out


def first_action_consistency(results: Sequence[CaseResult]) -> dict[str, int]:
    """How many cases opened with the same tool every time.

    Routing instability is what this round tries to move, and a pass rate
    cannot see it: a case that passes 3/3 by three different routes is stable
    in outcome and unstable in policy. Buckets are per CASE, not per run, so a
    suite with more repeats does not swell the counts.
    """
    grouped: dict[str, list[list[str]]] = {}
    for result in results:
        grouped.setdefault(result.case_id, []).append(
            [name for name, _ in result.tool_calls]
        )

    buckets = {"stable": 0, "mixed": 0, "highly_unstable": 0}
    for runs in grouped.values():
        firsts = [names[0] for names in runs if names]
        if len(firsts) != len(runs) or not firsts:
            # A run that called no tool at all has no first action, so it is
            # not evidence either way and the case is left out rather than
            # scored as if it had opened with something.
            continue
        distinct = len(set(firsts))
        if distinct == 1:
            buckets["stable"] += 1
        elif distinct == 2:
            buckets["mixed"] += 1
        else:
            buckets["highly_unstable"] += 1
    return buckets


def redundant_actions(results: Sequence[CaseResult]) -> dict[str, int]:
    """Calls that added no information: a re-read, or a read after a write.

    The two counters are DISJOINT. A read of a file this run already wrote is
    counted once, as a read-after-write, not also as a repeat -- the counters
    are read side by side, and one call must not appear as two units of waste.
    """
    repeated_reads = 0
    read_after_write = 0
    for result in results:
        seen_reads: set[str] = set()
        written: set[str] = set()
        for name, tool_input in result.tool_calls:
            arg = _PATH_ARG.get(name)
            if arg is None:
                continue
            path = str(tool_input.get(arg, ""))
            if not path:
                continue
            if name == "Read":
                if path in written:
                    read_after_write += 1
                elif path in seen_reads:
                    repeated_reads += 1
                seen_reads.add(path)
            elif name in _WRITE_TOOLS:
                written.add(path)
    return {"repeated_reads": repeated_reads, "read_after_write": read_after_write}


def notebook_edit_substitution(results: Sequence[CaseResult]) -> int:
    """Runs that edited a `.ipynb` with Edit and NEVER used NotebookEdit.

    "Instead of", not "as well as". A run that reached for Edit first and then
    corrected itself with NotebookEdit is a recovery, and counting it reports a
    defect where the agent actually caught its own mistake.

    Measured on the previous model's run: 7 runs touched a notebook with Edit,
    but only 4 of them never used NotebookEdit -- and those 4 are exactly the
    notebook failures. The loose reading inflated the count by 75% and measured
    a routing wobble rather than the defect this metric is named for.
    """
    count = 0
    for result in results:
        touched_with_edit = False
        used_notebook_edit = False
        for name, tool_input in result.tool_calls:
            if name == "NotebookEdit":
                used_notebook_edit = True
            elif name == "Edit" and str(tool_input.get("file_path", "")).endswith(".ipynb"):
                touched_with_edit = True
        if touched_with_edit and not used_notebook_edit:
            count += 1
    return count
