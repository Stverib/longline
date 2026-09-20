"""Stability classification and redundant-action counting.

The three mixed-case rules OVERLAP, so their order is part of the contract.
`test_content_driven_wins_over_overrun` pins that: e2e-406's passing sequence
is also a subsequence of another passing sequence, so an implementation that
tested `overrun` first would misclassify a case whose real evidence is two
byte-identical sequences with opposite outcomes.
"""

from __future__ import annotations

from longline.eval.runner import CaseResult
from longline.eval.stability import (
    ALWAYS_FAIL,
    CONTENT_DRIVEN,
    MIXED,
    OVERRUN,
    ROUTING,
    SINGLE,
    STABLE,
    case_stability,
    first_action_consistency,
    notebook_edit_substitution,
    redundant_actions,
)


def _run(
    case_id: str, repeat: int, passed: bool, calls: list[str],
    *, tags: list[str] | None = None, variant: str | None = None,
) -> CaseResult:
    return CaseResult(
        case_id=case_id, case_type="e2e", passed=passed, repeat_index=repeat,
        tags=tags or [], variant=variant,
        tool_calls=[(name, {}) for name in calls],
    )


class TestArmsAreNotRepeats:
    """两个臂共用一个 case_id, 但它们不是同一个东西的两次重复.

    配对套件把 single 与 multi 两臂放进**同一份** results 里 (这是它故意
    的: 一张表要同时看到两侧), 而 `case_stability` 只按 `case_id` 分组.
    于是一次运行里 "single 过, multi 挂" 被读成 `mixed(content_driven)` --
    一条关于**稳定性**的结论, 而它实际上是一条关于**架构**的结论, 且样本
    只有 1 次.

    付费探路里 pa-001 正是这样被报成 mixed 的: 它只跑了一次, 两臂各一次.
    `repeat_index` 是重复轴, `variant` 是另一条轴; 把 variant 当 repeat 统计,
    任何多臂报告都会得到这个错数.
    """

    def test_one_run_of_two_arms_is_not_a_mixed_case(self) -> None:
        got = case_stability([
            _run("pa-001", 0, True, ["Read"], variant="single_agent"),
            _run("pa-001", 0, False, ["Read"], variant="multi_agent"),
        ])

        assert len(got) == 2, "the two arms are two things, not one"
        assert {s.kind for s in got} == {SINGLE}, (
            "each arm ran once; one run is not evidence of stability"
        )

    def test_repeats_within_an_arm_still_classify(self) -> None:
        """分组变细之后, 臂内那 3 次重复仍要正常判 stable."""
        got = case_stability([
            _run("pa-001", i, True, ["Read"], variant="single_agent")
            for i in range(3)
        ] + [
            _run("pa-001", i, False, ["Read"], variant="multi_agent")
            for i in range(3)
        ])

        kinds = {s.variant: s.kind for s in got}
        assert kinds == {"single_agent": STABLE, "multi_agent": ALWAYS_FAIL}

    def test_a_variantless_suite_is_unchanged(self) -> None:
        """单臂套件不带 variant, 分组必须与从前完全一致."""
        got = case_stability([
            _run("c", 0, True, ["Read"]),
            _run("c", 1, False, ["Read"]),
        ])

        assert len(got) == 1
        assert got[0].kind == MIXED


class TestMixedCause:
    def test_identical_sequence_with_split_outcome_is_content_driven(self) -> None:
        got = case_stability([
            _run("c", 0, True, ["Glob", "Write", "Write"]),
            _run("c", 1, False, ["Glob", "Write", "Write"]),
        ])[0]

        assert got.kind == MIXED
        assert got.mixed_cause == CONTENT_DRIVEN

    def test_content_driven_wins_over_overrun(self) -> None:
        """e2e-406's shape: an identical pair AND a longer passing sequence."""
        got = case_stability([
            _run("c", 0, True, ["Glob", "Glob", "Write", "Write"]),
            _run("c", 1, False, ["Glob", "Glob", "Write", "Write"]),
            _run("c", 2, True, ["Glob", "Glob", "Bash", "Write", "Write"]),
        ])[0]

        assert got.mixed_cause == CONTENT_DRIVEN

    def test_a_failing_run_extending_a_passing_run_is_overrun(self) -> None:
        """e2e-205's shape: the same work, then one more call."""
        got = case_stability([
            _run("c", 0, True, ["Read", "Read", "Edit", "Bash"]),
            _run("c", 1, False, ["Read", "Read", "Edit", "Bash", "Bash"]),
        ])[0]

        assert got.mixed_cause == OVERRUN

    def test_an_extension_of_a_failing_run_is_not_overrun(self) -> None:
        """The prefix must be a PASSING run, or every long failure looks like one."""
        got = case_stability([
            _run("c", 0, False, ["Read"]),
            _run("c", 1, False, ["Read", "Read"]),
            _run("c", 2, True, ["Write"]),
        ])[0]

        assert got.mixed_cause == ROUTING

    def test_a_later_divergence_is_routing_with_its_index(self) -> None:
        got = case_stability([
            _run("c", 0, True, ["Read", "Glob", "Read", "Bash", "Edit", "Bash"]),
            _run("c", 1, False, ["Read", "Glob", "Read", "Bash", "Read", "Grep"]),
        ])[0]

        assert got.mixed_cause == ROUTING
        assert got.first_divergence == 4

    def test_a_first_call_divergence_reports_index_zero(self) -> None:
        got = case_stability([
            _run("c", 0, True, ["Read", "Glob", "Write"]),
            _run("c", 1, False, ["Bash", "Bash", "Bash"]),
        ])[0]

        assert got.first_divergence == 0


class TestKind:
    def test_all_passing_is_stable(self) -> None:
        got = case_stability([_run("a", 0, True, ["Read"]), _run("a", 1, True, ["Read"])])[0]

        assert got.kind == STABLE
        assert got.mixed_cause is None

    def test_all_failing_is_always_fail(self) -> None:
        got = case_stability([_run("b", 0, False, ["Read"]), _run("b", 1, False, ["Read"])])[0]

        assert got.kind == ALWAYS_FAIL
        assert got.mixed_cause is None

    def test_a_single_run_is_single_not_stable(self) -> None:
        """One run cannot be evidence of stability; saying so would be a lie."""
        got = case_stability([_run("a", 0, True, ["Read"])])[0]

        assert got.kind == SINGLE
        assert got.passes == 1

    def test_counts_are_reported_with_the_kind(self) -> None:
        got = case_stability([
            _run("c", 0, True, ["Read"]),
            _run("c", 1, False, ["Read"]),
            _run("c", 2, True, ["Read"]),
        ])[0]

        assert (got.runs, got.passes) == (3, 2)

    def test_cases_are_sorted_and_grouped_by_id(self) -> None:
        got = case_stability([_run("b", 0, True, ["Read"]), _run("a", 0, True, ["Read"])])

        assert [c.case_id for c in got] == ["a", "b"]


class TestFirstActionConsistency:
    def test_buckets_cases_by_how_many_distinct_openers(self) -> None:
        got = first_action_consistency([
            _run("stable", 0, True, ["Read"]),
            _run("stable", 1, True, ["Read"]),
            _run("stable", 2, True, ["Read"]),
            _run("mixed", 0, True, ["Glob"]),
            _run("mixed", 1, True, ["Grep"]),
            _run("mixed", 2, True, ["Glob"]),
            _run("wild", 0, True, ["Glob"]),
            _run("wild", 1, True, ["Grep"]),
            _run("wild", 2, True, ["Read"]),
        ])

        assert got == {"stable": 1, "mixed": 1, "highly_unstable": 1}

    def test_a_run_with_no_tool_call_is_not_bucketed(self) -> None:
        """An abstention has no first action, so it is not evidence either way."""
        got = first_action_consistency([
            _run("a", 0, True, ["Read"]),
            _run("a", 1, True, []),
        ])

        assert got == {"stable": 0, "mixed": 0, "highly_unstable": 0}


class TestRedundantActions:
    def test_counts_repeat_reads_and_reads_after_write_separately(self) -> None:
        rows = [CaseResult(case_id="c", case_type="e2e", passed=True, tool_calls=[
            ("Read", {"file_path": "a.py"}),
            ("Read", {"file_path": "a.py"}),   # repeated read
            ("Edit", {"file_path": "a.py", "old_string": "x", "new_string": "y"}),
            ("Read", {"file_path": "a.py"}),   # read after write
            ("Read", {"file_path": "b.py"}),   # first read of b, fine
        ])]
        got = redundant_actions(rows)

        assert got["repeated_reads"] == 1
        assert got["read_after_write"] == 1

    def test_the_two_counters_are_disjoint(self) -> None:
        """A read after a write is counted ONCE, as a read after a write.

        Counting it in both would report two units of waste for one call, and
        the two counters are read side by side.
        """
        rows = [CaseResult(case_id="c", case_type="e2e", passed=True, tool_calls=[
            ("Read", {"file_path": "a.py"}),
            ("Write", {"file_path": "a.py"}),
            ("Read", {"file_path": "a.py"}),
        ])]
        got = redundant_actions(rows)

        assert got == {"repeated_reads": 0, "read_after_write": 1}

    def test_reading_two_different_files_is_not_redundant(self) -> None:
        rows = [CaseResult(case_id="c", case_type="e2e", passed=True, tool_calls=[
            ("Read", {"file_path": "a.py"}),
            ("Read", {"file_path": "b.py"}),
        ])]

        assert redundant_actions(rows) == {"repeated_reads": 0, "read_after_write": 0}

    def test_counted_per_run_so_repeats_accumulate(self) -> None:
        rows = [
            CaseResult(case_id="c", case_type="e2e", passed=True, repeat_index=i, tool_calls=[
                ("Read", {"file_path": "a.py"}),
                ("Read", {"file_path": "a.py"}),
            ])
            for i in range(3)
        ]

        assert redundant_actions(rows)["repeated_reads"] == 3


class TestNotebookEditSubstitution:
    def test_counts_edit_on_a_notebook(self) -> None:
        """The measured cause of three of the four notebook failures."""
        rows = [
            CaseResult(
                case_id="ts-nb-02", case_type="tool_call", passed=False,
                tags=["notebook"],
                tool_calls=[
                    ("Read", {"file_path": "analysis.ipynb"}),
                    ("Edit", {"file_path": "analysis.ipynb"}),
                ],
            ),
            CaseResult(
                case_id="ts-nb-01", case_type="tool_call", passed=True,
                tags=["notebook"],
                tool_calls=[("NotebookEdit", {"notebook_path": "analysis.ipynb"})],
            ),
        ]

        assert notebook_edit_substitution(rows) == 1

    def test_counts_a_run_once_however_many_edits_it_made(self) -> None:
        """It is a defect signature per run, not a call count."""
        rows = [CaseResult(case_id="c", case_type="tool_call", passed=False, tool_calls=[
            ("Edit", {"file_path": "n.ipynb"}),
            ("Edit", {"file_path": "n.ipynb"}),
        ])]

        assert notebook_edit_substitution(rows) == 1

    def test_editing_a_normal_file_is_not_a_substitution(self) -> None:
        rows = [CaseResult(case_id="c", case_type="tool_call", passed=True, tool_calls=[
            ("Edit", {"file_path": "src/app.py"}),
        ])]

        assert notebook_edit_substitution(rows) == 0

    def test_a_corrected_attempt_is_a_recovery_not_a_substitution(self) -> None:
        """Edit first, then NotebookEdit, is the agent catching its own mistake.

        Counting it would report a defect where the run succeeded. On the
        previous model's run this loose reading reported 7 substitutions when
        only 4 runs never used NotebookEdit -- and those 4 were exactly the
        failures.
        """
        rows = [CaseResult(case_id="c", case_type="tool_call", passed=True, tool_calls=[
            ("Read", {"file_path": "analysis.ipynb"}),
            ("Edit", {"file_path": "analysis.ipynb"}),
            ("NotebookEdit", {"notebook_path": "analysis.ipynb"}),
            ("Bash", {"command": "python -c 'print(1)'"}),
        ])]

        assert notebook_edit_substitution(rows) == 0

    def test_notebook_edit_before_a_later_edit_still_counts_as_recovered(self) -> None:
        """Order does not matter: what matters is that the right tool was used."""
        rows = [CaseResult(case_id="c", case_type="tool_call", passed=False, tool_calls=[
            ("NotebookEdit", {"notebook_path": "analysis.ipynb"}),
            ("Edit", {"file_path": "analysis.ipynb"}),
        ])]

        assert notebook_edit_substitution(rows) == 0
