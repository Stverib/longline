"""Unit tests for longline/eval/cli.py — argument parsing + wiring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from longline.eval import cli
from longline.eval.types import ToolCallCase


def test_parse_known_args_defaults(tmp_path: Path) -> None:
    ns = cli.parse_args(["--case-file", str(tmp_path / "c.jsonl")])
    assert ns.type == "all"
    assert ns.model == "claude-sonnet-4-20250514"
    assert ns.case_file == str(tmp_path / "c.jsonl")
    assert ns.max_cases is None
    assert ns.out_dir is not None


def test_parse_args_type_filter() -> None:
    ns = cli.parse_args(["--type", "e2e", "--model", "claude-haiku-4-5-20251001", "--max-cases", "3"])
    assert ns.type == "e2e"
    assert ns.model == "claude-haiku-4-5-20251001"
    assert ns.max_cases == 3


def test_split_cases_by_type() -> None:
    from longline.eval.types import E2ECase, ToolCallCase

    cases = [
        ToolCallCase(id="a", task="t"),
        E2ECase(id="b", task="t"),
    ]
    tc, e2e = cli.split_cases(cases)
    assert [c.id for c in tc] == ["a"]
    assert [c.id for c in e2e] == ["b"]


# --- Task 1: new flags, backwards compatible with the legacy ones ---


def test_legacy_flags_are_unchanged() -> None:
    ns = cli.parse_args(["--type", "all", "--fixtures-dir", "f", "--out-dir", "o", "--md"])
    assert ns.type == "all"
    assert ns.fixtures_dir == "f"
    assert ns.out_dir == "o"
    assert ns.md is True


def test_new_flag_defaults() -> None:
    ns = cli.parse_args([])
    assert ns.suite is None
    assert ns.variant is None
    assert ns.repeats == 1
    assert ns.run_id is None
    assert ns.keep_sandbox_on_failure is False


def test_repeats_zero_is_rejected() -> None:
    with pytest.raises(SystemExit):
        cli.parse_args(["--repeats", "0"])


def test_new_flags_parsed() -> None:
    ns = cli.parse_args([
        "--suite", "e2e", "--variant", "candidate", "--repeats", "3",
        "--run-id", "run-42", "--keep-sandbox-on-failure",
    ])
    assert ns.suite == "e2e"
    assert ns.variant == "candidate"
    assert ns.repeats == 3
    assert ns.run_id == "run-42"
    assert ns.keep_sandbox_on_failure is True


def test_apply_suite_sets_type_and_default_case_file() -> None:
    ns = cli.parse_args([])
    cli._apply_suite(ns)
    assert ns.type == "all"
    assert Path(ns.case_file).name == "tool_calls.jsonl"


def test_apply_suite_e2e_switches_case_file() -> None:
    ns = cli.parse_args(["--suite", "e2e"])
    cli._apply_suite(ns)
    assert ns.type == "e2e"
    assert Path(ns.case_file).name == "e2e.jsonl"


def test_apply_suite_explicit_case_file_wins() -> None:
    # The explicit-flag check reads the argv the parser was given.
    argv = ["--suite", "e2e", "--case-file", "custom.jsonl"]
    ns = cli.parse_args(argv)
    cli._apply_suite(ns, argv)
    assert ns.case_file == "custom.jsonl"
    assert ns.type == "e2e"


def test_apply_suite_explicit_type_wins() -> None:
    argv = ["--suite", "tool_calls", "--type", "e2e"]
    ns = cli.parse_args(argv)
    cli._apply_suite(ns, argv)
    assert ns.type == "e2e"
    assert Path(ns.case_file).name == "tool_calls.jsonl"


def test_apply_suite_legacy_type_only_is_untouched() -> None:
    ns = cli.parse_args(["--type", "tool_call"])
    cli._apply_suite(ns)
    assert ns.type == "tool_call"


def test_apply_suite_none_is_noop() -> None:
    ns = cli.parse_args([])
    before = ns.type
    cli._apply_suite(ns)
    assert ns.type == before


def test_split_cases_by_tag() -> None:
    from longline.eval.types import E2ECase

    cases = [
        E2ECase(id="a", task="t", tags=["blind"]),
        E2ECase(id="b", task="t", tags=["instruction_following"]),
        E2ECase(id="c", task="t"),
    ]
    assert [c.id for c in cli._select_by_tag(cases, "blind")] == ["a"]
    assert [c.id for c in cli._select_by_tag(cases, None)] == ["a", "b", "c"]


def test_default_run_id_is_filesystem_safe_and_unique() -> None:
    a = cli.make_run_id("claude-sonnet-4-20250514", "e2e")
    b = cli.make_run_id("claude-sonnet-4-20250514", "e2e")
    assert "/" not in a and ":" not in a
    assert a.startswith("claude-sonnet-4-20250514_e2e_")
    assert len(a.split("_")) >= 3
    assert a != b  # microsecond + process entropy


# --- run directory layout (evals/README.md §3) ---


def _fake_results() -> list[Any]:
    from longline.eval.runner import CaseResult
    from longline.eval.trajectory import ToolExecution

    return [
        CaseResult(
            case_id="e2e-001", case_type="e2e", passed=True, turns=2,
            input_tokens=10, output_tokens=5, duration_ms=12.5,
            tags=["create"], variant="baseline", repeat_index=0, trial=0,
            tool_executions=[ToolExecution("t1", "Write", False, 0, 1_000_000)],
            tool_calls=[("Write", {"file_path": "a"})],
        ),
        CaseResult(
            case_id="e2e-002", case_type="e2e", passed=False, turns=3,
            input_tokens=20, output_tokens=7, duration_ms=30.0,
            tags=["create"], variant="baseline", repeat_index=0, trial=1,
            error_type="max_turns",
        ),
    ]


def test_write_run_dir_produces_contract_filenames(tmp_path: Path) -> None:
    results = _fake_results()
    run_dir = tmp_path / "run-1"
    metadata = cli.run_metadata(
        run_id="run-1", suite="e2e", variant="baseline",
        model="m", case_file=tmp_path / "c.jsonl", repeat_index=0, repeats_completed=1,
    )
    cli._write_run_dir(run_dir=run_dir, results=results, metadata=metadata)

    assert (run_dir / "raw.jsonl").is_file()
    assert (run_dir / "summary.json").is_file()
    assert (run_dir / "report.md").is_file()


def test_raw_jsonl_is_recomputable_source_of_truth(tmp_path: Path) -> None:
    results = _fake_results()
    run_dir = tmp_path / "run-1"
    cli._write_run_dir(
        run_dir=run_dir, results=results,
        metadata=cli.run_metadata(
            run_id="run-1", suite="e2e", variant=None, model="m",
            case_file=tmp_path / "c.jsonl", repeat_index=0, repeats_completed=1,
        ),
    )
    rows = [json.loads(x) for x in (run_dir / "raw.jsonl").read_text("utf-8").splitlines()]
    assert len(rows) == 2
    assert rows[0]["case_id"] == "e2e-001"
    assert rows[0]["duration_ms"] == 12.5
    assert rows[0]["num_tool_calls_executed"] == 1
    assert rows[0]["num_successful_tool_calls"] == 1
    # The numerator/denominator of the summary must be derivable from these rows.
    assert sum(1 for r in rows if r["passed"]) == 1
    assert len(rows) == 2


def test_summary_json_carries_metadata_and_ratios(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-1"
    cli._write_run_dir(
        run_dir=run_dir, results=_fake_results(),
        metadata=cli.run_metadata(
            run_id="run-1", suite="e2e", variant="candidate", model="m",
            case_file=tmp_path / "c.jsonl", repeat_index=0, repeats_completed=3,
        ),
    )
    summary = json.loads((run_dir / "summary.json").read_text("utf-8"))
    for key in ("run_id", "suite", "variant", "model", "git_sha", "started_at",
                "platform", "python_version", "case_file_sha256",
                "repeat_index", "repeats_completed"):
        assert key in summary["metadata"], key
    metrics = summary["metrics"]
    assert metrics["l2_pass1"]["numerator"] == 1
    assert metrics["l2_pass1"]["denominator"] == 2
    assert metrics["l2_pass1"]["value"] == 0.5
    assert len(metrics["l2_pass1"]["ci95_wilson"]) == 2
    assert metrics["latency_ms"]["p50"] == pytest.approx(21.25)


def test_report_md_prints_numerator_over_denominator(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-1"
    cli._write_run_dir(
        run_dir=run_dir, results=_fake_results(),
        metadata=cli.run_metadata(
            run_id="run-1", suite="e2e", variant=None, model="m",
            case_file=tmp_path / "c.jsonl", repeat_index=0, repeats_completed=1,
        ),
    )
    md = (run_dir / "report.md").read_text("utf-8")
    assert "1/2" in md
    assert "## By category" in md
    assert "max_turns" in md


def test_jsonl_results_round_trip(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-1"
    original = _fake_results()
    cli._write_run_dir(
        run_dir=run_dir, results=original,
        metadata=cli.run_metadata(
            run_id="run-1", suite="e2e", variant="baseline", model="m",
            case_file=tmp_path / "c.jsonl", repeat_index=0, repeats_completed=1,
        ),
    )
    reloaded = cli._load_jsonl_results(run_dir / "raw.jsonl")
    assert [r.case_id for r in reloaded] == ["e2e-001", "e2e-002"]
    assert [r.passed for r in reloaded] == [True, False]
    assert reloaded[0].duration_ms == 12.5
    assert reloaded[1].error_type == "max_turns"


def test_paired_delta_rejects_unaligned_runs() -> None:
    from longline.eval.report import paired_report_delta

    baseline = _fake_results()
    candidate = _fake_results()[:1]
    with pytest.raises(ValueError, match="not aligned"):
        paired_report_delta(baseline, candidate)


def test_paired_delta_on_aligned_runs() -> None:
    from longline.eval.report import paired_report_delta

    baseline = _fake_results()
    candidate = _fake_results()
    candidate[1].passed = True
    out = paired_report_delta(baseline, candidate)
    assert out["n_pairs"] == 2
    assert out["duration_ms"]["per_case"][0]["case_id"] == "e2e-001"


def test_paired_delta_excludes_unmeasured_duration_not_zero() -> None:
    """A None duration means "not measured" and must NOT be counted as 0.0.

    Coercing it to zero would drag the paired mean toward zero while looking
    like a real observation.
    """
    from longline.eval.report import paired_report_delta

    baseline = _fake_results()   # durations 12.5 and 30.0
    candidate = _fake_results()
    baseline[1].duration_ms = None  # case e2e-002 was never timed

    out = paired_report_delta(baseline, candidate)
    assert out["n_pairs"] == 2
    assert out["n_pairs_duration"] == 1
    assert out["n_pairs_duration_excluded"] == 1
    # Only the timed pair survives, and its delta is the real one.
    assert out["duration_ms"]["per_case"] == [{"case_id": "e2e-001", "delta": 0.0}]
    assert out["duration_ms"]["n_pairs"] == 1


def test_paired_delta_unmeasured_duration_does_not_bias_the_mean() -> None:
    """The excluded pair must not pull the mean toward zero."""
    from longline.eval.report import paired_report_delta

    baseline = _fake_results()
    candidate = _fake_results()
    # candidate case 0 is 100ms slower; case 1 was never timed on either side
    candidate[0].duration_ms = 112.5
    baseline[1].duration_ms = None
    candidate[1].duration_ms = None

    out = paired_report_delta(baseline, candidate)
    # With the None coerced to 0.0 the mean would be (100.0 + 0.0) / 2 = 50.0.
    assert out["duration_ms"]["mean"] == pytest.approx(100.0)
    assert out["n_pairs_duration_excluded"] == 1


def test_paired_delta_all_durations_unmeasured_is_not_zero() -> None:
    """If nothing was timed, the duration delta is unmeasured — not 0.0."""
    from longline.eval.report import paired_report_delta

    baseline = _fake_results()
    candidate = _fake_results()
    for r in [*baseline, *candidate]:
        r.duration_ms = None

    out = paired_report_delta(baseline, candidate)
    assert out["n_pairs_duration"] == 0
    assert out["n_pairs_duration_excluded"] == 2
    assert out["duration_ms"]["mean"] is None
    assert out["duration_ms"]["per_case"] == []


def test_paired_delta_turns_unaffected_by_missing_duration() -> None:
    """`turns` is always known, so it is never excluded."""
    from longline.eval.report import paired_report_delta

    baseline = _fake_results()
    candidate = _fake_results()
    for r in [*baseline, *candidate]:
        r.duration_ms = None

    out = paired_report_delta(baseline, candidate)
    assert out["turns"]["n_pairs"] == 2


# --- Task 2: suite / profile / blind-vs-instruction split ---


class TestToolSelectionSuiteWiring:
    def test_tool_selection_suite_selects_its_case_file(self) -> None:
        args = cli.parse_args(["--suite", "tool_selection"])
        cli._apply_suite(args, ["--suite", "tool_selection"])
        assert args.case_file.endswith("tool_selection.jsonl")
        assert args.type == "tool_call"

    def test_legacy_tool_calls_suite_unchanged(self) -> None:
        args = cli.parse_args(["--suite", "tool_calls"])
        cli._apply_suite(args, ["--suite", "tool_calls"])
        assert args.case_file.endswith("tool_calls.jsonl")

    def test_blind_only_selects_blind_tagged_cases(self) -> None:
        cases: list[ToolCallCase] = [
            ToolCallCase(id="a", task="t", tags=["blind"]),
            ToolCallCase(id="b", task="t", tags=["instruction-following"]),
            ToolCallCase(id="c", task="t", tags=["blind", "read"]),
        ]
        assert [c.id for c in cli._select_by_tag(cases, cli.BLIND_TAG)] == ["a", "c"]
        assert [c.id for c in cli._select_by_tag(cases, cli.INSTRUCTION_FOLLOWING_TAG)] == ["b"]

    def test_profile_for_case_maps_family_tags(self) -> None:
        assert cli.profile_for_case(ToolCallCase(id="a", task="t", tags=["web"]), "core") == "web"
        assert cli.profile_for_case(ToolCallCase(id="a", task="t", tags=["task"]), "core") == "task"
        assert cli.profile_for_case(
            ToolCallCase(id="a", task="t", tags=["notebook"]), "core"
        ) == "notebook"
        assert cli.profile_for_case(ToolCallCase(id="a", task="t", tags=["read"]), "core") == "core"

    def test_profile_for_case_multi_tag_wins(self) -> None:
        # 多工具路径需要所有相关族同时可用。
        assert cli.profile_for_case(
            ToolCallCase(id="a", task="t", tags=["read", "multi"]), "core"
        ) == "all"

    def test_profile_for_case_falls_back_to_default(self) -> None:
        assert cli.profile_for_case(ToolCallCase(id="a", task="t", tags=["legacy"]), "web") == "web"
        assert cli.profile_for_case(ToolCallCase(id="a", task="t"), "core") == "core"

    def test_tool_profile_flag_defaults_to_core(self) -> None:
        assert cli.parse_args([]).tool_profile == "core"
        assert cli.parse_args(["--tool-profile", "all"]).tool_profile == "all"

    def test_blind_and_instruction_flags_default_off(self) -> None:
        args = cli.parse_args([])
        assert args.blind_only is False
        assert args.instruction_only is False


# --- Task 4: the compression suite -----------------------------------------


class TestCompressionSuite:
    def test_suite_preset_selects_the_compression_case_file(self) -> None:
        ns = cli.parse_args(["--suite", "compression"])
        cli._apply_suite(ns, ["--suite", "compression"])
        assert ns.case_file.endswith("compression.jsonl")

    def test_compression_is_a_distinct_type_not_e2e(self) -> None:
        """A compression case carries `history`/`key_facts`; `E2ECase` has no shape for them.

        Routing it through `--type e2e` would load it as a bare E2E case and
        silently drop the transcript, which is the only thing the suite measures.
        """
        ns = cli.parse_args(["--suite", "compression"])
        cli._apply_suite(ns, ["--suite", "compression"])
        assert ns.type == "compression"

    def test_explicit_case_file_still_wins(self) -> None:
        argv = ["--suite", "compression", "--case-file", "custom.jsonl"]
        ns = cli.parse_args(argv)
        cli._apply_suite(ns, argv)
        assert ns.case_file == "custom.jsonl"

    def test_compression_summary_is_written_to_summary_json(self, tmp_path: Path) -> None:
        """Contract §3: a number that lives only in report.md is not a number."""
        from longline.eval.compression_runner import aggregate_compression
        from tests.unit.eval.test_report import _fake_compression_run

        run = _fake_compression_run("cc-1", True, True, 1000, 500, 5)
        summary = aggregate_compression([run])
        cli._write_run_dir(
            run_dir=tmp_path / "run",
            results=[],
            metadata={"run_id": "r", "variant": None},
            compression=summary,
        )
        payload = json.loads((tmp_path / "run" / cli.SUMMARY_NAME).read_text(encoding="utf-8"))
        assert "compression" in payload
        assert payload["compression"]["token_units"] == "estimated"
        assert payload["compression"]["key_info_retention"]["denominator"] == 5

    def test_compression_report_lands_in_report_md(self, tmp_path: Path) -> None:
        from longline.eval.compression_runner import aggregate_compression
        from tests.unit.eval.test_report import _fake_compression_run

        run = _fake_compression_run("cc-1", True, True, 1000, 500, 5)
        cli._write_run_dir(
            run_dir=tmp_path / "run",
            results=[],
            metadata={"run_id": "r", "variant": None},
            compression=aggregate_compression([run]),
        )
        md = (tmp_path / "run" / cli.REPORT_NAME).read_text(encoding="utf-8")
        assert "CompressionRatio" in md


# --- Task 6: the latency suite ---------------------------------------------


class TestLatencySuite:
    def test_suite_preset_selects_the_latency_layer(self) -> None:
        ns = cli.parse_args(["--suite", "latency"])
        cli._apply_suite(ns, ["--suite", "latency"])
        assert ns.type == "latency"

    def test_explicit_type_still_wins(self) -> None:
        argv = ["--suite", "latency", "--type", "e2e"]
        ns = cli.parse_args(argv)
        cli._apply_suite(ns, argv)
        assert ns.type == "e2e"

    def test_latency_knobs_default_to_none_so_the_module_decides(self) -> None:
        """FAILS ON: the CLI hard-coding a sample count that drifts from the contract."""
        ns = cli.parse_args(["--suite", "latency"])
        assert ns.samples is None
        assert ns.warmups is None
        assert ns.time_scale is None

    def test_latency_knobs_parse(self) -> None:
        ns = cli.parse_args([
            "--suite", "latency", "--samples", "35", "--warmups", "2", "--time-scale", "3.5",
        ])
        assert ns.samples == 35
        assert ns.warmups == 2
        assert ns.time_scale == 3.5

    @staticmethod
    def _stub_suite(monkeypatch: pytest.MonkeyPatch) -> None:
        """Make `run_latency_suite` return a canned pair instead of driving one.

        The wiring these tests check -- which files land on disk, what shape the
        rows have, whether `raw.jsonl` alone recomputes the reduction -- does not
        depend on the timing being real. Running the genuine suite at the
        shipped scale costs ~60-90 s per test, and three of those turned this
        file into a five-minute hazard.

        Exactly ONE test below runs the real suite
        (`test_a_real_run_lands_on_disk_and_recomputes`), so the wiring is still
        verified end to end somewhere; everything else here checks file layout
        and arithmetic, which a canned summary checks just as well.
        """
        from longline.eval import latency_runner as lr

        async def canned(
            cases: object, *, samples: int = 40, warmups: int = 5,
            time_scale: float = 20.0, clock: object = None,
        ) -> tuple[object, list[object]]:
            case = next(iter(cases))  # type: ignore[call-overload]
            truth = case.truth(time_scale)
            rows = []
            for _ in range(samples):
                for variant in ("buffered", "streaming"):
                    start_ns = (
                        truth.response_ns if variant == "buffered"
                        else truth.streaming_tool_start_ns
                    )
                    rows.append(lr.Sample(
                        variant=variant, case_id=case.id,
                        timestamps={
                            "request_start": 0,
                            "tool_block_complete": truth.block_offsets_ns[0],
                            "tool_execute_start": start_ns,
                            "response_complete": truth.response_ns,
                            "tool_execute_end": start_ns + truth.tool_ns,
                            "turn_complete": truth.buffered_turn_ns
                            if variant == "buffered" else truth.streaming_turn_ns,
                        },
                        result_texts=list(case.results()),
                        tool_durations_ms=[truth.tool_ns / 1e6] * case.num_calls,
                        time_scale=time_scale, grid_lag_ns=[0], release_ns=0, sink_ns=0,
                    ))
            summary = lr.LatencySummary(
                samples_per_arm=samples, warmups_per_arm=warmups,
                time_scale=time_scale,
                cases=[lr.summarize_case(
                    case, [(rows[i * 2], rows[i * 2 + 1]) for i in range(samples)],
                    warmups=warmups, time_scale=time_scale,
                )],
            )
            return summary, rows

        monkeypatch.setattr(lr, "run_latency_suite", canned)

    async def test_run_latency_writes_the_contract_layout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """FAILS ON: a run that prints a number but leaves no `raw.jsonl` behind.

        Contract §3: a figure that is not recomputable from raw.jsonl is not a
        figure. The suite is stubbed -- see `_stub_suite` -- because this test is
        about the FILES, not about the timing.
        """
        self._stub_suite(monkeypatch)
        args = cli.parse_args([
            "--suite", "latency", "--run-id", "lat-test", "--max-cases", "1",
            "--samples", "30", "--warmups", "0",
        ])
        cli._apply_suite(args, ["--suite", "latency"])
        code = await cli._run_latency(
            args, case_file=Path(args.case_file), out_dir=tmp_path, run_id="lat-test",
        )
        assert code == 0
        run_dir = tmp_path / "lat-test"
        rows = [
            json.loads(line)
            for line in (run_dir / cli.RAW_NAME).read_text(encoding="utf-8").splitlines()
        ]
        # 30 samples x 2 arms, and every row tagged as latency rather than E2E.
        assert len(rows) == 60
        assert all("latency" in r["tags"] for r in rows)
        assert {r["variant"] for r in rows} == {"buffered", "streaming"}

        summary = json.loads((run_dir / cli.SUMMARY_NAME).read_text(encoding="utf-8"))
        assert summary["latency"]["reduction_units"] == "ratio_of_durations"
        assert summary["metadata"]["suite"] == "latency"

    async def test_raw_jsonl_alone_recomputes_the_reduction(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The §3 rule, checked as arithmetic rather than as a file's existence."""
        self._stub_suite(monkeypatch)
        args = cli.parse_args([
            "--suite", "latency", "--run-id", "lat-recompute", "--max-cases", "1",
            "--samples", "30", "--warmups", "0",
        ])
        cli._apply_suite(args, ["--suite", "latency"])
        await cli._run_latency(
            args, case_file=Path(args.case_file), out_dir=tmp_path, run_id="lat-recompute",
        )
        run_dir = tmp_path / "lat-recompute"
        rows = [
            json.loads(line)
            for line in (run_dir / cli.RAW_NAME).read_text(encoding="utf-8").splitlines()
        ]
        baseline = [r["tool_start_latency_ms"] for r in rows if r["variant"] == "buffered"]
        candidate = [r["tool_start_latency_ms"] for r in rows if r["variant"] == "streaming"]
        recomputed = (sum(baseline) / len(baseline) - sum(candidate) / len(candidate)) / (
            sum(baseline) / len(baseline)
        )
        summary = json.loads((run_dir / cli.SUMMARY_NAME).read_text(encoding="utf-8"))
        assert summary["latency"]["cases"][0]["reduction"] == pytest.approx(recomputed)

    async def test_a_real_run_lands_on_disk_and_recomputes(self, tmp_path: Path) -> None:
        """The one UNSTUBBED end-to-end latency run in the whole suite.

        Everything else in this class stubs `run_latency_suite`, so without this
        the wiring assertions above could all pass while the real runner wrote
        nothing usable. It drives one case for real at the shipped scale, which
        costs ~60-90 s, and it is deliberately the only test that pays it.

        It also carries the load-bearing claim through the CLI: streaming starts
        its tool earlier on every one of the 30 samples, never later.
        """
        args = cli.parse_args([
            "--suite", "latency", "--run-id", "lat-real", "--max-cases", "1",
            "--samples", "30", "--warmups", "0",
        ])
        cli._apply_suite(args, ["--suite", "latency"])
        await cli._run_latency(
            args, case_file=Path(args.case_file), out_dir=tmp_path, run_id="lat-real",
        )
        run_dir = tmp_path / "lat-real"
        rows = [
            json.loads(line)
            for line in (run_dir / cli.RAW_NAME).read_text(encoding="utf-8").splitlines()
        ]
        assert len(rows) == 60
        baseline = [r["tool_start_latency_ms"] for r in rows if r["variant"] == "buffered"]
        candidate = [r["tool_start_latency_ms"] for r in rows if r["variant"] == "streaming"]
        # Every paired sample favours streaming, on a real run.
        assert all(c < b for b, c in zip(baseline, candidate, strict=True))

        summary = json.loads((run_dir / cli.SUMMARY_NAME).read_text(encoding="utf-8"))
        case = summary["latency"]["cases"][0]
        assert case["reduction"] > 0
        assert case["samples_with_streaming_faster"] == 30
        assert case["samples_with_streaming_slower"] == 0
        # And the report renders it as a ratio of durations, never as `pp`.
        md = (run_dir / cli.REPORT_NAME).read_text(encoding="utf-8")
        assert "ratio of durations" in md

    async def test_missing_run_id_is_refused(self) -> None:
        """FAILS ON: emitting a headline number with no durable record behind it."""
        args = cli.parse_args([
            "--suite", "latency", "--max-cases", "1", "--samples", "30",
            "--warmups", "0",
        ])
        cli._apply_suite(args, ["--suite", "latency"])
        with pytest.raises(SystemExit, match="run-id"):
            await cli._run_latency(
                args, case_file=Path(args.case_file), out_dir=Path("."), run_id=None,
            )

    async def test_a_tag_that_matches_nothing_is_refused(self) -> None:
        """FAILS ON: silently running zero cases and reporting a summary of none."""
        args = cli.parse_args([
            "--suite", "latency", "--run-id", "x", "--tag", "no-such-tag",
        ])
        cli._apply_suite(args, ["--suite", "latency"])
        with pytest.raises(SystemExit, match="no latency cases"):
            await cli._run_latency(
                args, case_file=Path(args.case_file), out_dir=Path("."), run_id="x",
            )
