"""Unit tests for the four tool-calling metrics, computed in the runner.

契约 §5.2:四个指标的分母互相独立,且每个都必须能从 raw.jsonl 单独重算.
这一组测试就是对「能重算」这件事的断言——用同一个 CaseResult 的
to_raw_dict() 输出走一遍聚合,结果必须与跑分时算出来的一致.
"""

from __future__ import annotations

from longline.eval.metrics import Ratio
from longline.eval.report import aggregate
from longline.eval.runner import CaseResult
from longline.eval.trajectory import ToolExecution
from longline.eval.types import ToolCallCase


def _case(**kw: object) -> ToolCallCase:
    base: dict[str, object] = {
        "id": "ts-001", "type": "tool_call", "task": "t", "accepted_tool_steps": [["Read"]],
    }
    base.update(kw)
    return ToolCallCase.from_dict(base)


def _result(
    *,
    case_id: str,
    calls: list[tuple[str, dict]] | None = None,
    executions: list[ToolExecution] | None = None,
    tags: list[str] | None = None,
) -> CaseResult:
    return CaseResult(
        case_id=case_id,
        case_type="tool_call",
        passed=True,
        tool_calls=list(calls or []),
        tool_executions=list(executions or []),
        tags=list(tags or ["blind"]),
    )


class TestFourMetricsHaveIndependentDenominators:
    def test_selection_accuracy_counts_only_blind_cases(self) -> None:
        # 12 条 instruction-following 用例必须被分母排除(契约 §5.2 排除条件).
        blind = [_result(case_id="b1", tags=["blind"]), _result(case_id="b2", tags=["blind"])]
        instr = [_result(case_id="i1", tags=["instruction-following"])]
        for r in (*blind, *instr):
            r.detail = {"steps": {"all_steps_matched": r.case_id == "b1"}}
        report = aggregate([*blind, *instr])
        assert report.tool_selection_case_accuracy == Ratio(1, 2)

    def test_precision_denominator_is_every_call_including_extras(self) -> None:
        # 额外调用不免费:它必须留在分母里(契约 §5.2 关键规则).
        r = _result(
            case_id="b1",
            calls=[("Read", {}), ("Bash", {"command": "ls"}), ("Bash", {"command": "pwd"})],
        )
        r.detail = {"steps": {"all_steps_matched": True, "num_extra_calls": 2}}
        report = aggregate([r])
        assert report.tool_call_precision == Ratio(1, 3)

    def test_argument_call_accuracy_denominator_excludes_unchecked_calls(self) -> None:
        r = _result(case_id="b1")
        r.detail = {"args": {"correct_calls": 2, "checked_calls": 5,
                             "correct_fields": 3, "checked_fields": 7}}
        report = aggregate([r])
        assert report.argument_call_accuracy == Ratio(2, 5)

    def test_argument_field_accuracy_is_its_own_ratio(self) -> None:
        # 同一个 detail 里两个参数指标必须给出不同的分母,不能被合并.
        r = _result(case_id="b1")
        r.detail = {"args": {"correct_calls": 2, "checked_calls": 5,
                             "correct_fields": 3, "checked_fields": 7}}
        report = aggregate([r])
        assert report.argument_field_accuracy == Ratio(3, 7)
        assert report.argument_call_accuracy != report.argument_field_accuracy

    def test_execution_success_is_computed_from_executions_only(self) -> None:
        # 未执行的调用不进分母:请求了但没下发 ≠ 执行失败.
        r = _result(
            case_id="b1",
            calls=[("Read", {}), ("Read", {}), ("Read", {})],
            executions=[
                ToolExecution(tool_id="1", tool_name="Read", is_error=False),
                ToolExecution(tool_id="2", tool_name="Read", is_error=True),
            ],
        )
        report = aggregate([r])
        assert report.tool_execution_rate == Ratio(1, 2)

    def test_all_four_are_recomputable_from_raw_jsonl(self) -> None:
        """核心验收:raw.jsonl 的行本身携带重算所需的全部计数."""
        r = _result(
            case_id="b1",
            calls=[("Read", {}), ("Bash", {"command": "ls"})],
            executions=[ToolExecution(tool_id="1", tool_name="Read", is_error=False)],
        )
        r.detail = {
            "steps": {"all_steps_matched": False, "num_extra_calls": 1},
            "args": {"correct_calls": 1, "checked_calls": 1,
                     "correct_fields": 2, "checked_fields": 2},
        }
        row = r.to_raw_dict()
        # 不依赖任何内存对象,只靠 raw.jsonl 的一行重算四个指标.
        selection = Ratio(int(bool(row["judge_detail"]["steps"]["all_steps_matched"])), 1)  # type: ignore[index]
        precision = Ratio(
            row["num_tool_calls"] - row["judge_detail"]["steps"]["num_extra_calls"],  # type: ignore[index]
            row["num_tool_calls"],
        )
        call_acc = Ratio(
            row["judge_detail"]["args"]["correct_calls"],  # type: ignore[index]
            row["judge_detail"]["args"]["checked_calls"],  # type: ignore[index]
        )
        field_acc = Ratio(
            row["judge_detail"]["args"]["correct_fields"],  # type: ignore[index]
            row["judge_detail"]["args"]["checked_fields"],  # type: ignore[index]
        )
        exec_rate = Ratio(row["num_successful_tool_calls"], row["num_tool_calls_executed"])
        assert selection == Ratio(0, 1)
        assert precision == Ratio(1, 2)
        assert call_acc == Ratio(1, 1)
        assert field_acc == Ratio(2, 2)
        assert exec_rate == Ratio(1, 1)


class TestAggregateZeros:
    def test_no_tool_cases_means_not_measured(self) -> None:
        report = aggregate([])
        assert report.tool_selection_case_accuracy.value is None
        assert report.tool_call_precision.value is None
        assert report.argument_call_accuracy.value is None
        assert report.argument_field_accuracy.value is None

    def test_zero_denominator_is_none_not_zero(self) -> None:
        # 契约红线:没测到就是 None,绝不能写成 0%.
        r = _result(case_id="b1", calls=[("Read", {})])
        r.detail = {"steps": {"all_steps_matched": True, "num_extra_calls": 0},
                    "args": {"correct_calls": 0, "checked_calls": 0,
                             "correct_fields": 0, "checked_fields": 0}}
        report = aggregate([r])
        # 参数没有被检查过 -> 未测到,不是「0% 正确」.
        assert report.argument_call_accuracy.value is None
        assert report.argument_field_accuracy.value is None
        # 选择与 precision 测到了:这一步成了,且这次调用不额外.
        assert report.tool_call_precision.value == 1.0
        assert report.tool_selection_case_accuracy.value == 1.0

    def test_no_tool_calls_at_all_leaves_precision_unmeasured(self) -> None:
        # 模型一次工具都没调:precision 的分母是 0,属于未测到.
        r = _result(case_id="b1")
        r.detail = {"steps": {"all_steps_matched": False, "num_extra_calls": 0}}
        report = aggregate([r])
        assert report.tool_call_precision.value is None
        assert report.tool_call_precision.denominator == 0
