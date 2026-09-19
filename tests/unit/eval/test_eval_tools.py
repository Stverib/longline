"""Unit tests for longline/eval/eval_tools.py — offline tool profiles.

这些替身的唯一职责是让「工具选择」可测:schema 必须与生产工具逐字段一致,
否则模型看到的选项就和线上不同,选择率也就不是线上那个数字.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from longline.eval.engine_factory import EVAL_TOOL_NAMES
from longline.eval.eval_tools import (
    ALL_EVAL_TOOL_NAMES,
    WEB_FAMILY,
    EvalWebFetchTool,
    EvalWebSearchTool,
    build_tool_profile,
)
from longline.tools.file_read.file_read_tool import FileReadTool
from longline.tools.glob_tool.glob_tool import GlobTool
from longline.tools.grep_tool.grep_tool import GrepTool
from longline.tools.notebook.notebook_edit_tool import NotebookEditTool
from longline.tools.task_tools.task_tools import TaskStore
from longline.tools.web_fetch.web_fetch_tool import WebFetchTool
from longline.tools.web_search.web_search_tool import WebSearchTool


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def _schema_dict(tool: Any) -> dict[str, Any]:
    s = tool.get_schema()
    return {"name": s.name, "description": s.description, "input_schema": s.input_schema}


class TestWebStandInsMatchProductionSchema:
    """schema 一致性是这一节存在的理由,必须逐字段比对,不是抽查."""

    def test_web_search_schema_identical_to_production(self) -> None:
        assert _schema_dict(EvalWebSearchTool()) == _schema_dict(WebSearchTool())

    def test_web_fetch_schema_identical_to_production(self) -> None:
        assert _schema_dict(EvalWebFetchTool()) == _schema_dict(WebFetchTool())

    def test_names_match_production(self) -> None:
        assert EvalWebSearchTool().get_name() == WebSearchTool().get_name() == "WebSearch"
        assert EvalWebFetchTool().get_name() == WebFetchTool().get_name() == "WebFetch"


class TestWebStandInsAreOffline:
    def test_web_search_returns_canned_success_without_network(self) -> None:
        r = _run(EvalWebSearchTool().execute({"query": "python asyncio"}))
        assert r.is_error is False
        assert "python asyncio" in r.content

    def test_web_search_is_deterministic(self) -> None:
        tool = EvalWebSearchTool()
        a = _run(tool.execute({"query": "x"}))
        b = _run(tool.execute({"query": "x"}))
        assert a.content == b.content

    def test_web_search_without_query_is_an_error(self) -> None:
        # 与生产工具同样的入参校验:真实执行失败要能被 ExecutionSuccessRate 看见.
        assert _run(EvalWebSearchTool().execute({})).is_error is True

    def test_web_fetch_returns_canned_page(self) -> None:
        r = _run(EvalWebFetchTool().execute({"url": "https://example.com/docs"}))
        assert r.is_error is False
        assert "https://example.com/docs" in r.content

    def test_web_fetch_without_url_is_an_error(self) -> None:
        assert _run(EvalWebFetchTool().execute({})).is_error is True

    def test_no_network_module_is_imported(self) -> None:
        # 离线是硬约束:替身模块不得引入 httpx 之类的客户端.
        import longline.eval.eval_tools as mod

        src = open(mod.__file__, encoding="utf-8").read()  # noqa: SIM115
        for banned in ("httpx", "requests", "urllib.request", "socket"):
            assert banned not in src, f"eval_tools must stay offline, found {banned!r}"


class TestProfiles:
    def test_core_profile_is_the_existing_registry(self) -> None:
        assert build_tool_profile("core") == EVAL_TOOL_NAMES

    def test_web_profile_adds_both_web_tools(self) -> None:
        assert set(WEB_FAMILY) <= set(build_tool_profile("web"))

    def test_all_profile_covers_every_family(self) -> None:
        names = set(build_tool_profile("all"))
        assert set(EVAL_TOOL_NAMES) <= names
        assert set(ALL_EVAL_TOOL_NAMES) <= names

    def test_unknown_profile_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown tool profile"):
            build_tool_profile("nope")

    def test_all_eval_tool_names_lists_every_family(self) -> None:
        assert {"WebSearch", "WebFetch", "NotebookEdit"} <= set(ALL_EVAL_TOOL_NAMES)
        assert {"TaskCreate", "TaskGet", "TaskList", "TaskUpdate", "TaskStop"} <= set(
            ALL_EVAL_TOOL_NAMES
        )


class TestNotebookAndTaskToolsAreTheProductionOnes:
    def test_notebook_tool_name(self) -> None:
        assert NotebookEditTool().get_name() == "NotebookEdit"

    def test_task_tools_share_an_injectable_store(self) -> None:
        # 用例必须能在自己的沙箱里放一个 store,避免任务状态跨用例泄漏.
        from longline.eval.eval_tools import build_eval_registry

        store = TaskStore()
        registry = build_eval_registry("/tmp/sandbox", profile="task", task_store=store)
        created = _run(registry.get("TaskCreate").execute({"subject": "s"}))  # type: ignore[union-attr]
        assert created.is_error is False
        assert [t.subject for t in store.list_all()] == ["s"]


class TestNoToolEscapesTheSandbox:
    """A case must not be able to read or write the repository around it.

    Shipped once: `Glob`/`Grep` defaulted to the PROCESS cwd, which for an eval
    run is the repository root. A model asked about "the sandbox's
    analysis.ipynb" ran `Glob("**/analysis.ipynb")`, was handed an absolute path
    to `evals/fixtures/notebook_repo/analysis.ipynb`, and edited it in place --
    corrupting a tracked fixture and breaking the next case's starting state.
    """

    def _registry(self, sandbox: Path) -> Any:
        from longline.eval.eval_tools import build_eval_registry

        return build_eval_registry(str(sandbox), profile="all")

    def test_glob_searches_the_sandbox_not_the_process_cwd(self, tmp_path: Path) -> None:
        """FAILS ON: the shipped bug -- a glob walking the repo root.

        The outside file is named the same as the fixture that was corrupted, so
        a regression reintroduces the exact failure rather than a lookalike.
        """
        sandbox = tmp_path / "sandbox"
        outside = tmp_path / "outside"
        sandbox.mkdir()
        outside.mkdir()
        (outside / "analysis.ipynb").write_text("{}", encoding="utf-8")

        registry = self._registry(sandbox)
        result = _run(registry.get("Glob").execute({"pattern": "**/*.ipynb"}))  # type: ignore[union-attr]

        assert "analysis.ipynb" not in result.content
        assert str(outside) not in result.content

    def test_glob_finds_a_file_inside_the_sandbox(self, tmp_path: Path) -> None:
        """The confinement must not cost the tool its normal function."""
        sandbox = tmp_path / "sandbox"
        sandbox.mkdir()
        (sandbox / "kept.ipynb").write_text("{}", encoding="utf-8")

        registry = self._registry(sandbox)
        result = _run(registry.get("Glob").execute({"pattern": "**/*.ipynb"}))  # type: ignore[union-attr]

        assert "kept.ipynb" in result.content

    def test_grep_stays_inside_the_sandbox(self, tmp_path: Path) -> None:
        """FAILS ON: a Grep that searches the process cwd for the same reason."""
        sandbox = tmp_path / "sandbox"
        outside = tmp_path / "outside"
        sandbox.mkdir()
        outside.mkdir()
        (outside / "leak.txt").write_text("NEEDLE_OUTSIDE\n", encoding="utf-8")

        registry = self._registry(sandbox)
        result = _run(registry.get("Grep").execute({"pattern": "NEEDLE_OUTSIDE"}))  # type: ignore[union-attr]

        assert "NEEDLE_OUTSIDE" not in result.content

    @pytest.mark.parametrize(
        "tool,args",
        [
            ("Read", {"file_path": "OUTSIDE"}),
            ("Write", {"file_path": "OUTSIDE", "content": "x"}),
            ("Edit", {"file_path": "OUTSIDE", "old_string": "a", "new_string": "b"}),
            ("Glob", {"pattern": "*", "path": "OUTSIDE"}),
            ("Grep", {"pattern": "x", "path": "OUTSIDE"}),
            ("NotebookEdit", {"notebook_path": "OUTSIDE", "command": "delete_cell", "cell_index": 0}),
        ],
    )
    def test_an_absolute_path_outside_is_refused(
        self, tmp_path: Path, tool: str, args: dict[str, Any]
    ) -> None:
        """Every path-bearing tool, not just the one that was caught."""
        sandbox = tmp_path / "sandbox"
        outside = tmp_path / "outside"
        sandbox.mkdir()
        outside.mkdir()
        target = outside / "target.txt"
        target.write_text("secret\n", encoding="utf-8")
        args = {k: (str(target) if v == "OUTSIDE" else v) for k, v in args.items()}

        registry = self._registry(sandbox)
        result = _run(registry.get(tool).execute(args))  # type: ignore[union-attr]

        assert result.is_error is True, f"{tool} did not refuse an outside path"
        assert "sandbox" in result.content

    def test_the_refusal_leaves_the_outside_file_untouched(self, tmp_path: Path) -> None:
        """A refusal that still writes is worse than no check at all."""
        sandbox = tmp_path / "sandbox"
        outside = tmp_path / "outside"
        sandbox.mkdir()
        outside.mkdir()
        target = outside / "keep.txt"
        target.write_text("original\n", encoding="utf-8")

        registry = self._registry(sandbox)
        _run(registry.get("Write").execute({"file_path": str(target), "content": "CLOBBERED"}))  # type: ignore[union-attr]

        assert target.read_text(encoding="utf-8") == "original\n"

    def test_a_relative_path_resolves_against_the_sandbox(self, tmp_path: Path) -> None:
        """The sandbox IS the working directory the system prompt declares."""
        sandbox = tmp_path / "sandbox"
        sandbox.mkdir()
        (sandbox / "notes.md").write_text("hello\n", encoding="utf-8")

        registry = self._registry(sandbox)
        result = _run(registry.get("Read").execute({"file_path": "notes.md"}))  # type: ignore[union-attr]

        assert result.is_error is False
        assert "hello" in result.content

    def test_an_absolute_path_inside_is_allowed(self, tmp_path: Path) -> None:
        sandbox = tmp_path / "sandbox"
        sandbox.mkdir()
        (sandbox / "notes.md").write_text("hello\n", encoding="utf-8")

        registry = self._registry(sandbox)
        result = _run(registry.get("Read").execute({"file_path": str(sandbox / "notes.md")}))  # type: ignore[union-attr]

        assert result.is_error is False

    def test_the_model_still_sees_the_production_tool(self, tmp_path: Path) -> None:
        """Confining `execute` must not change the menu the model is shown.

        A wrapper that altered `get_schema` would silently change which tool the
        model believes it is choosing, and the selection rate would no longer be
        a statement about the production tools.
        """
        from longline.eval.eval_tools import build_eval_registry

        sandbox = tmp_path / "sandbox"
        sandbox.mkdir()
        wrapped = build_eval_registry(str(sandbox), profile="all")

        for name, production in (
            ("Read", FileReadTool()),
            ("Glob", GlobTool()),
            ("Grep", GrepTool()),
        ):
            got = wrapped.get(name).get_schema()  # type: ignore[union-attr]
            want = production.get_schema()
            assert got.name == want.name, name
            assert got.input_schema == want.input_schema, name


# --- workload / reconcile forwarding ----------------------------------------
#
# The wrappers an eval registry installs are the tools the journal actually asks
# about, and `Tool.workload` defaults to `{}` and `Tool.reconcile` to UNKNOWN. A
# wrapper that did not forward would declare nothing and verify nothing for every
# tool it wraps -- silently emptying the journal's digests, which is what
# reconciliation AND the workspace identity both read. The suite would then report
# a clean recovery for a runtime that had been blinded, and no number would say
# so. These tests exist because that is exactly what happened once.


def test_the_sandboxed_tool_resolves_the_same_path_execute_does(tmp_path: Path) -> None:
    """Not a pass-through: `execute` REWRITES the argument before delegating.

    A pass-through would digest a different file from the one the tool touches.
    """
    from longline.eval.eval_tools import SandboxedTool
    from longline.tools.file_read.file_read_tool import FileReadTool

    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    wrapper = SandboxedTool(FileReadTool(), str(sandbox), "file_path")

    declared = wrapper.workload({"file_path": "src/calc.py"})
    assert declared == {str((sandbox / "src" / "calc.py").resolve()): "read"}


def test_the_sandboxed_tool_declares_nothing_for_a_refused_path(tmp_path: Path) -> None:
    """A call that will be refused touches nothing inside the sandbox."""
    from longline.eval.eval_tools import SandboxedTool
    from longline.tools.file_write.file_write_tool import FileWriteTool

    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    wrapper = SandboxedTool(FileWriteTool(), str(sandbox), "file_path")
    assert wrapper.workload({"file_path": str(tmp_path / "outside.txt"), "content": "x"}) == {}


def test_the_sandboxed_tool_forwards_reconcile(tmp_path: Path) -> None:
    from longline.eval.eval_tools import SandboxedTool
    from longline.tools.base import ReconcileOutcome
    from longline.tools.file_write.file_write_tool import FileWriteTool

    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    (sandbox / "a.txt").write_text("written\n", encoding="utf-8")
    wrapper = SandboxedTool(FileWriteTool(), str(sandbox), "file_path")

    outcome = wrapper.reconcile({"file_path": "a.txt", "content": "written\n"})
    assert outcome is ReconcileOutcome.APPLIED


def test_the_gated_tool_forwards_workload_and_reconcile(tmp_path: Path) -> None:
    """`GatedTool` is what the loop-resume suite swaps in for every tool.

    It subclasses `Tool`, so it inherits the two defaults -- which is why it has
    to override them explicitly rather than relying on the inner tool.
    """
    from longline.eval.failpoints import BEFORE_TOOL, FailpointGate, GatedTool
    from longline.tools.base import ReconcileOutcome
    from longline.tools.file_write.file_write_tool import FileWriteTool

    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    (sandbox / "a.txt").write_text("written\n", encoding="utf-8")
    gate = FailpointGate(claude_dir=tmp_path, failpoint=BEFORE_TOOL, armed=False)
    gated = GatedTool(inner=FileWriteTool(), gate=gate)

    target = str((sandbox / "a.txt").resolve())
    assert gated.workload({"file_path": str(sandbox / "a.txt"), "content": "x"}) == {
        target: "write"
    }
    assert gated.reconcile({"file_path": str(sandbox / "a.txt"), "content": "written\n"}) is (
        ReconcileOutcome.APPLIED
    )
