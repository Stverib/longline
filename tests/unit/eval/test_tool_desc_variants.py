"""The description-variant scaffold must not change the baseline arm.

An ablation whose control arm is also perturbed measures nothing, so the most
important test here is `test_baseline_registry_serves_the_production_text`: the
wrapper has to be present in BOTH arms and be a no-op in one of them.
"""

from __future__ import annotations

import pytest

from longline.eval.eval_tools import ALL_EVAL_TOOL_NAMES, build_eval_registry
from longline.eval.tool_desc_variants import (
    NOTEBOOK_STEERED,
    STEERED_DESCRIPTIONS,
    DescriptionVariantTool,
    steered_tool_names,
)
from longline.tools.bash.bash_tool import BashTool


class TestVariantSelection:
    def test_baseline_steers_nothing(self) -> None:
        assert steered_tool_names("baseline") == frozenset()

    def test_steered_covers_every_tool_it_declares_a_description_for(self) -> None:
        assert steered_tool_names("steered") == frozenset(STEERED_DESCRIPTIONS)

    def test_the_notebook_variant_steers_exactly_the_substitution_pair(self) -> None:
        """The measured cause is Edit replacing NotebookEdit, so the arm that
        isolates it must move those two descriptions and nothing else."""
        assert steered_tool_names("notebook") == NOTEBOOK_STEERED
        assert set(NOTEBOOK_STEERED) == {"Edit", "NotebookEdit"}

    def test_an_unknown_variant_raises(self) -> None:
        """A typo'd flag that silently produced two identical arms would look
        like a clean null result."""
        with pytest.raises(ValueError, match="unknown tool description variant"):
            steered_tool_names("steerd")

    def test_every_steered_name_is_one_the_registry_can_serve(self) -> None:
        """A typo'd key would silently steer nothing."""
        assert set(STEERED_DESCRIPTIONS) <= set(ALL_EVAL_TOOL_NAMES)
        assert set(NOTEBOOK_STEERED) <= set(ALL_EVAL_TOOL_NAMES)


class TestTheWrapper:
    def test_replaces_only_the_description(self) -> None:
        inner = BashTool(cwd=".")
        before = inner.get_schema()
        after = DescriptionVariantTool(inner, "STEERED TEXT").get_schema()

        assert after.description == "STEERED TEXT"
        assert after.name == before.name
        assert after.input_schema == before.input_schema

    def test_delegates_the_name_and_the_concurrency_check(self) -> None:
        inner = BashTool(cwd=".")
        tool = DescriptionVariantTool(inner, "x")

        assert tool.get_name() == inner.get_name()
        assert tool.is_concurrency_safe({"command": "ls"}) == inner.is_concurrency_safe(
            {"command": "ls"}
        )

    async def test_delegates_execution(self) -> None:
        inner = BashTool(cwd=".")
        tool = DescriptionVariantTool(inner, "x")

        result = await tool.execute({"command": "echo hi"})

        assert "hi" in result.text


class TestTheRegistry:
    def test_baseline_serves_the_production_text(self) -> None:
        """The control arm must be untouched, or the A/B compares two changes."""
        registry = build_eval_registry(".", profile="core", tool_desc_variant="baseline")

        assert (
            registry.get("Bash").get_schema().description
            == BashTool(cwd=".").get_schema().description
        )

    def test_steered_serves_the_steered_text(self) -> None:
        registry = build_eval_registry(".", profile="core", tool_desc_variant="steered")

        assert (
            registry.get("Bash").get_schema().description
            == STEERED_DESCRIPTIONS["Bash"]
        )

    def test_steering_changes_no_schema_but_the_description(self) -> None:
        baseline = build_eval_registry(".", profile="core", tool_desc_variant="baseline")
        steered = build_eval_registry(".", profile="core", tool_desc_variant="steered")

        for name in ("Bash", "Read", "Grep", "Glob", "Edit"):
            before = baseline.get(name).get_schema()
            after = steered.get(name).get_schema()
            assert before.input_schema == after.input_schema, name
            assert before.description != after.description, name

    def test_steering_hides_no_tool(self) -> None:
        """The registry still offers every tool in the profile. This is a
        wording experiment, never a visibility one: hiding a tool would make
        the evaluator do half the routing the metric is supposed to measure.

        Compared through `get_api_schemas` because that is the list the model
        actually receives, not the registry's internal one.
        """
        baseline = build_eval_registry(".", profile="all", tool_desc_variant="baseline")
        steered = build_eval_registry(".", profile="all", tool_desc_variant="steered")

        before = [s["name"] for s in baseline.get_api_schemas()]
        after = [s["name"] for s in steered.get_api_schemas()]

        assert before == after
        assert len(before) > 6  # the "all" profile, not just the core set

    def test_notebook_variant_leaves_bash_alone(self) -> None:
        registry = build_eval_registry(
            ".", profile="notebook", tool_desc_variant="notebook",
        )

        assert (
            registry.get("Bash").get_schema().description
            == BashTool(cwd=".").get_schema().description
        )
        assert (
            registry.get("NotebookEdit").get_schema().description
            == STEERED_DESCRIPTIONS["NotebookEdit"]
        )
        assert (
            registry.get("Edit").get_schema().description
            == STEERED_DESCRIPTIONS["Edit"]
        )
