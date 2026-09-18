"""Tests for system prompt sections.

Verifies T3.1: Prompt text presence and key phrase checks.
"""

from longline.prompts.sections import (
    get_actions_section,
    get_doing_tasks_section,
    get_intro_section,
    get_minimal_tool_use_section,
    get_output_efficiency_section,
    get_system_section,
    get_tone_style_section,
    get_using_tools_section,
)


class TestPromptSections:
    def test_intro_contains_claude_code(self) -> None:
        text = get_intro_section()
        assert len(text) > 0
        assert "software engineering" in text.lower() or "interactive agent" in text.lower()

    def test_system_section_mentions_tools(self) -> None:
        text = get_system_section()
        assert "tool" in text.lower()

    def test_doing_tasks_mentions_security(self) -> None:
        text = get_doing_tasks_section()
        assert "security" in text.lower()

    def test_actions_mentions_reversibility(self) -> None:
        text = get_actions_section()
        assert "reversibility" in text.lower()

    def test_using_tools_mentions_read_and_bash(self) -> None:
        text = get_using_tools_section()
        assert "Read" in text
        assert "Bash" in text

    def test_tone_style_mentions_emoji(self) -> None:
        text = get_tone_style_section()
        assert "emoji" in text.lower()

    def test_minimal_tool_use_section_exists(self) -> None:
        text = get_minimal_tool_use_section()
        assert "enough" in text
        assert "Read" in text and "Edit" in text

    def test_minimal_tool_use_mentions_sufficiency_not_just_fewer_calls(self) -> None:
        text = get_minimal_tool_use_section()
        assert "# Minimal tool use" in text
        assert "act" in text

    def test_output_efficiency_not_empty(self) -> None:
        text = get_output_efficiency_section()
        assert len(text) > 100

    def test_all_sections_are_strings(self) -> None:
        sections = [
            get_intro_section(),
            get_system_section(),
            get_doing_tasks_section(),
            get_actions_section(),
            get_using_tools_section(),
            get_tone_style_section(),
            get_output_efficiency_section(),
        ]
        for s in sections:
            assert isinstance(s, str)
            assert len(s) > 50

    def test_total_prompt_length(self) -> None:
        total = sum(len(s) for s in [
            get_intro_section(),
            get_system_section(),
            get_doing_tasks_section(),
            get_actions_section(),
            get_using_tools_section(),
            get_tone_style_section(),
            get_output_efficiency_section(),
        ])
        assert total > 5000


def test_retrieval_policy_is_in_the_prompt():
    from longline.prompts.sections import get_retrieval_policy_section
    text = get_retrieval_policy_section()
    assert "Grep" in text and "Glob" in text
    assert "lines" in text or "range" in text   # hit -> read around the hit


def test_builder_wires_retrieval_after_minimal():
    from longline.prompts.builder import build_system_prompt
    sections = build_system_prompt(cwd="/tmp", model="m")
    joined = "\n".join(sections)
    assert "# Retrieval policy" in joined
    assert joined.index("# Retrieval policy") > joined.index("# Minimal tool use")
