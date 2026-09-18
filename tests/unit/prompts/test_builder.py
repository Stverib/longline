"""Tests for system prompt builder.

Verifies T3.2: Dynamic prompt assembly.
"""

from longline.prompts.builder import build_system_prompt, compute_env_info


class TestComputeEnvInfo:
    def test_contains_cwd(self) -> None:
        info = compute_env_info("/tmp/test", "claude-sonnet-4-20250514")
        assert "/tmp/test" in info

    def test_contains_model(self) -> None:
        info = compute_env_info("/tmp", "claude-sonnet-4-20250514")
        assert "claude-sonnet-4-20250514" in info

    def test_contains_platform(self) -> None:
        info = compute_env_info("/tmp", "test-model")
        assert "Platform" in info


class TestBuildSystemPrompt:
    def test_returns_list_of_strings(self) -> None:
        result = build_system_prompt("/tmp", "claude-sonnet-4-20250514")
        assert isinstance(result, list)
        assert all(isinstance(s, str) for s in result)
        assert len(result) >= 7  # At least the static sections + env

    def test_contains_static_sections(self) -> None:
        result = build_system_prompt("/tmp", "claude-sonnet-4-20250514")
        joined = "\n".join(result)
        assert "interactive agent" in joined.lower()
        assert "Bash" in joined
        assert "emoji" in joined.lower()
        assert "# Minimal tool use" in joined
        assert "# Retrieval policy" in joined

    def test_contains_dynamic_env(self) -> None:
        result = build_system_prompt("/tmp/myproject", "test-model")
        joined = "\n".join(result)
        assert "/tmp/myproject" in joined
        assert "test-model" in joined

    def test_claude_md_injected(self) -> None:
        result = build_system_prompt("/tmp", "m", claude_md_content="Always use tabs.")
        joined = "\n".join(result)
        assert "Always use tabs." in joined
        assert "CLAUDE.md" in joined

    def test_no_claude_md_section_when_none(self) -> None:
        result = build_system_prompt("/tmp", "m")
        joined = "\n".join(result)
        assert "Codebase and user instructions" not in joined

    def test_memory_prompt_injected(self) -> None:
        """Memory prompt includes full behavioral instructions when memory_dir is set."""
        result = build_system_prompt(
            "/tmp", "m",
            memory_dir="/home/user/.claude/projects/abc123/memory",
        )
        joined = "\n".join(result)
        # Should have the full memory instructions, not just raw content
        assert "# auto memory" in joined
        assert "persistent, file-based memory system" in joined
        assert "## Types of memory" in joined
        assert "## What NOT to save" in joined
        assert "## How to save memories" in joined
        assert "## When to access memories" in joined
        assert "## Before recommending from memory" in joined
        assert "/home/user/.claude/projects/abc123/memory" in joined

    def test_memory_prompt_with_index(self) -> None:
        """MEMORY.md content is embedded in the memory prompt section."""
        result = build_system_prompt(
            "/tmp", "m",
            memory_dir="/tmp/memory",
            memory_index_content="- [User role](user_role.md) — data scientist",
        )
        joined = "\n".join(result)
        assert "## MEMORY.md" in joined
        assert "data scientist" in joined

    def test_memory_prompt_empty_index(self) -> None:
        """Empty MEMORY.md shows helpful placeholder."""
        result = build_system_prompt(
            "/tmp", "m",
            memory_dir="/tmp/memory",
            memory_index_content=None,
        )
        joined = "\n".join(result)
        assert "currently empty" in joined

    def test_no_memory_when_no_dir(self) -> None:
        result = build_system_prompt("/tmp", "m")
        joined = "\n".join(result)
        assert "auto memory" not in joined

    def test_memory_and_claude_md_both_injected(self) -> None:
        result = build_system_prompt(
            "/tmp", "m",
            claude_md_content="Use black.",
            memory_dir="/tmp/memory",
            memory_index_content="- [Feedback](feedback.md) — prefers ruff",
        )
        joined = "\n".join(result)
        assert "Use black" in joined
        assert "prefers ruff" in joined
        assert "# auto memory" in joined


class TestPromptVariant:
    """Which optional sections the ablation assembles.

    The retrieval policy paragraph did not move the e2e pass rate last round,
    but that category's failures were deterministic (three cases failed 0/3),
    so the pass rate had no room to move. An A/B needs the variant to be
    switchable and provably the ONLY difference.
    """

    def test_baseline_carries_the_retrieval_policy(self) -> None:
        joined = "\n".join(build_system_prompt("/tmp", "m"))

        assert "# Retrieval policy" in joined

    def test_default_is_baseline(self) -> None:
        assert build_system_prompt("/tmp", "m") == build_system_prompt(
            "/tmp", "m", prompt_variant="baseline",
        )

    def test_no_retrieval_drops_exactly_that_section(self) -> None:
        joined = "\n".join(build_system_prompt("/tmp", "m", prompt_variant="no-retrieval"))

        assert "# Retrieval policy" not in joined
        # The ablation is clean only if the OTHER sections are untouched.
        assert "# Minimal tool use" in joined
        assert "# Using your tools" in joined
        assert "# Tone and style" in joined

    def test_no_retrieval_keeps_every_other_section(self) -> None:
        baseline = build_system_prompt("/tmp", "m")
        without = build_system_prompt("/tmp", "m", prompt_variant="no-retrieval")

        assert len(without) == len(baseline) - 1
        assert [s for s in baseline if s not in without] == [
            s for s in baseline if "Retrieval policy" in s
        ]

    def test_an_unknown_variant_raises(self) -> None:
        """A typo must not silently produce the baseline prompt and a
        plausible-looking null result in which both arms were the same arm."""
        import pytest

        with pytest.raises(ValueError, match="unknown prompt variant"):
            build_system_prompt("/tmp", "m", prompt_variant="no-retreival")
