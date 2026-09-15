"""End-to-end CLI tests.

Verifies T9.1: CLI --print mode with real API, and --help.

W4: Online tests now detect network failures (subprocess timeout or
connection error in output) and skip instead of fail.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).parent.parent.parent


def get_api_key() -> str | None:
    import os

    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    env_file = PROJECT_DIR / ".env"
    if env_file.is_file():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("ANTHROPIC_API_KEY="):
                return line.split("=", 1)[1].strip()
    return None


class TestCLIHelp:
    def test_help(self) -> None:
        """python -m longline --help shows help."""
        result = subprocess.run(
            [sys.executable, "-m", "longline", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(PROJECT_DIR),
        )
        assert result.returncode == 0
        assert "longline" in result.stdout


skip_no_key = pytest.mark.skipif(get_api_key() is None, reason="No API key")


@skip_no_key
class TestWelcomeBanner:
    def test_banner_branding(self) -> None:
        """The REPL welcome banner announces the product by name.

        print_welcome() only runs in the interactive REPL branch, so
        `--help` (short-circuited by Click) never reaches it. This test
        exercises the real REPL until the banner appears, then quits.

        Covers a gap: no other test asserts on the banner, so a stale
        brand string there would otherwise go unnoticed.
        """
        proc = subprocess.Popen(
            [sys.executable, "-m", "longline"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(PROJECT_DIR),
            encoding="utf-8",
            errors="replace",
        )
        try:
            out, _ = proc.communicate(input="/exit\n", timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            out, _ = proc.communicate()
            pytest.skip("REPL did not exit within timeout — likely no network access")
        finally:
            if proc.poll() is None:
                proc.kill()

        assert "longline v" in out, f"banner not found in REPL output:\n{out[:500]}"


skip_no_key = pytest.mark.skipif(get_api_key() is None, reason="No API key")


@skip_no_key
class TestCLIPrintMode:
    def test_print_mode(self) -> None:
        """python -m longline -p 'prompt' produces output.

        W4: Detects network failure (timeout or connection error) and
        skips instead of failing.
        """
        import os

        env = {**os.environ, "ANTHROPIC_API_KEY": get_api_key() or ""}
        try:
            result = subprocess.run(
                [sys.executable, "-m", "longline", "-p", "Say exactly one word: hello"],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(PROJECT_DIR),
                env=env,
            )
        except subprocess.TimeoutExpired:
            pytest.skip("CLI timed out — likely no network access to Anthropic API")
            return

        # W4: Check for connection error in stderr
        if "Connection error" in result.stderr:
            pytest.skip(
                f"Network unavailable — skipping CLI test. stderr: {result.stderr[:200]}"
            )

        assert result.returncode == 0, (
            f"CLI failed with code {result.returncode}. "
            f"stdout: {result.stdout[:200]}, stderr: {result.stderr[:200]}"
        )
        assert len(result.stdout.strip()) > 0
