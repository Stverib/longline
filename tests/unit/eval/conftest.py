"""Pytest configuration for the eval unit tests.

Exists for one reason: to register and deselect the `slow_idle_host` marker.
Both halves have to live in a conftest, because the marker is declared in
`pyproject.toml`'s `[tool.pytest.ini_options]` and that file is outside this
task's write scope -- but a conftest can register a marker and filter on it
without touching project config.

=== What `slow_idle_host` means ===

A test so marked is (a) slow and (b) **only valid on a machine that is not
otherwise loaded**. That combination is not an accident of how the test was
written; it is a property of the streaming-latency benchmark itself, whose truth
check sizes its tolerance against the host's measured jitter. See
`longline/eval/latency_runner.py`'s `DEFAULT_TIME_SCALE` for the measurements.

Two real consequences, which is why this is enforced rather than documented:

- Running such a test while anything else saturates the box inflates the jitter
  past the tolerance and makes the gate reject CORRECT runs. Measured: the same
  test passed in 220 s on an idle host and failed at exactly 100.0 ms of error
  against a 100.0 ms tolerance while another test run shared the machine.
- A 220 s test in the DEFAULT run is a hazard in its own right. Two agents have
  already been lost on this task to blocking on a long-running command.

Described rather than run by default. To run them:

    uv run --extra dev pytest tests/unit/eval -q --slow-idle-host

or select them directly:

    uv run --extra dev pytest tests/unit/eval -q -m slow_idle_host
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator

SLOW_IDLE_HOST = "slow_idle_host"


def pytest_addoption(parser: pytest.Parser) -> None:
    """`--slow-idle-host` opts IN to the load-sensitive tests.

    Framed as opt-in rather than opt-out because the failure mode of running
    them accidentally is a false red -- a correct run reported as broken -- and
    that is the expensive direction to be wrong in.
    """
    parser.addoption(
        "--slow-idle-host",
        action="store_true",
        default=False,
        help=(
            "Also run tests marked slow_idle_host. They need an idle machine: "
            "the latency benchmark's truth check is sized against the host's "
            "jitter, so a loaded box makes them fail on correct runs."
        ),
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        f"{SLOW_IDLE_HOST}: slow AND only valid on an unloaded host "
        "(streaming-latency benchmark; needs --slow-idle-host)",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item],
) -> None:
    """Deselect the marked tests unless `--slow-idle-host` was passed.

    Explicit `-m slow_idle_host` also runs them: a caller who asked for them by
    marker has already opted in, and silently deselecting the very tests they
    named would be its own kind of wrong.
    """
    if config.getoption("--slow-idle-host"):
        return
    if config.getoption("-m", default=""):
        # An explicit marker expression is the caller's decision. Only a
        # `-m` that does NOT name this marker is treated as "not asked for".
        expression: str = config.getoption("-m")
        if SLOW_IDLE_HOST in expression:
            return

    skip = pytest.mark.skip(
        reason=f"{SLOW_IDLE_HOST}: needs an idle host; pass --slow-idle-host to run",
    )
    for item in items:
        if SLOW_IDLE_HOST in item.keywords:
            item.add_marker(skip)


def pytest_report_header(config: pytest.Config) -> Iterator[str] | str:
    """Say which mode the run is in, so a green result is not misread."""
    if config.getoption("--slow-idle-host"):
        return (
            "slow_idle_host: ENABLED -- these tests time real sleeps and need an "
            "idle machine; results under load are not meaningful"
        )
    return "slow_idle_host: deselected (pass --slow-idle-host to include)"
