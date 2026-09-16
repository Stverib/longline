# projectkit

A small sample package used by the agent evaluation suite.

## Layout

| Path | Purpose |
| --- | --- |
| `src/projectkit/stats.py` | numeric helpers |
| `src/projectkit/text.py` | string helpers |
| `src/projectkit/report.py` | rendering helpers |
| `tests/` | test suite |

## Running the tests

    python -m pytest -q

`pytest` is available; the tests import the package from `src/`, so run them
from the repository root with `PYTHONPATH=src`.
