import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.generated import LIMIT


def test_limit() -> None:
    assert LIMIT == 5
