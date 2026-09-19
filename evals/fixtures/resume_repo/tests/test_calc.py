import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.calc import add


def test_add() -> None:
    assert add(2, 3) == 5
