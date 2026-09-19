import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.units import to_grams


def test_to_grams() -> None:
    assert to_grams(<seed_a>) == <seed_a> * 1000
