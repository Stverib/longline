import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.alpha import offset, scale


def test_scale() -> None:
    assert scale(<seed_a>) == <seed_a> * 2


def test_offset() -> None:
    assert offset(<seed_a>) == <seed_a> + 1
