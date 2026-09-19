import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.shapes import area_rect, perimeter_rect


def test_area_rect() -> None:
    assert area_rect(<seed_a>, <seed_b>) == <seed_a> * <seed_b>


def test_perimeter_rect() -> None:
    assert perimeter_rect(<seed_a>, <seed_b>) == 2 * (<seed_a> + <seed_b>)
